"""Graph nodes and routing functions.

A node takes the state (and optionally the runtime) and returns a dict with
only the keys it changes. A routing function returns the name of the next
node. Everything marked TODO is yours to write; the stubs are deliberately
conservative so the graph runs end to end before any of it is done.

Log with `log.info(...)` anywhere in here: it shows on the console with -v
and, when tracing, as its own step under this node in the LangSmith trace.
"""
import hashlib
import logging
import random
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from gmail_assist.agent.labels import (
    EXAMPLES, GUIDANCE, ME_LABEL, OWNER_NAME, automated, choices, is_parent, parent,
)
from gmail_assist.agent.llm import LLM, Classification, Models
from gmail_assist.agent.state import Context, EmailState

log = logging.getLogger(__name__)

# Small local models do worse with very long inputs; trim bodies to this.
MAX_BODY_CHARS = 4000
# Tracking links can fill the whole budget; only their domain says anything.
URL = re.compile(r'https?://([^/\s\]>)?]+)[^\s\]>)]*')

LOCAL_MODEL = 'local:gemma-3-4b-it'
STRONG_MODEL = 'anthropic:claude-opus-5-5'
STRONG_EFFORT = 'medium'
# The local model answers with a letter LOCAL_VOTES times, the options
# shuffled each time; its confidence is the share of votes that agree. Its
# own probabilities aren't used: they're ~1.0 even when it's wrong.
LOCAL_VOTES = 3
LOCAL_MIN_AGREEMENT = 1.0    # unanimous, or escalate
# The strong model reports its own confidence; below this it goes to review.
MIN_CONFIDENCE = 0.7
NONE = 'NONE'                # "no label fits"

# Known senders: automated mail skips the models when the sender's last
# SENDER_HISTORY emails (at least SENDER_MIN of them) all got the same label.
# One in SENDER_SPOT_CHECK still goes to a model, so a sender that starts
# sending something else is noticed: a different answer breaks the streak.
SENDER_MIN = 3
SENDER_HISTORY = 10
SENDER_SPOT_CHECK = 20

# A parent-label answer gets a second look for a better or new sub-label.
REFINE_EFFORT = 'low'
REFINE_RECENT = 15           # other emails on the parent shown for context


# -- routing ----------------------------------------------------------------

def route_by_kind(state: EmailState) -> Literal['detect_login_code', 'review_subscription']:
    """Personal mail is checked for login codes; bulk mail for its list's history."""
    return 'detect_login_code' if state['kind'] == 'personal' else 'review_subscription'


def after_login_check(state: EmailState) -> Literal['check_thread', '__end__']:
    """A login code is fully decided ('expire'); anything else gets labelled."""
    return END if state.get('is_login_code') else 'check_thread'


def after_thread_check(state: EmailState) -> Literal['check_sender', '__end__']:
    """A reply in an already-labelled thread inherits that label."""
    return END if state.get('inherited') else 'check_sender'


def after_sender_check(state: EmailState) -> Literal['choose_model', '__end__']:
    """Mail from a sender that's always labelled the same way needs no model."""
    return END if state.get('known_sender') else 'choose_model'


def acceptable(attempt: dict, labels: list[str]) -> Literal['ok', 'none', 'invalid', 'low_confidence']:
    """Whether a classify attempt is good enough to act on."""
    c = attempt['classification']
    if c['label'] == NONE:
        return 'none'
    if c['label'] not in labels:
        return 'invalid'
    floor = LOCAL_MIN_AGREEMENT if attempt['model'] == LOCAL_MODEL else MIN_CONFIDENCE
    return 'ok' if c['confidence'] >= floor else 'low_confidence'


def after_classify(state: EmailState, runtime: Runtime[Context]
                   ) -> Literal['decide', 'refine_parent', 'escalate', 'flag_review']:
    """Act on a good answer (a parent label first gets a look for a better
    sub-label); otherwise retry on the strong model once, then hand it to
    you. Escalating only when STRONG_MODEL hasn't run yet means this can't
    loop."""
    ctx = runtime.context
    attempts = state['attempts']
    last = attempts[-1]
    result = acceptable(last, list(ctx.definitions))
    if result == 'ok':
        if is_parent(last['classification']['label'], ctx.definitions) and Models.configured('anthropic'):
            return 'refine_parent'
        return 'decide'
    reason = last['classification']['reason']
    if STRONG_MODEL not in {a['model'] for a in attempts}:
        log.info('%s: %s (%s): escalating to %s', last['model'], result, reason, STRONG_MODEL)
        return 'escalate'
    log.info('%s: %s (%s) after escalation: flagging for review', last['model'], result, reason)
    return 'flag_review'


# -- rules ----------------------------------------------------------------------

def tag_me(state: EmailState) -> dict:
    """Mail you sent gets the Me label on top of its topic label."""
    if OWNER_NAME in (state.get('sender_email') or ''):
        return {'decisions': [{
            'action': 'label', 'label': ME_LABEL, 'decided_by': 'rule:me',
            'reason': f'sender address contains {OWNER_NAME!r}',
        }]}
    return {}


def check_thread(state: EmailState, runtime: Runtime[Context]) -> dict:
    """Thread labels are fixed: reuse one another message here already has."""
    ctx = runtime.context
    prior = ctx.store.thread_label(ctx.account, state['thread_id'], exclude_id=state['message_id'])
    if prior is None:
        return {'inherited': False}
    log.info('thread already labelled %r (%s)', prior.label, prior.decided_by)
    return {'inherited': True, 'decisions': [{
        'action': 'label', 'label': prior.label, 'decided_by': 'thread',
        'confidence': prior.confidence, 'model': prior.model,
        'reason': f'thread already labelled by {prior.decided_by}',
    }]}


def sender_streak(history: list, definitions: dict[str, str]) -> str | None:
    """The label a sender's mail always gets, from its latest decisions
    (Store.sender_history, newest first), or None if it varies or there's
    too little history."""
    labels = {d.label if d.action == 'label' else None for d in history[:SENDER_HISTORY]}
    if len(history) < SENDER_MIN or len(labels) != 1:
        return None
    label = labels.pop()
    # A parent label is where unsorted mail waits for a sub-label: keep
    # sending it through, so refine_parent sees it.
    if label is None or label not in definitions or is_parent(label, definitions):
        return None
    return label


def check_sender(state: EmailState, runtime: Runtime[Context]) -> dict:
    """Label automated mail the way its sender's mail always has been.
    Personal mail always goes to a model: people write about anything."""
    ctx = runtime.context
    sender = state.get('sender_email')
    if not sender or not automated(state['kind'], sender):
        return {'known_sender': False}
    history = ctx.store.sender_history(ctx.account, sender, limit=SENDER_HISTORY)
    label = sender_streak(history, ctx.definitions)
    if label is None:
        return {'known_sender': False}
    digest = hashlib.sha256(state['message_id'].encode()).digest()
    if int.from_bytes(digest[:4], 'big') % SENDER_SPOT_CHECK == 0:
        log.info('known sender (%s), spot-checking with a model', label)
        return {'known_sender': False}
    return {'known_sender': True, 'decisions': [{
        'action': 'label', 'label': label, 'decided_by': 'sender',
        'reason': f'the last {len(history)} emails from {sender} were all labelled {label}',
    }]}


# -- login codes / sign-in links -----------------------------------------------

LOGIN_SUBJECT = re.compile(
    r'\b(verification code|one[- ]time (pass)?code|security code|sign[- ]?in link|'
    r'login code|your code is)\b',
    re.IGNORECASE,
)


def detect_login_code(state: EmailState) -> dict:
    if LOGIN_SUBJECT.search(state.get('subject') or ''):
        return {
            'is_login_code': True,
            'decisions': [{
                'action': 'expire',
                'decided_by': 'rule:login_code',
                'reason': 'subject looks like a login code or sign-in link',
            }],
        }
    return {'is_login_code': False}


# -- newsletters ----------------------------------------------------------------

# Flagging only queues a list for your review; nothing unsubscribes on its
# own. Only an 'active' list is ever changed (run.py passes
# only_from='active'), so a list you marked 'keep' stays put.
STATS_DAYS = 90
MIN_DAYS_KNOWN = 21          # too new to judge before this
MOSTLY = 0.6                 # share of a list's mail in one Gmail category
PROMO_MIN_RECEIVED = 5       # promo_unread: enough mail to judge...
PROMO_MAX_READ_RATE = 0.10   # ...and almost none of it opened
UPDATES_MIN_PER_WEEK = 3     # noisy_updates: only high-volume notifications
UPDATES_MAX_READ_RATE = 0.10
SLOW_MIN_TIMED_READS = 3     # slow_reader: needs real read times (read_at)
SLOW_MIN_HOURS = 72
SLOW_MAX_READ_RATE = 0.30


def unsubscribe_flag(stats: dict, days_known: float) -> dict | None:
    """The first unsubscribe rule `stats` (from subscription_stats) trips,
    as a SubscriptionFlag, or None to leave the list alone."""
    received = stats['received']
    if not received or days_known < MIN_DAYS_KNOWN or stats['starred']:
        return None
    read_rate = stats['read_rate'] or 0
    summary = (f"opened {stats['read']} of {received} in {STATS_DAYS} days "
               f"(~{stats['per_week']:.1f}/week)")

    if (stats['promotions'] / received >= MOSTLY and received >= PROMO_MIN_RECEIVED
            and read_rate < PROMO_MAX_READ_RATE):
        return {'status': 'flagged', 'by': 'rule:promo_unread',
                'reason': f'promotions you rarely open: {summary}'}

    if (stats['updates'] / received >= MOSTLY and stats['per_week'] >= UPDATES_MIN_PER_WEEK
            and read_rate < UPDATES_MAX_READ_RATE):
        return {'status': 'flagged', 'by': 'rule:noisy_updates',
                'reason': f'high-volume notifications you rarely open: {summary}'}

    hours = stats['avg_hours_to_read']
    if (stats['timed_reads'] >= SLOW_MIN_TIMED_READS and hours is not None
            and hours >= SLOW_MIN_HOURS and read_rate < SLOW_MAX_READ_RATE):
        return {'status': 'flagged', 'by': 'rule:slow_reader',
                'reason': f'opened ~{hours / 24:.0f} days after arrival on average: {summary}'}

    return None


def review_subscription(state: EmailState, runtime: Runtime[Context]) -> dict:
    """Look at the list's history and decide whether to flag it for unsubscribing."""
    ctx = runtime.context
    rows = ctx.store.subscription_stats(ctx.account, days=STATS_DAYS, key=state['subscription_key'])
    if not rows:
        return {'subscription_stats': None}
    row = rows[0]
    stats = {k: row[k] for k in (
        'received', 'read', 'read_rate', 'per_week', 'promotions', 'updates',
        'starred', 'timed_reads', 'avg_hours_to_read',
    )}
    # SQLite hands datetimes back without a zone; they're stored as UTC.
    first_seen = row['subscription'].first_seen.replace(tzinfo=timezone.utc)
    days_known = (datetime.now(timezone.utc) - first_seen).total_seconds() / 86400

    flag = unsubscribe_flag(stats, days_known)
    if flag:
        log.info('flagging %s: %s', state['subscription_key'], flag['reason'])
    return {'subscription_stats': stats, 'subscription_flag': flag}


# -- model selection ----------------------------------------------------------------

def choose_model(state: EmailState) -> dict:
    """Every message starts on the local model."""
    return {'model': LOCAL_MODEL, 'effort': None}


def escalate(state: EmailState) -> dict:
    """Switch to the strong model for a second classify pass."""
    return {'model': STRONG_MODEL, 'effort': STRONG_EFFORT}


# -- classification ---------------------------------------------------------------

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXY'   # Z is reserved for "none fits"


def system_prompt(options: list[str], definitions: dict[str, str]) -> str:
    """Guidance, definitions and examples. Identical for every message of a
    kind, so the local server can reuse its cached processing of it."""
    lines = ['You sort emails into Gmail labels.', '', GUIDANCE, '', 'Labels:']
    lines += [f'- {label}: {definitions[label]}' for label in options]
    examples = [(label, ex) for label in options for ex in EXAMPLES.get(label, [])]
    if examples:
        lines += ['', 'Examples:']
        lines += [f'- From: {sender} | Subject: {subject} -> {label}' for label, (sender, subject) in examples]
    return '\n'.join(lines)


def render_email(state: EmailState) -> str:
    """The email as the model sees it."""
    body = URL.sub(r'<\1 link>', state.get('body') or '')[:MAX_BODY_CHARS]
    parts = [
        f"From: {state.get('sender') or ''}",
        f"Subject: {state.get('subject') or ''}",
        f"Kind: {'newsletter / bulk mail' if state['kind'] == 'bulk' else 'personal'}",
    ]
    # Bulk mail has no body, only Gmail's snippet.
    parts.append(f"\n{body}" if body else f"Snippet: {state.get('snippet') or ''}")
    return '\n'.join(parts)


def _vote(llm: LLM, state: EmailState, options: list[str], definitions: dict[str, str]) -> dict:
    """Ask for a letter LOCAL_VOTES times, options shuffled each time."""
    # Seeded by message, so re-running a message asks the same questions.
    rng = random.Random(state['message_id'])
    system = SystemMessage(system_prompt(options, definitions))
    email = render_email(state)
    votes = []
    for _ in range(LOCAL_VOTES):
        order = options[:]
        rng.shuffle(order)
        letters = dict(zip(LETTERS, order)) | {'Z': NONE}
        menu = '\n'.join(f'{letter}) {label}' for letter, label in letters.items())
        answer, _ = llm.choose([system, HumanMessage(
            f'{email}\n\nWhich label fits best? Answer with only the letter.\n{menu}'
            f'\n(Z means none of these fit.)'
        )])
        votes.append(letters.get(answer[:1].upper(), 'INVALID'))
    winner, count = Counter(votes).most_common(1)[0]
    return {
        'model': llm.name, 'effort': None, 'votes': votes,
        'classification': {
            'label': winner, 'confidence': count / len(votes),
            'reason': f'{count}/{len(votes)} votes agree' + (
                '' if count == len(votes) else f": {', '.join(votes)}"),
        },
    }


def _ask(llm: LLM, state: EmailState, options: list[str], definitions: dict[str, str]) -> dict:
    """One structured answer with a confidence and reason."""
    result: Classification = llm.structured(Classification).invoke([
        SystemMessage(system_prompt(options, definitions)),
        HumanMessage(render_email(state)),
    ])
    return {'model': llm.name, 'effort': state.get('effort'), 'classification': result.model_dump()}


def classify(state: EmailState, runtime: Runtime[Context]) -> dict:
    """Run the chosen model and record the attempt."""
    ctx = runtime.context
    llm = ctx.models.get(state['model'], effort=state.get('effort'))
    options = choices(ctx.definitions, automated(state['kind'], state.get('sender_email')))
    ask = _vote if llm.provider == 'local' else _ask
    return {'attempts': [ask(llm, state, options, ctx.definitions)]}


def _decision(action: str, attempt: dict, reason: str) -> dict:
    c = attempt['classification']
    return {
        'action': action,
        'label': None if c['label'] in (NONE, 'INVALID') else c['label'],
        'confidence': c['confidence'],
        'reason': reason,
        'decided_by': 'llm',
        'model': attempt['model'],
    }


def decide(state: EmailState) -> dict:
    """The last attempt passed acceptable(): label the message with it."""
    last = state['attempts'][-1]
    return {'decisions': [_decision('label', last, last['classification']['reason'])]}


# -- sub-labels ---------------------------------------------------------------------

class Refinement(BaseModel):
    choice: Literal['existing', 'proposal', 'keep'] = Field(
        description='existing: a listed sub-label fits after all. proposal: it belongs to a '
                    'recurring category that deserves its own sub-label (a pending proposal '
                    'or a new one). keep: a one-off; leave it on the parent label.')
    label: str = Field(description='The sub-label (existing or proposed) as "Parent/name"; '
                                   'empty for keep.')
    description: str = Field(description='For a new proposal: a one-sentence definition. '
                                         'Otherwise empty.')
    reason: str = Field(description='One short sentence explaining the choice.')


def refine_prompt(label: str, definitions: dict[str, str], pending: list[dict],
                  rejected: list[str], recent: list) -> str:
    subs = [f'- {name}: {desc}' for name, desc in definitions.items() if parent(name) == label]
    lines = [
        'You maintain a Gmail label scheme. The email below was sorted into the',
        f'top-level label "{label}" because none of its sub-labels seemed to fit.',
        'Decide whether a sub-label should hold it.',
        '',
        'Propose a new sub-label only for a kind of mail that will keep arriving',
        '(several senders or a steady stream), not for a single sender or topic.',
        f'Name it like the existing ones: "{label}/" plus a short lowercase,',
        'hyphenated name. Define it in the same style: what belongs, in general',
        'terms; name an organization only if the category is about that one',
        'organization.',
        '',
        GUIDANCE,
        '',
        f'Existing sub-labels of "{label}":', *(subs or ['(none)']),
    ]
    if pending:
        lines += ['', 'Pending proposals (prefer one of these over a near-duplicate):']
        lines += [f"- {p['proposal'].name}: {p['proposal'].description} ({p['messages']} emails)"
                  for p in pending]
    if rejected:
        lines += ['', 'Rejected before; do not propose again: ' + ', '.join(rejected)]
    if recent:
        lines += ['', f'Other recent emails labelled "{label}":']
        lines += [f'- From: {m.sender_email} | Subject: {m.subject}' for m in recent]
    return '\n'.join(lines)


def refine_parent(state: EmailState, runtime: Runtime[Context]) -> dict:
    """The answer was a parent label: look for an existing sub-label that
    fits, or a recurring category worth a new one. The message keeps the
    parent label unless an existing sub-label fits; a proposal only gathers
    support until it's approved (see Store.decide_proposal)."""
    ctx = runtime.context
    last = state['attempts'][-1]
    label = last['classification']['label']
    decision = _decision('label', last, last['classification']['reason'])
    proposals = [p for p in ctx.store.proposals(ctx.account) if parent(p['proposal'].name) == label]
    pending = [p for p in proposals if p['proposal'].status == 'pending']
    rejected = [p['proposal'].name for p in proposals if p['proposal'].status == 'rejected']
    recent = ctx.store.recent_on_label(ctx.account, label, REFINE_RECENT, exclude_id=state['message_id'])

    llm = ctx.models.get(STRONG_MODEL, effort=REFINE_EFFORT)
    try:
        result: Refinement = llm.structured(Refinement).invoke([
            SystemMessage(refine_prompt(label, ctx.definitions, pending, rejected, recent)),
            HumanMessage(render_email(state)),
        ])
    except Exception as err:
        # The parent label is still a good answer; don't fail the message.
        log.warning('refine_parent failed, keeping %r: %s: %s', label, type(err).__name__, err)
        return {'refinement': None, 'decisions': [decision]}

    name = result.label.strip()
    log.info('refine %r: %s %r (%s)', label, result.choice, name, result.reason)
    if result.choice != 'keep' and parent(name) == label:
        if name in ctx.definitions:
            # An existing sub-label fits (or was "proposed" by mistake): use it.
            decision |= {'label': name, 'model': llm.name, 'reason': result.reason}
        elif result.choice == 'proposal' and name not in rejected:
            decision |= {'proposal': name, 'proposal_description': result.description.strip(),
                         'proposed_by': llm.name,
                         'reason': f"{decision['reason']}; would fit proposed {name}: {result.reason}"}
    return {'refinement': {'model': llm.name, **result.model_dump()}, 'decisions': [decision]}


def flag_review(state: EmailState) -> dict:
    """No model was confident enough: leave it for you, with the best guess."""
    attempts = state['attempts']
    # Prefer an attempt that named a real label over one that said none fit.
    best = max(attempts, key=lambda a: (a['classification']['label'] not in (NONE, 'INVALID'),
                                        a['classification']['confidence']))
    tried = '; '.join(
        f"{a['model']} said {a['classification']['label']!r} at {a['classification']['confidence']:.2f}"
        for a in attempts
    )
    return {'decisions': [_decision('review', best, f'low confidence: {tried}')]}
