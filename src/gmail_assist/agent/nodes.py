"""Graph nodes and routing functions.

A node takes the state (and optionally the runtime) and returns a dict with
only the keys it changes. A routing function returns the name of the next
node. Everything marked TODO is yours to write; the stubs are deliberately
conservative so the graph runs end to end before any of it is done.

Log with `log.info(...)` anywhere in here: it shows on the console with -v
and, when tracing, as its own step under this node in the LangSmith trace.
"""
import logging
import re
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime

from gmail_assist.agent.llm import Classification
from gmail_assist.agent.state import Context, EmailState

log = logging.getLogger(__name__)

# Small local models do worse with very long inputs; trim bodies to this.
MAX_BODY_CHARS = 4000

LOCAL_MODEL = 'local:gemma-3-4b-it'
STRONG_MODEL = 'anthropic:claude-opus-5-5'
STRONG_EFFORT = 'medium'
# TODO: tune once you've seen real scores. Below this, an answer escalates
# (from LOCAL_MODEL) or goes to review (from STRONG_MODEL).
MIN_CONFIDENCE = 0.7


# -- routing ----------------------------------------------------------------

def route_by_kind(state: EmailState) -> Literal['detect_login_code', 'review_subscription']:
    """Personal mail is checked for login codes; bulk mail for its list's history."""
    return 'detect_login_code' if state['kind'] == 'personal' else 'review_subscription'


def after_login_check(state: EmailState) -> Literal['choose_model', '__end__']:
    """A login code is fully decided ('expire'); anything else gets classified."""
    return END if state.get('is_login_code') else 'choose_model'


def acceptable(attempt: dict, labels: list[str]) -> Literal['ok', 'invalid', 'low_confidence']:
    """Whether a classify attempt is good enough to act on."""
    c = attempt['classification']
    if not c['is_new_label'] and c['label'] not in labels:
        return 'invalid'
    if c['is_new_label'] and c['label'].rsplit('/', 1)[0] not in labels:
        return 'invalid'
    return 'ok' if c['confidence'] >= MIN_CONFIDENCE else 'low_confidence'


def after_classify(state: EmailState, runtime: Runtime[Context]) -> Literal['decide', 'escalate', 'flag_review']:
    """Act on a good answer; otherwise retry on the strong model once, then
    hand it to you. Escalating only when STRONG_MODEL hasn't run yet means
    this can't loop."""
    attempts = state['attempts']
    last = attempts[-1]
    result = acceptable(last, runtime.context.labels)
    if result == 'ok':
        return 'decide'
    if result == 'invalid':
        return 'flag_review'
    
    confidence = last['classification']['confidence']
    if STRONG_MODEL not in {a['model'] for a in attempts}:
        log.info('%s confidence %.2f < %.2f: escalating to %s',
                 last['model'], confidence, MIN_CONFIDENCE, STRONG_MODEL)
        return 'escalate'
    log.info('%s confidence %.2f < %.2f after escalation: flagging for review',
             last['model'], confidence, MIN_CONFIDENCE)
    return 'flag_review'


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

SYSTEM_PROMPT = """You sort emails into Gmail labels.

Existing labels:
{labels}

Pick the single best existing label. Pick the single best existing label. 
Only pick a root level label if there are no sub-labels.
Only if none fits, propose a new
sub-label under an existing parent, formatted "Parent/Child".
"""


def render_email(state: EmailState) -> str:
    """The email as the model sees it."""
    body = (state.get('body') or '')[:MAX_BODY_CHARS]
    parts = [
        f"From: {state.get('sender') or ''}",
        f"Subject: {state.get('subject') or ''}",
        f"Kind: {'newsletter / bulk mail' if state['kind'] == 'bulk' else 'personal'}",
    ]
    # Bulk mail has no body, only Gmail's snippet.
    parts.append(f"\n{body}" if body else f"Snippet: {state.get('snippet') or ''}")
    return '\n'.join(parts)


def classify(state: EmailState, runtime: Runtime[Context]) -> dict:
    """Run the chosen model once and record the attempt."""
    ctx = runtime.context
    llm = ctx.models.get(state['model'], effort=state.get('effort'))

    result: Classification = llm.structured(Classification).invoke([
        SystemMessage(SYSTEM_PROMPT.format(labels='\n'.join(f'- {l}' for l in ctx.labels))),
        HumanMessage(render_email(state)),
    ])
    return {'attempts': [{
        'model': llm.name,
        'effort': state.get('effort'),
        'classification': result.model_dump(),
    }]}


def _decision(action: str, attempt: dict, reason: str) -> dict:
    c = attempt['classification']
    return {
        'action': action,
        'label': c['label'],
        'new_label': c['is_new_label'],
        'confidence': c['confidence'],
        'reason': reason,
        'decided_by': 'llm',
        'model': attempt['model'],
    }


def decide(state: EmailState) -> dict:
    """The last attempt passed acceptable(): label the message with it."""
    last = state['attempts'][-1]
    return {'decisions': [_decision('label', last, last['classification']['reason'])]}


def flag_review(state: EmailState) -> dict:
    """No model was confident enough: leave it for you, with the best guess."""
    attempts = state['attempts']
    best = max(attempts, key=lambda a: a['classification']['confidence'])
    tried = '; '.join(
        f"{a['model']} said {a['classification']['label']!r} at {a['classification']['confidence']:.2f}"
        for a in attempts
    )
    return {'decisions': [_decision('review', best, f'low confidence: {tried}')]}
