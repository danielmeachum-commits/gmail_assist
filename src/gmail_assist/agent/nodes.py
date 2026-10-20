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
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime

from gmail_assist.agent.llm import Classification
from gmail_assist.agent.state import Context, EmailState

log = logging.getLogger(__name__)

# Small local models do worse with very long inputs; trim bodies to this.
MAX_BODY_CHARS = 4000

LOCAL_MODEL = 'local:qwen3:8b'
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


def acceptable(attempt: dict, labels: list[str]) -> bool:
    """Whether a classify attempt is good enough to act on."""
    c = attempt['classification']
    # TODO: also reject answers that are confident but invalid -- a label
    # that isn't in `labels` while is_new_label is False, or a new label
    # whose parent doesn't exist. They then escalate or go to review like a
    # low-confidence answer.
    return c['confidence'] >= MIN_CONFIDENCE


def after_classify(state: EmailState, runtime: Runtime[Context]) -> Literal['decide', 'escalate', 'flag_review']:
    """Act on a good answer; otherwise retry on the strong model once, then
    hand it to you. Escalating only when STRONG_MODEL hasn't run yet means
    this can't loop."""
    attempts = state['attempts']
    last = attempts[-1]
    if acceptable(last, runtime.context.labels):
        return 'decide'
    confidence = last['classification']['confidence']
    if STRONG_MODEL not in {a['model'] for a in attempts}:
        log.info('%s confidence %.2f < %.2f: escalating to %s',
                 last['model'], confidence, MIN_CONFIDENCE, STRONG_MODEL)
        return 'escalate'
    log.info('%s confidence %.2f < %.2f after escalation: flagging for review',
             last['model'], confidence, MIN_CONFIDENCE)
    return 'flag_review'


# -- login codes / sign-in links -----------------------------------------------

# TODO: a starter pattern only, matched against the subject. Extend it (body
# text, sender domains, "magic link" phrasing) and watch for false positives:
# anything this matches gets trashed after a day.
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

def review_subscription(state: EmailState, runtime: Runtime[Context]) -> dict:
    """Look at the list's history and decide whether to flag it for unsubscribing."""
    ctx = runtime.context
    rows = ctx.store.subscription_stats(ctx.account, days=90, key=state['subscription_key'])
    if not rows:
        return {'subscription_stats': None}
    row = rows[0]
    stats = {k: row[k] for k in ('received', 'read', 'read_rate', 'per_week', 'avg_hours_to_read')}

    flag = None
    # TODO: your unsubscribe rule. For example: at least N emails in 90 days
    # and a read rate under X% -> flag. Only an 'active' list is ever changed
    # (run.py passes only_from='active'), so a list you marked 'keep' stays put.
    # if stats['received'] >= 8 and (stats['read_rate'] or 0) < 0.1:
    #     flag = {'status': 'flagged', 'by': 'rule:low_read_rate',
    #             'reason': f"read {stats['read']} of {stats['received']} in 90 days"}

    return {'subscription_stats': stats, 'subscription_flag': flag}


# -- model selection ----------------------------------------------------------------

def choose_model(state: EmailState) -> dict:
    """Every message starts on the local model."""
    # TODO: if some mail should skip straight to STRONG_MODEL (very long
    # bodies, particular senders), decide that here.
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
