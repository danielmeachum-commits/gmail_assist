"""Graph state and runtime context for the email triage graph.

The graph runs once per message. State starts as the message's fields and
picks up working values and decisions as it moves through the nodes. The
graph itself never writes to the store or Gmail: decisions come back in the
final state and run.py records them, which keeps every node easy to test.
"""
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from gmail_assist.agent.llm import Models
from gmail_assist.store import Action, Message, Store, SubscriptionStatus


class DecisionDraft(TypedDict, total=False):
    """Mirrors Store.record_decision()'s arguments."""
    action: Action             # required
    decided_by: str            # required: 'rule:<name>' or 'llm'
    label: str
    new_label: bool
    reason: str
    model: str
    confidence: float
    # A sub-label this message supports; it's labelled with the parent meanwhile.
    proposal: str
    proposal_description: str
    proposed_by: str           # the model that proposed it


class SubscriptionFlag(TypedDict):
    status: SubscriptionStatus
    reason: str
    by: str


class EmailState(TypedDict, total=False):
    # -- input: filled from the stored message by initial_state() ----------
    account: str
    message_id: str
    kind: str                  # 'personal' | 'bulk'
    sender: str | None
    sender_email: str | None
    subject: str | None
    snippet: str | None
    body: str | None           # None for bulk mail (headers only)
    label_ids: list[str]
    subscription_key: str | None
    thread_id: str

    # -- working values set by nodes ---------------------------------------
    is_login_code: bool
    inherited: bool             # took an earlier message's thread label
    known_sender: bool          # labelled from the sender's history, no model
    subscription_stats: dict[str, Any] | None
    # The model classify uses next, as "<provider>:<model>"; set by
    # choose_model and changed by escalate. None means no model available.
    model: str | None
    effort: str | None          # Claude only: low | medium | high | xhigh | max
    # One entry per classify run: {'model': ..., 'classification': {...}}.
    # Only the last becomes a decision; all are kept in the message's result.
    attempts: Annotated[list[dict[str, Any]], operator.add]
    # What refine_parent made of a parent-label answer: {'model', 'choice',
    # 'label', 'description', 'reason'}.
    refinement: dict[str, Any] | None

    # -- output -------------------------------------------------------------
    # operator.add makes this a reducer: each node returns only its new
    # decisions and LangGraph appends them, rather than overwriting.
    decisions: Annotated[list[DecisionDraft], operator.add]
    subscription_flag: SubscriptionFlag | None


@dataclass
class Context:
    """Per-run dependencies, reached in nodes via runtime.context."""
    store: Store
    account: str
    # The labels models may choose, name -> definition: those defined in the
    # store and present in Gmail, plus approved sub-labels not created yet.
    definitions: dict[str, str] = field(default_factory=dict)
    models: Models = field(default_factory=Models)


def initial_state(msg: Message) -> EmailState:
    return {
        'account': msg.account,
        'message_id': msg.id,
        'kind': msg.kind,
        'sender': msg.sender,
        'sender_email': msg.sender_email,
        'subject': msg.subject,
        'snippet': msg.snippet,
        'body': msg.body_text(),
        'label_ids': msg.label_ids,
        'subscription_key': msg.subscription_key,
        'thread_id': msg.thread_id,
        'attempts': [],
        'decisions': [],
        'subscription_flag': None,
    }
