"""What each Gmail label means: the one place the prompt, the rules and the
evaluation all read from. Edit DEFINITIONS to change how mail is sorted.

Labels that exist in Gmail but aren't here are never offered to a model.
"""
import json
import re
from pathlib import Path

from gmail_assist.store import _data_dir

# Added by the applier to every message it handles, so Gmail shows what
# this process touched. A marker, not a category.
PROCESSED_LABEL = 'MeachAI'

# Added by rule (never by a model) to mail you sent, on top of the topic label.
ME_LABEL = 'From a person/Me'
# A sender address containing this is you.
OWNER_NAME = 'your-name'

# How to tell senders apart, given to every model before the label list.
GUIDANCE = """\
Decide by WHO sent the email and why, not by its format:
- "From a person" means an individual wrote it personally to you (including
  replies and forwards from people). Automated or company mail is never "From
  a person", even when it is signed with someone's name.
- A newsletter from an organization you belong to or serve (a church,
  volunteer group, school, the military) goes under that organization's topic.
  A newsletter or email from a business selling something is promotional,
  whatever its content.
- Prefer the most specific sub-label that fits. Use the top-level "From an
  organization" only when no sub-label fits."""

DEFINITIONS: dict[str, str] = {
    'From a person':
        'Written personally to you by an individual, including replies and forwards '
        'from people. Never automated mail.',
    'From an organization':
        'Automated or organizational mail that fits none of the sub-labels: service '
        'notices, account settings, family-app notices (e.g. Google Family Link).',
    'From an organization/account-alerts':
        'Alerts about your accounts that need no payment: bank and card transaction '
        'or fraud alerts, utility usage alerts, insurance policy documents.',
    'From an organization/attorney':
        'Correspondence from a law firm or attorney.',
    'From an organization/billing-payment-due':
        'Money you owe: bills and statements, payment due or scheduled (including '
        'autopay), payment failed or returned. Offers to lower a payment are promotional.',
    'From an organization/career-opportunity':
        'Job postings, recruiters, job applications and hiring.',
    'From an organization/community':
        'Volunteer groups, clubs, HOA, nonprofits and civic groups you are involved '
        'with, including their newsletters.',
    'From an organization/military':
        'Military and veteran organizations and services.',
    'From an organization/one-time-code':
        'Login codes, verification codes and sign-in links.',
    'From an organization/order-shipment':
        'Order confirmations, shipping and delivery updates, returns.',
    'From an organization/promotional':
        'Marketing from businesses: sales, coupons, product launches, feedback '
        'surveys, and newsletters whose aim is to sell.',
    'From an organization/receipts':
        'Proof of a completed payment: receipts, payment confirmations, subscription '
        'renewals.',
    'From an organization/religious':
        'Churches and faith communities, including their newsletters. A religious '
        "store's sales are promotional.",
    'From an organization/school':
        'Schools, teachers and classroom platforms.',
}


def examples_path() -> Path:
    return _data_dir() / 'examples.json'


def _load_examples() -> dict[str, list[tuple[str, str]]]:
    """Real emails shown to the models as (sender, subject) per label, a
    couple each, picked from the verified reference set (gmail-assist eval).
    They're your mail, so they live in the data directory, not the repo."""
    try:
        return {label: [tuple(ex) for ex in exs]
                for label, exs in json.loads(examples_path().read_text()).items()}
    except FileNotFoundError:
        return {}


EXAMPLES = _load_examples()


# Addresses nobody personally writes from.
NO_REPLY = re.compile(r'no-?reply|do-?not-?reply|notifications?@', re.IGNORECASE)


def automated(kind: str, sender_email: str | None) -> bool:
    """Mail no person wrote: bulk (it has an unsubscribe header) or from a
    no-reply address."""
    return kind == 'bulk' or bool(NO_REPLY.search(sender_email or ''))


def choices(gmail_labels: list[str], automated: bool = False) -> list[str]:
    """The labels a model may pick for one message: defined, present in
    Gmail, and not rule-only. Automated mail can't be from a person."""
    available = set(gmail_labels)
    return [
        label for label in DEFINITIONS
        if label in available and not (automated and label.startswith('From a person'))
    ]
