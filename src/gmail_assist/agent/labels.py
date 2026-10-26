"""What each Gmail label means: the one place the prompt, the rules and the
evaluation all read from.

DEFINITIONS seeds the store's label_definitions table, which is what the
models actually see: editing one here updates the store, unless you've since
edited it in the dashboard. Sub-labels approved from proposals are added
there too. Labels in Gmail without a definition are never offered to a model.

Writing a definition: say what belongs, in general terms ("bank and card
alerts"), not who sends it. Name an organization or person only when the
model couldn't work it out from the email itself -- a private group, or an
acronym a small model may not know. Specific well-known senders belong in
the examples (examples.json) or are learned from sender history.

The repo is public, so anything about you -- labels for your own work or
groups, definitions naming them, your name -- goes in personal.json in the
data directory instead:

    {"owner": "<text in your sender addresses>",
     "definitions": {"<label>": "<definition>", ...}}

Its definitions are added to (or replace) the ones here.
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


def _load_personal() -> dict:
    try:
        return json.loads((_data_dir() / 'personal.json').read_text())
    except FileNotFoundError:
        return {}


PERSONAL = _load_personal()
DEFINITIONS |= PERSONAL.get('definitions', {})
# A sender address containing this is you (besides the account itself).
OWNER_NAME: str | None = PERSONAL.get('owner')


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


def choices(definitions: dict[str, str], automated: bool = False) -> list[str]:
    """The labels a model may pick for one message, from the definitions in
    use (see Context.definitions). Automated mail can't be from a person."""
    return [label for label in definitions
            if not (automated and label.startswith('From a person'))]


def parent(label: str) -> str | None:
    return label.rsplit('/', 1)[0] if '/' in label else None


def is_parent(label: str, definitions: dict[str, str]) -> bool:
    """Whether other labels sit under this one, so a sub-label could fit better."""
    return any(parent(other) == label for other in definitions)
