"""Apply triage decisions to Gmail: the applier.

After each sync, serve takes every message with a decision not yet applied
and makes its Gmail labels match what was decided:

- add the message's current topic label (its latest decision: a model's,
  a thread's, a known sender's, or your correction in the dashboard), plus
  PROCESSED_LABEL so Gmail shows what this touched, plus ME_LABEL on mail
  you sent;
- remove a topic label this applied earlier that's no longer current (you
  corrected it, or a sub-label was approved). Only labels it added itself:
  never one you put on by hand.

Nothing is archived or moved: mail stays where it is, labelled. Labels
missing from Gmail (a new sub-label, one you created in the dashboard) are
created first. Messages needing the same change share batchModify calls.
Decisions that ask for no label (review, expire) are left unapplied for now.

Uses the 'modify' grant; the apply_labels setting turns it off.
"""
import logging
from collections import defaultdict
from datetime import datetime, timezone

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy import select, update

from gmail_assist.agent.labels import ME_LABEL, PROCESSED_LABEL
from gmail_assist.gmail.auth import get_credentials
from gmail_assist.store import Decision, Message, Store

log = logging.getLogger(__name__)

BATCH_MODIFY_MAX = 1000


def _plan(store: Store, account: str) -> dict[str, tuple[set[str], set[str], list[int]]]:
    """For each message with unapplied decisions: (labels to add, labels to
    remove, ids of its decisions to mark applied). Messages whose current
    decision has no label to apply yet are left out (and stay unapplied)."""
    with store.ReadSession() as s:
        pending = set(s.scalars(select(Decision.message_id).where(
            Decision.account == account, Decision.applied_at.is_(None), Decision.apply_error.is_(None))))
        if not pending:
            return {}
        rows = s.execute(
            select(Decision).join(Message, (Message.account == Decision.account) & (Message.id == Decision.message_id))
            .where(Decision.account == account, Decision.message_id.in_(list(pending)))
            .order_by(Decision.id)).scalars().all()
    by_message: dict[str, list[Decision]] = defaultdict(list)
    for d in rows:
        by_message[d.message_id].append(d)

    plan = {}
    for message_id, decisions in by_message.items():
        # ME_LABEL is only ever the rule's (mail you sent). An older classifier
        # could pick it as a topic; such a pick is neither a topic nor yours.
        topic = [d for d in decisions if not (d.action == 'label' and d.label == ME_LABEL)]
        current = topic[-1] if topic else None
        if current is None or current.action != 'label':
            continue
        add = {current.label, PROCESSED_LABEL}
        if any(d.decided_by == 'rule:me' for d in decisions):
            add.add(ME_LABEL)
        added_before = {d.label for d in topic if d.applied_at and d.action == 'label'}
        remove = added_before - {current.label}
        plan[message_id] = (add, remove, [d.id for d in decisions if d.applied_at is None])
    return plan


def _label_ids(gmail, names: set[str]) -> dict[str, str]:
    """Gmail label ids by name, creating any that don't exist (parents first,
    so nested labels show nested)."""
    existing = {l['name']: l['id'] for l in gmail.users().labels().list(userId='me').execute()['labels']}
    wanted = set(names)
    for name in names:                 # "A/B/C" needs "A" and "A/B" too
        parts = name.split('/')
        wanted |= {'/'.join(parts[:i]) for i in range(1, len(parts))}
    for name in sorted(wanted - set(existing), key=lambda n: n.count('/')):
        created = gmail.users().labels().create(userId='me', body={
            'name': name, 'labelListVisibility': 'labelShow', 'messageListVisibility': 'show'}).execute()
        existing[name] = created['id']
        log.info('created Gmail label %r', name)
    return {n: existing[n] for n in names}


def apply_decisions(store: Store, account: str) -> int:
    """Apply what's pending. Returns how many messages were updated."""
    if store.get_setting(account, 'apply_labels') != 'on':
        return 0
    plan = _plan(store, account)
    if not plan:
        return 0
    gmail = build('gmail', 'v1', credentials=get_credentials('modify', interactive=False),
                  cache_discovery=False)
    ids = _label_ids(gmail, {n for add, remove, _ in plan.values() for n in add | remove})

    groups: dict[tuple[frozenset, frozenset], list[str]] = defaultdict(list)
    for message_id, (add, remove, _) in plan.items():
        groups[(frozenset(add), frozenset(remove))].append(message_id)

    done, failed = [], {}
    for (add, remove), messages in groups.items():
        body = {'addLabelIds': [ids[n] for n in add], 'removeLabelIds': [ids[n] for n in remove]}
        for start in range(0, len(messages), BATCH_MODIFY_MAX):
            chunk = messages[start:start + BATCH_MODIFY_MAX]
            try:
                gmail.users().messages().batchModify(userId='me', body={'ids': chunk, **body}).execute(num_retries=5)
                done += chunk
            except HttpError as err:
                # One bad id (deleted since) fails the batch: go one by one to find it.
                log.warning('batch of %d failed (%s); retrying one at a time', len(chunk), err.resp.status)
                for message_id in chunk:
                    try:
                        gmail.users().messages().modify(userId='me', id=message_id, body=body).execute(num_retries=5)
                        done.append(message_id)
                    except HttpError as one:
                        failed[message_id] = f'HTTP {one.resp.status}: {one.reason}'

    now = datetime.now(timezone.utc)
    with store.Session.begin() as s:
        for message_id in done:
            s.execute(update(Decision).where(Decision.id.in_(plan[message_id][2])).values(applied_at=now))
        for message_id, error in failed.items():
            s.execute(update(Decision).where(Decision.id.in_(plan[message_id][2])).values(apply_error=error))
    if failed:
        log.warning('could not label %d messages, e.g. %s', len(failed), next(iter(failed.values())))
    return len(done)
