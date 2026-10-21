"""Morning briefings and evening recaps of the mail that needs you.

Each message gets an attention level as it's triaged (nodes.assess_attention).
A summary takes the 'act' and 'read' mail from its period, plus 'act' mail
from the few days before that's still unread, leaving out anything you've
checked off in the dashboard, and has Claude write a short
briefing: what to do and by when, then what's worth a look. One Claude call
per summary; with nothing to report there's no call at all.

The schedule (local time): the morning briefing covers the 24 hours before
MORNING; the evening recap covers MORNING to EVENING. serve writes each one
on time, or when it starts if it was down then (up to CATCH_UP late).
"""
import logging
import threading
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from gmail_assist.agent.llm import Models
from gmail_assist.agent.nodes import STRONG_MODEL, URL
from gmail_assist.store import Store

log = logging.getLogger(__name__)

MORNING = time(7, 0)
EVENING = time(18, 0)
SUMMARY_EFFORT = 'medium'
STILL_OPEN_DAYS = 3          # unread 'act' mail this old is carried forward
CATCH_UP = timedelta(hours=12)   # after downtime, write a missed summary this late at most
MAX_EMAILS = 80              # newest first, 'act' before 'read'
EXCERPT_CHARS = 500


class Item(BaseModel):
    message_ids: list[str] = Field(description='The email ids (from the list) this item covers; '
                                               'several when they are about the same thing.')
    title: str = Field(description='What it is, at most 6 words, e.g. "Duke Energy bill".')
    due: str | None = Field(description='The due date, event date or deadline, short, e.g. '
                                        '"Thu Oct 22, 7 PM" or "by Nov 3"; null if there is none.')
    start: str | None = Field(description='The same date in ISO 8601 local time: "2026-10-22T19:00" '
                                          'with a time, "2026-10-22" without. Null if none. Resolve '
                                          'weekdays and missing years against Now. Never guess a '
                                          'date the email does not give.')
    end: str | None = Field(description='When it ends, same format, if the email says; else null.')
    note: str = Field(description='What to do or why it matters, at most 12 words.')


class Briefing(BaseModel):
    headline: str = Field(description='One sentence on the period overall.')
    act: list[Item] = Field(description='Everything that needs doing, most urgent first.')
    read: list[Item] = Field(description='Worth a look but nothing to do; at most 10.')


PROMPT = """\
You write a short email briefing for one person. Each email below was
marked "act" (something to do or a date to keep) or "read" (worth reading)
by a small model, which is sometimes wrong: move an email to the other list
if it belongs there, and leave out any that don't matter after all.

- act: bills and payments due, failed payments, fraud or security alerts,
  events, appointments, RSVPs, deadlines, forms, people waiting on a reply.
  Merge emails about the same thing into one item. Most urgent first.
- read: statements, account and school notices, community and personal
  news. At most 10, most useful first.
- Emails marked "still open" arrived earlier and are unread: include them
  under act if they still look current.
- Never invent a date. Use only the email ids listed.
- Be terse: this is read on a phone. Titles of at most 6 words, notes of at
  most 12, a headline of at most 20. No filler like "please note".

Now: {now}"""


def _iso_local(value: str | None) -> str | None:
    """`value` if it's a valid date ("2026-10-22") or local date and time
    ("2026-10-22T19:00"), else None: the calendar button needs one of those."""
    if not value:
        return None
    value = value.strip()
    try:
        if len(value) == 10:
            return date.fromisoformat(value).isoformat()
        parsed = datetime.fromisoformat(value)
        # A zone, if the model added one, is dropped: times are local here.
        return parsed.replace(tzinfo=None, second=0, microsecond=0).isoformat(timespec='minutes')
    except ValueError:
        return None


def _excerpt(msg) -> str:
    text = msg.body_text() or msg.snippet or ''
    return ' '.join(URL.sub(r'<\1 link>', text).split())[:EXCERPT_CHARS]


def _render(msg, decision, still_open: bool) -> str:
    received = datetime.fromtimestamp((msg.internal_date or 0) / 1000).astimezone()
    tags = msg.attention + (', still open' if still_open else '')
    return (f'[{msg.id}] ({tags}) {received:%a %b %d %H:%M} | From: {msg.sender} | '
            f'Label: {decision.label if decision else "?"}\n'
            f'Subject: {msg.subject}\n{_excerpt(msg)}')


def write_summary(store: Store, account: str, kind: str, start: datetime, end: datetime,
                  models: Models | None = None) -> dict[str, Any]:
    """Write and store the summary for [start, end). Returns its content."""
    # Mail you've already checked off in the dashboard is handled: leave it out.
    in_period = store.attention_messages(account, ['act', 'read'], start, end, open_only=True)
    open_before = store.attention_messages(account, ['act'], start - timedelta(days=STILL_OPEN_DAYS),
                                           start, unread_only=True, open_only=True)
    rows = sorted(in_period, key=lambda r: r[0].attention != 'act')[:MAX_EMAILS]
    rows += open_before[:max(0, MAX_EMAILS - len(rows))]
    open_ids = {m.id for m, _ in open_before}
    by_id = {m.id: m for m, _ in rows}
    counts = {level: sum(1 for m, _ in in_period if m.attention == level) for level in ('act', 'read')}
    content: dict[str, Any] = {'counts': counts | {'still_open': len(open_before)},
                               'headline': 'Nothing needs you.', 'act': [], 'read': []}
    model = None
    if rows:
        llm = (models or Models()).get(STRONG_MODEL, effort=SUMMARY_EFFORT)
        model = llm.name
        emails = '\n\n'.join(_render(m, d, m.id in open_ids) for m, d in rows)
        briefing: Briefing = llm.structured(Briefing).invoke([
            SystemMessage(PROMPT.format(now=end.astimezone().strftime('%A %B %d %Y, %H:%M'))),
            HumanMessage(emails),
        ])
        content['headline'] = briefing.headline

        def items(entries: list[Item]) -> list[dict[str, Any]]:
            out = []
            for item in entries:
                ids = [i for i in item.message_ids if i in by_id]
                if not ids:
                    continue  # an id the model made up
                start, end = _iso_local(item.start), _iso_local(item.end)
                if not start or (end and len(end) != len(start)):
                    end = None
                out.append({**item.model_dump(), 'start': start, 'end': end,
                            'message_ids': ids, 'emails': [
                    {'id': i, 'sender_email': by_id[i].sender_email, 'subject': by_id[i].subject,
                     'unread': by_id[i].unread} for i in ids]})
            return out

        content['act'], content['read'] = items(briefing.act), items(briefing.read)
    store.add_summary(account, kind, start, end, content, model)
    log.info('%s summary: %d to act on, %d to read', kind, len(content['act']), len(content['read']))
    return content


def due_slots(now: datetime) -> list[tuple[str, datetime, datetime]]:
    """The most recent morning and evening slots at or before `now`, as
    (kind, period start, period end), in local time."""
    now = now.astimezone()
    today = now.date()

    def at(day, t):
        return datetime.combine(day, t).astimezone()

    morning = at(today, MORNING) if now >= at(today, MORNING) else at(today - timedelta(days=1), MORNING)
    evening = at(today, EVENING) if now >= at(today, EVENING) else at(today - timedelta(days=1), EVENING)
    return [('morning', morning - timedelta(days=1), morning),
            ('evening', at(evening.date(), MORNING), evening)]


class SummaryScheduler(threading.Thread):
    """Writes each scheduled summary that hasn't been written yet, then
    sleeps until the next one. Only the latest missed slot of each kind is
    written; a long outage doesn't produce a pile of old summaries."""

    def __init__(self, account: str, db: Path | str | None = None):
        super().__init__(name='summaries', daemon=True)
        self.account = account
        self.db = db
        self._wake = threading.Event()

    def run(self) -> None:
        with Store(self.db) as store:
            while True:
                failed = False
                now = datetime.now().astimezone()
                for kind, start, end in due_slots(now):
                    if now - end > CATCH_UP:
                        continue
                    last = store.last_summary(self.account, kind)
                    # SQLite hands datetimes back without a zone; they're stored as UTC.
                    if last is None or last.period_end.replace(tzinfo=timezone.utc) < end:
                        try:
                            write_summary(store, self.account, kind, start, end)
                        except Exception:
                            log.exception('%s summary failed; retrying in an hour', kind)
                            failed = True
                self._wake.wait(3600 if failed else self._seconds_to_next())

    @staticmethod
    def _seconds_to_next() -> float:
        now = datetime.now().astimezone()
        upcoming = [datetime.combine(now.date() + timedelta(days=d), t).astimezone()
                    for d in (0, 1) for t in (MORNING, EVENING)]
        return min((u - now).total_seconds() for u in upcoming if u > now) + 5
