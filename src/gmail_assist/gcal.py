"""Add summary items to Google Calendar, and take them off again.

Uses the 'calendar' grant (calendar.events.owned): events on calendars you
own, nothing else. Sign in once with `gmail-assist login calendar`. Every
call builds its own client, since googleapiclient clients aren't
thread-safe and the dashboard serves requests on several threads.
"""
from datetime import date, datetime, timedelta

from googleapiclient.discovery import build

from gmail_assist.gmail.auth import get_credentials

CALENDAR = 'primary'
DEFAULT_LENGTH = timedelta(hours=1)


def item_key(item: dict) -> str:
    """Identifies a summary item across summaries (the dashboard uses the same)."""
    return f"{','.join(item['message_ids'])}@{item['start']}"


def gmail_url(message_id: str) -> str:
    return f'https://mail.google.com/mail/u/0/#all/{message_id}'


def event_body(item: dict, tz: str | None) -> dict:
    """A Calendar event for a summary item with a start (see summary.Item).
    All-day events end the day after their last day (Calendar's end is exclusive)."""
    start, end = item['start'], item.get('end')
    if len(start) == 10:
        last = date.fromisoformat(end or start)
        times = ({'date': start}, {'date': (last + timedelta(days=1)).isoformat()})
    else:
        until = end or (datetime.fromisoformat(start) + DEFAULT_LENGTH).isoformat(timespec='minutes')
        zone = {'timeZone': tz} if tz else {}
        times = ({'dateTime': f'{start}:00', **zone}, {'dateTime': f'{until}:00', **zone})
    links = '\n'.join(gmail_url(i) for i in item['message_ids'])
    return {
        'summary': item['title'],
        'description': f"{item.get('note') or ''}\n\n{links}".strip(),
        'start': times[0],
        'end': times[1],
        # Marks it as ours, should you ever want to find them all.
        'extendedProperties': {'private': {'gmail_assist': item_key(item)}},
    }


def _service():
    return build('calendar', 'v3', credentials=get_credentials('calendar', interactive=False),
                 cache_discovery=False)


def add_event(item: dict, tz: str | None) -> dict:
    """Create the event; returns Calendar's event resource (id, htmlLink, ...)."""
    return _service().events().insert(calendarId=CALENDAR, body=event_body(item, tz)).execute()


def delete_event(event_id: str) -> None:
    _service().events().delete(calendarId=CALENDAR, eventId=event_id).execute()
