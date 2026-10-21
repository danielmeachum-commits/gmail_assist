"""A small web dashboard for watching triage and correcting it.

Shows the daily summaries and the mail that needs you, what was decided and
by whom, the review queue, proposed sub-labels (approve, rename or reject
them, and switch label creation between manual and auto), the labels and
their definitions (create, rename, merge, delete), and the senders labelled
from history. Corrections are recorded as decisions by 'user', which
replace the earlier decision and feed sender history.

It binds to 127.0.0.1 only. Reach it from your other devices through
Tailscale, tailnet only (never Funnel, which is public):

    tailscale serve --bg --https=8443 http://127.0.0.1:8090

Requests must name this machine in their Host header (so a web page can't
reach it by rebinding a DNS name to 127.0.0.1), and changes must be JSON
with an X-Gmail-Assist header (which a cross-site form can't send).

It reads and writes only the local store, and never calls Gmail. The
outside calls are "write a summary now" (Claude, like the schedule) and
adding a summary item to Google Calendar (see gcal.py), and one-click
unsubscribe requests to mailing lists you select (see unsubscribe.py).

Signing in to Google also happens here (see auth.py, "signing in from the
dashboard"): /api/auth/start hands the browser a Google URL, and Google
sends it back to /oauth/callback with a code. The state value that ties the
two together is single-use, expires after SIGNIN_TTL_S, and can only be
created by a request that passed the header check.
"""
import json
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from gmail_assist import gcal, unsubscribe
from gmail_assist.agent import nodes, summary
from gmail_assist.agent.labels import DEFINITIONS, automated, parent
from gmail_assist.agent.run import PROPOSAL_MIN_MESSAGES, PROPOSAL_MIN_SENDERS
from gmail_assist.gmail import auth
from gmail_assist.gmail.auth import AuthRequired
from gmail_assist.store import Decision, Message, Store, Summary

log = logging.getLogger(__name__)

PORT = 8090
MAX_BODY = 64 * 1024
# One-click unsubscribes per request, and how many run at once.
MAX_BULK = 300
# Lists you left stay in the suggestions (marked, with Undo) this long.
RECENTLY_LEFT_DAYS = 14
UNSUBSCRIBE_WORKERS = 4
CSRF_HEADER = 'X-Gmail-Assist'
GRANTS = ('read', 'modify', 'calendar')
SIGNIN_TTL_S = 600


class SignIns:
    """Sign-ins started but not finished, by OAuth state."""

    def __init__(self):
        self._pending: dict[str, tuple[str, Any, float]] = {}
        self._lock = threading.Lock()

    def start(self, access: str) -> str:
        url, state, flow = auth.start_web_signin(access)
        with self._lock:
            now = time.monotonic()
            self._pending = {k: v for k, v in self._pending.items() if v[2] > now}
            self._pending[state] = (access, flow, now + SIGNIN_TTL_S)
        return url

    def finish(self, query: str) -> str:
        """Complete the sign-in a callback's query belongs to; returns its grant."""
        params = {k: v[-1] for k, v in parse_qs(query).items()}
        with self._lock:
            pending = self._pending.pop(params.get('state', ''), None)
        if pending is None or pending[2] < time.monotonic():
            raise ValueError('that sign-in expired or was already used; start again')
        access, flow, _ = pending
        if 'error' in params:
            raise ValueError(f"Google said: {params['error']}")
        auth.finish_web_signin(access, flow, query)
        log.info('signed in to %s access from the dashboard', access)
        return access


def tailnet_name() -> str | None:
    """This machine's Tailscale name (e.g. host.tailnet.ts.net), if any."""
    try:
        out = subprocess.run(['tailscale', 'status', '--json'], capture_output=True,
                             text=True, timeout=5, check=True).stdout
        return json.loads(out)['Self']['DNSName'].rstrip('.') or None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return None


def _iso(dt: datetime | None) -> str | None:
    # SQLite hands datetimes back without a zone; they're stored as UTC.
    return dt.replace(tzinfo=timezone.utc).isoformat() if dt else None


def _received(msg: Message) -> str | None:
    if not msg.internal_date:
        return None
    return datetime.fromtimestamp(msg.internal_date / 1000, timezone.utc).isoformat()


def _decision_row(d: Decision, m: Message) -> dict[str, Any]:
    return {
        'message_id': m.id,
        'received': _received(m),
        'sender': m.sender,
        'sender_email': m.sender_email,
        'subject': m.subject,
        'kind': m.kind,
        'action': d.action,
        'label': d.label,
        'decided_by': d.decided_by,
        'model': d.model,
        'confidence': d.confidence,
        'reason': d.reason,
        'proposal_id': d.proposal_id,
        'decided_at': _iso(d.created_at),
        'attention': m.attention,
        'attention_by': m.attention_by,
        'unread': m.unread,
    }


def local_zone() -> str | None:
    """This machine's IANA zone (e.g. America/New_York): summaries' dates
    are in it, and calendar links say so."""
    try:
        return str(Path('/etc/localtime').resolve()).split('zoneinfo/', 1)[1]
    except (OSError, IndexError):
        return None


def _summary_row(s: Summary) -> dict[str, Any]:
    return {'id': s.id, 'kind': s.kind, 'period_start': _iso(s.period_start),
            'period_end': _iso(s.period_end), 'created_at': _iso(s.created_at),
            'model': s.model, 'tz': local_zone(), **s.content}


class UnsubscribeJob(threading.Thread):
    """One bulk unsubscribe run, in the background so the page can show
    progress and pick it up again after a reload. Each list goes queued ->
    sending -> unsubscribed / failed; then, if asked, the accepted lists'
    emails go to Trash."""

    def __init__(self, api: 'Api', keys: list[str], trash: bool):
        super().__init__(name='unsubscribe', daemon=True)
        self.api, self.trash = api, trash
        self.started_at = datetime.now(timezone.utc)
        self.lists = {k: {'key': k, 'state': 'queued', 'detail': None, 'trashed': 0} for k in keys}
        self.phase = 'unsubscribing'
        self.trash_total = self.trash_done = 0
        self.error: str | None = None
        self.finished = False
        self._lock = threading.Lock()

    def _set(self, key: str, **changes) -> None:
        with self._lock:
            self.lists[key].update(changes)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {'started_at': self.started_at.isoformat(), 'phase': self.phase, 'trash': self.trash,
                    'trash_total': self.trash_total, 'trash_done': self.trash_done,
                    'error': self.error, 'finished': self.finished,
                    'lists': [dict(v) for v in self.lists.values()]}

    def run(self) -> None:
        api = self.api
        try:
            subs = {s.key: s for s in api.store.subscriptions(api.account)}
            latest = api.store.latest_by_list(api.account)

            def one(key: str) -> None:
                self._set(key, state='sending')
                ok, detail = api._one_click(key, subs.get(key), latest.get(key))
                self._set(key, state='unsubscribed' if ok else 'failed', detail=detail)
                if not ok:
                    log.info('one-click unsubscribe from %s refused: %s', key, detail)

            with ThreadPoolExecutor(max_workers=UNSUBSCRIBE_WORKERS) as pool:
                list(pool.map(one, list(self.lists)))
            left = [k for k, v in self.lists.items() if v['state'] == 'unsubscribed']
            log.info('one-click unsubscribe: %d of %d accepted', len(left), len(self.lists))
            if self.trash and left:
                with self._lock:
                    self.phase = 'trashing'
                    self.trash_total = sum(len(api.store.list_message_ids(api.account, k)) for k in left)

                def progress(n: int) -> None:
                    with self._lock:
                        self.trash_done += n

                try:
                    moved = unsubscribe.trash_lists(api.store, api.account, left, progress)
                    for k, n in moved.items():
                        self._set(k, trashed=n)
                except AuthRequired:
                    with self._lock:
                        self.error = 'Emails not moved: Gmail change access is needed (connect it in Settings)'
        except Exception as err:
            log.exception('unsubscribe run failed')
            with self._lock:
                self.error = f'The run stopped: {type(err).__name__}: {err}'
        finally:
            with self._lock:
                self.phase, self.finished = 'done', True


class Api:
    """The dashboard's JSON endpoints, over one store and account."""

    def __init__(self, store: Store, account: str):
        self.store = store
        self.account = account
        self.signins = SignIns()
        # The latest bulk unsubscribe run (kept after it finishes, so a
        # reloaded page can still show how it went).
        self.job: UnsubscribeJob | None = None
        self._job_lock = threading.Lock()

    def auth_status(self) -> dict[str, Any]:
        return {'grants': {g: auth.status(g) for g in GRANTS},
                'web_client': auth.web_redirect_uri() is not None}

    def start_signin(self, body: dict) -> dict[str, Any]:
        if body.get('access') not in GRANTS:
            raise ValueError(f'access must be one of {", ".join(GRANTS)}')
        try:
            return {'url': self.signins.start(body['access'])}
        except FileNotFoundError as err:
            raise ValueError(str(err)) from None

    def definitions(self) -> dict[str, str]:
        return {d.name: d.description for d in self.store.definitions(self.account)}

    # -- reads --------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            'account': self.account,
            'day': self.store.decision_counts(self.account, now - timedelta(days=1)),
            'week': self.store.decision_counts(self.account, now - timedelta(days=7)),
            'queue': self.store.counts(),
            'review': len(self.store.review_queue(self.account)),
            'proposals_pending': len(self.store.proposals(self.account, status='pending')),
            'label_creation': self.store.get_setting(self.account, 'label_creation'),
            'proposal_min_messages': PROPOSAL_MIN_MESSAGES,
            'proposal_min_senders': PROPOSAL_MIN_SENDERS,
        }

    def decisions(self, view: str, limit: int) -> list[dict[str, Any]]:
        if view == 'review':
            rows = self.store.recent_decisions(self.account, limit, action='review')
        elif view in ('sender', 'user', 'thread', 'proposal'):
            rows = self.store.recent_decisions(self.account, limit, decided_by=view)
        else:
            rows = self.store.recent_decisions(self.account, limit)
        out = [_decision_row(d, m) for d, m in rows]
        if view == 'claude':
            out = [r for r in out if r['decided_by'] == 'llm' and (r['model'] or '').startswith('anthropic:')]
        return out

    def proposals(self) -> list[dict[str, Any]]:
        out = []
        for p in self.store.proposals(self.account):
            prop = p['proposal']
            examples = self.store.proposal_messages(self.account, prop.id)[:8]
            out.append({
                'id': prop.id, 'name': prop.name, 'description': prop.description,
                'status': prop.status, 'proposed_by': prop.proposed_by,
                'created_at': _iso(prop.created_at), 'decided_at': _iso(prop.decided_at),
                'decided_by': prop.decided_by,
                'messages': p['messages'], 'senders': p['senders'],
                'ready': (p['messages'] >= PROPOSAL_MIN_MESSAGES
                          and p['senders'] >= PROPOSAL_MIN_SENDERS),
                'examples': [{'sender_email': m.sender_email, 'subject': m.subject,
                              'received': _received(m)} for m in examples],
            })
        return out

    def stats(self) -> dict[str, Any]:
        """Numbers for the Overview: per-day triage by who decided (14 days),
        7-day tiles against the 7 days before, top labels (30 days),
        attention mix (7 days) and the queue."""
        now = datetime.now(timezone.utc)
        today = datetime.now().astimezone().date()
        days = [today - timedelta(days=i) for i in range(13, -1, -1)]
        daily = {d.isoformat(): {'local': 0, 'claude': 0, 'no_model': 0, 'review': 0} for d in days}
        week = {'this': {}, 'last': {}}
        for created, by, model, action, reason in self.store.decision_log(self.account, now - timedelta(days=14)):
            if by in ('user', 'proposal'):
                bucket = 'corrected' if by == 'user' and (reason or '').startswith('corrected') else None
            elif action == 'review':
                bucket = 'review'
            elif by == 'llm':
                bucket = 'local' if (model or '').startswith('local:') else 'claude'
            else:
                bucket = 'no_model'      # rules, thread labels, known senders
            if bucket is None:
                continue
            day = created.astimezone().date().isoformat()
            if bucket != 'corrected' and day in daily:
                daily[day][bucket] += 1
            period = 'this' if created >= now - timedelta(days=7) else 'last'
            week[period][bucket] = week[period].get(bucket, 0) + 1

        def tiles(w: dict[str, int]) -> dict[str, Any]:
            triaged = sum(n for k, n in w.items() if k != 'corrected')
            models = w.get('local', 0) + w.get('claude', 0)
            return {'triaged': triaged,
                    'no_model_share': w.get('no_model', 0) / triaged if triaged else None,
                    'claude_share': w.get('claude', 0) / models if models else None,
                    'correction_rate': w.get('corrected', 0) / triaged if triaged else None,
                    'review': w.get('review', 0)}

        labels = sorted(self.store.label_counts(self.account, now - timedelta(days=30)).items(),
                        key=lambda kv: -kv[1])
        return {
            'daily': [{'day': d, **v} for d, v in daily.items()],
            'this_week': tiles(week['this']), 'last_week': tiles(week['last']),
            'labels': [{'label': k, 'n': n} for k, n in labels[:10]]
                      + ([{'label': 'Other', 'n': sum(n for _, n in labels[10:])}] if len(labels) > 10 else []),
            'attention': self.store.attention_counts(self.account, now - timedelta(days=7)),
            'queue': self.store.counts(),
            'review_open': len(self.store.review_queue(self.account)),
        }

    def unsubscribe_suggestions(self) -> dict[str, Any]:
        """Mailing lists the unsubscribe rules (nodes.unsubscribe_flag) pick
        out, over every list's last 90 days, busiest first. Lists you kept
        or already left aren't suggested."""
        now = datetime.now(timezone.utc)
        latest = self.store.latest_by_list(self.account)
        trashed = self.store.trashed_counts(self.account)
        out, kept = [], 0
        for row in self.store.subscription_stats(self.account, days=nodes.STATS_DAYS):
            sub = row['subscription']
            if sub.status == 'keep':
                kept += 1
                continue
            days_known = (now - sub.first_seen.replace(tzinfo=timezone.utc)).total_seconds() / 86400
            flag = nodes.unsubscribe_flag(row, days_known)
            if sub.status == 'unsubscribed':
                # Shown for a while after you leave, with Undo; then dropped.
                when = sub.status_at.replace(tzinfo=timezone.utc) if sub.status_at else now
                if now - when > timedelta(days=RECENTLY_LEFT_DAYS):
                    continue
                flag = flag or {'reason': sub.status_reason or '', 'by': 'user'}
            elif not flag:
                continue
            last = latest.get(sub.key)
            out.append({
                'key': sub.key, 'sender': sub.sender, 'sender_email': sub.sender_email,
                'reason': flag['reason'], 'rule': flag['by'],
                'received': row['received'], 'read': row['read'], 'per_week': row['per_week'],
                # Header values come from the sender: only web and mail links get through.
                'url': next((u for u in sub.unsubscribe_urls if u.startswith(('https://', 'http://'))), None),
                'mailto': next((u for u in sub.unsubscribe_mailtos if u.startswith('mailto:')), None),
                'one_click': sub.one_click,
                # Can be unsubscribed from here: one-click, over https, with an email to check DKIM on.
                'auto': bool(sub.one_click and last
                             and any(u.startswith('https://') for u in sub.unsubscribe_urls)),
                'unsubscribed': sub.status == 'unsubscribed',
                'unsubscribed_by': sub.status_by if sub.status == 'unsubscribed' else None,
                'trashed': trashed.get(sub.key, 0),
                'last_subject': last.subject if last else None,
                'last_received': _received(last) if last else None,
                'last_id': last.id if last else None,
            })
        out.sort(key=lambda r: -r['per_week'])
        return {'lists': out, 'kept': kept}

    def _keys(self, body: dict) -> list[str]:
        keys = body.get('keys')
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
            raise ValueError('keys must be a list of list keys')
        if len(keys) > MAX_BULK:
            raise ValueError(f'at most {MAX_BULK} at a time')
        return list(dict.fromkeys(keys))

    def _gmail_change(self, call):
        """Run a trash/restore, turning a missing sign-in into a message."""
        try:
            return call()
        except AuthRequired:
            raise ValueError('Moving mail needs Gmail change access: connect it in Settings') from None

    def _after_status(self, keys: list[str], status: str, trash: bool) -> dict[str, int]:
        """Trash the lists' emails when leaving them (if asked); put them back
        when undoing. Returns emails moved per list."""
        if status == 'unsubscribed' and trash:
            return self._gmail_change(lambda: unsubscribe.trash_lists(self.store, self.account, keys))
        if status == 'active':
            return {k: self._gmail_change(lambda k=k: unsubscribe.restore_list(self.store, self.account, k))
                    for k in keys}
        return {}

    def bulk_subscriptions(self, body: dict) -> dict[str, Any]:
        """Set the same status on several lists (keep / unsubscribed / active).
        With trash, leaving a list also moves its emails to Trash."""
        status = body.get('status')
        if status not in ('keep', 'unsubscribed', 'active'):
            raise ValueError('status must be keep, unsubscribed or active')
        keys = self._keys(body)
        for key in keys:
            self.store.set_subscription_status(self.account, key, status, by='user',
                                               reason='set in the dashboard')
        moved = self._after_status(keys, status, bool(body.get('trash')))
        return {'ok': True, 'updated': len(keys), 'moved': moved}

    def start_unsubscribe(self, body: dict) -> dict[str, Any]:
        """Start a background run of one-click unsubscribes for the selected
        lists (and trashing their emails, if asked). Returns the run; poll
        /api/jobs/latest for its progress. One run at a time."""
        keys = self._keys(body)
        with self._job_lock:
            if self.job and not self.job.finished:
                raise ValueError('an unsubscribe run is already going; wait for it to finish')
            self.job = UnsubscribeJob(self, keys, bool(body.get('trash')))
        self.job.start()
        return self.job.snapshot()

    def latest_job(self) -> dict[str, Any] | None:
        return self.job.snapshot() if self.job else None

    def _one_click(self, key: str, sub, last) -> tuple[bool, str]:
        """Unsubscribe from one list; (accepted, what happened)."""
        url = next((u for u in (sub.unsubscribe_urls if sub else []) if u.startswith('https://')), None)
        if not sub or not sub.one_click or not url or not last:
            return False, "doesn't support one-click unsubscribe"
        try:
            detail = unsubscribe.one_click(url, last.id)
        except unsubscribe.Refused as err:
            return False, str(err)
        except AuthRequired:
            return False, 'Gmail sign-in needed to check DKIM'
        except Exception as err:
            log.warning('one-click unsubscribe from %s failed: %s: %s', key, type(err).__name__, err)
            return False, f'failed: {type(err).__name__}'
        self.store.set_subscription_status(self.account, key, 'unsubscribed', by='one-click',
                                           reason=f'one-click unsubscribe: {detail}')
        return True, detail

    def set_subscription(self, body: dict) -> dict[str, Any]:
        """keep (never suggest it again), unsubscribed (you left it), or active (undo)."""
        status = body.get('status')
        if status not in ('keep', 'unsubscribed', 'active'):
            raise ValueError('status must be keep, unsubscribed or active')
        key = body.get('key') or ''
        self.store.set_subscription_status(self.account, key, status, by='user', reason='set in the dashboard')
        moved = self._after_status([key], status, bool(body.get('trash')))
        return {'ok': True, 'moved': moved.get(key, 0)}

    def summaries(self) -> list[dict[str, Any]]:
        """Recent summaries; items already added to Calendar carry 'calendar'."""
        added = self.store.calendar_events(self.account)
        rows = [_summary_row(s) for s in self.store.summaries(self.account)]
        items = [item for row in rows for item in row['act'] + row['read']]
        # Looked up now rather than stored, so older summaries get them too.
        ids = {e['id'] for item in items for e in item['emails']}
        received = self.store.received(self.account, ids)
        done = self.store.done_ids(self.account, ids)
        for item in items:
            for e in item['emails']:
                if e['id'] in received:
                    e['received'] = datetime.fromtimestamp(received[e['id']] / 1000, timezone.utc).isoformat()
            # An item is done when all its emails are (checked off here, or in another summary).
            item['done'] = all(e['id'] in done for e in item['emails'])
            if item.get('start') and (event := added.get(gcal.item_key(item))):
                item['calendar'] = {'key': event.key, 'link': event.html_link}
        return rows

    def add_to_calendar(self, body: dict) -> dict[str, Any]:
        """Add one summary item, found by summary id, section and position
        (the event is built from the stored summary, not from the request)."""
        with self.store.ReadSession() as s:
            sm = s.get(Summary, body.get('summary_id'))
        if sm is None or sm.account != self.account or body.get('section') not in ('act', 'read'):
            raise ValueError('unknown summary item')
        try:
            item = sm.content[body['section']][int(body.get('index'))]
        except (IndexError, TypeError, ValueError):
            raise ValueError('unknown summary item') from None
        if not item.get('start'):
            raise ValueError('that item has no date')
        key = gcal.item_key(item)
        if existing := self.store.calendar_events(self.account).get(key):
            return {'key': key, 'link': existing.html_link}
        event = self._calendar(lambda: gcal.add_event(item, local_zone()))
        self.store.add_calendar_event(self.account, key, event['id'], event.get('htmlLink'), item['title'])
        # On your calendar now: that counts as handled.
        self.store.set_done(self.account, item['message_ids'], True, by='calendar')
        return {'key': key, 'link': event.get('htmlLink'), 'done': True}

    def remove_from_calendar(self, body: dict) -> dict[str, Any]:
        event = self.store.calendar_events(self.account).get(body.get('key') or '')
        if event is None:
            raise ValueError('not in Calendar')
        self._calendar(lambda: gcal.delete_event(event.event_id))
        self.store.remove_calendar_event(self.account, event.key)
        # Un-check only what the calendar add checked off, not what you did.
        ids = event.key.rsplit('@', 1)[0].split(',')
        state = self.store.set_done(self.account, ids, False, only_by='calendar')
        return {'ok': True, 'done': all(state.values())}

    def set_done(self, body: dict) -> dict[str, Any]:
        ids = body.get('message_ids')
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise ValueError('message_ids must be a list of ids')
        state = self.store.set_done(self.account, ids, bool(body.get('done')))
        if len(state) != len(set(ids)):
            raise ValueError('unknown message')
        return {'ok': True, 'done': all(state.values())}

    @staticmethod
    def _calendar(call):
        """Run a Calendar call, turning its failures into messages for the page."""
        from googleapiclient.errors import HttpError

        try:
            return call()
        except AuthRequired:
            raise ValueError('Calendar needs signing in: run `uv run gmail-assist login calendar` '
                             'in a terminal on the server') from None
        except HttpError as err:
            if err.resp.status in (404, 410):   # already gone (undo of a deleted event)
                return None
            raise ValueError(f'Google Calendar said: {err.reason}') from None

    def needs_you(self) -> list[dict[str, Any]]:
        """'act' mail from the last few days, newest first."""
        since = datetime.now(timezone.utc) - timedelta(days=summary.STILL_OPEN_DAYS)
        return [(_decision_row(d, m) if d else {
                    'message_id': m.id, 'received': _received(m), 'sender': m.sender,
                    'sender_email': m.sender_email, 'subject': m.subject, 'label': None,
                    'attention': m.attention, 'unread': m.unread}) | {'done': m.done_at is not None}
                for m, d in self.store.attention_messages(self.account, ['act'], since)]

    def labels(self) -> list[dict[str, Any]]:
        counts = self.store.label_counts(self.account)
        return [{'name': d.name, 'description': d.description, 'source': d.source,
                 'builtin': DEFINITIONS.get(d.name), 'updated_at': _iso(d.updated_at),
                 'messages': counts.get(d.name, 0), 'parent': parent(d.name)}
                for d in self.store.definitions(self.account)]

    def senders(self) -> list[dict[str, Any]]:
        definitions = self.definitions()
        out = []
        for sender, entry in self.store.sender_histories(self.account).items():
            if not automated(entry['kind'], sender):
                continue
            label = nodes.sender_streak(entry['history'], definitions)
            if label is None and not entry['shortcuts']:
                continue
            recent = entry['history'][:nodes.SENDER_HISTORY]
            out.append({
                'sender_email': sender, 'label': label, 'streak': len(recent) if label else 0,
                'shortcuts': entry['shortcuts'],
                # A sender that used to be known but whose latest answers differ.
                'broken': label is None,
                'recent_labels': sorted({d.label or d.action for d in recent}),
            })
        return sorted(out, key=lambda r: (r['broken'], -r['shortcuts'], r['sender_email']))

    # -- writes -------------------------------------------------------------

    def feedback(self, body: dict) -> dict[str, Any]:
        """Correct (or confirm) a message's label."""
        message_id, label = body.get('message_id'), body.get('label')
        if self.store.get_message(self.account, message_id or '') is None:
            raise ValueError('unknown message')
        if label not in self.definitions():
            raise ValueError(f'unknown label {label!r}')
        self.store.record_decision(
            self.account, message_id, 'label', decided_by='user', label=label, confidence=1.0,
            reason='confirmed in the dashboard' if body.get('confirm') else 'corrected in the dashboard',
        )
        return {'ok': True}

    def decide_proposal(self, proposal_id: int, body: dict) -> dict[str, Any]:
        action = body.get('action')
        if action not in ('approve', 'reject'):
            raise ValueError('action must be approve or reject')
        name = (body.get('name') or '').strip() or None
        if name and name in self.definitions():
            raise ValueError(f'{name!r} already exists')
        moved = self.store.decide_proposal(
            self.account, proposal_id, 'approved' if action == 'approve' else 'rejected',
            by='user', name=name, description=(body.get('description') or '').strip() or None,
        )
        return {'ok': True, 'relabelled': moved}

    def set_label(self, body: dict) -> dict[str, Any]:
        name, description = body.get('name'), (body.get('description') or '').strip()
        if name not in self.definitions():
            raise ValueError(f'unknown label {name!r}')
        if not description:
            raise ValueError('description is empty')
        self.store.set_definition(self.account, name, description, source='user')
        return {'ok': True}

    def create_label(self, body: dict) -> dict[str, Any]:
        self.store.create_label(self.account, body.get('name') or '', body.get('description') or '')
        return {'ok': True}

    def rename_label(self, body: dict) -> dict[str, Any]:
        moved = self.store.rename_label(self.account, body.get('name') or '', body.get('new_name') or '')
        return {'ok': True, 'moved': moved}

    def delete_label(self, body: dict) -> dict[str, Any]:
        self.store.delete_label(self.account, body.get('name') or '')
        return {'ok': True}

    def attention(self, body: dict) -> dict[str, Any]:
        level = body.get('level')
        if level not in nodes.LEVELS:
            raise ValueError(f'level must be one of {", ".join(nodes.LEVELS)}')
        if self.store.get_message(self.account, body.get('message_id') or '') is None:
            raise ValueError('unknown message')
        self.store.set_attention(self.account, body['message_id'], level, 'user')
        return {'ok': True}

    def write_summary(self) -> dict[str, Any]:
        """An extra summary now, covering the 24 hours to now."""
        end = datetime.now(timezone.utc)
        summary.write_summary(self.store, self.account, 'manual', end - timedelta(days=1), end)
        return _summary_row(self.store.summaries(self.account, limit=1)[0])

    def settings(self, body: dict) -> dict[str, Any]:
        value = body.get('label_creation')
        if value not in ('manual', 'auto'):
            raise ValueError('label_creation must be manual or auto')
        self.store.set_setting(self.account, 'label_creation', value)
        return {'ok': True}


def make_handler(api: Api, allowed_hosts: set[str]) -> type[BaseHTTPRequestHandler]:
    page = resources.files('gmail_assist').joinpath('dashboard.html').read_bytes()

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if not self._host_ok():
                return self._send(HTTPStatus.MISDIRECTED_REQUEST, {'error': 'unknown host'})
            url = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if url.path == '/':
                return self._send(HTTPStatus.OK, page, 'text/html; charset=utf-8')
            if url.path == '/oauth/callback':
                return self._signin_callback(url.query)
            routes = {
                '/api/summary': api.summary,
                '/api/decisions': lambda: api.decisions(query.get('view', 'all'),
                                                        min(int(query.get('limit', 200)), 1000)),
                '/api/proposals': api.proposals,
                '/api/labels': api.labels,
                '/api/senders': api.senders,
                '/api/summaries': api.summaries,
                '/api/needs-you': api.needs_you,
                '/api/auth': api.auth_status,
                '/api/stats': api.stats,
                '/api/unsubscribe': api.unsubscribe_suggestions,
                '/api/jobs/latest': api.latest_job,
            }
            if url.path not in routes:
                return self._send(HTTPStatus.NOT_FOUND, {'error': 'not found'})
            self._run(routes[url.path])

        def do_POST(self):
            if not self._host_ok():
                return self._send(HTTPStatus.MISDIRECTED_REQUEST, {'error': 'unknown host'})
            if (self.headers.get(CSRF_HEADER) != '1'
                    or not (self.headers.get('Content-Type') or '').startswith('application/json')):
                return self._send(HTTPStatus.FORBIDDEN, {'error': 'forbidden'})
            length = int(self.headers.get('Content-Length') or 0)
            if length > MAX_BODY:
                return self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {'error': 'too large'})
            try:
                body = json.loads(self.rfile.read(length) or b'{}')
            except ValueError:
                return self._send(HTTPStatus.BAD_REQUEST, {'error': 'bad JSON'})
            path = urlparse(self.path).path
            if path == '/api/feedback':
                self._run(lambda: api.feedback(body))
            elif path.startswith('/api/proposals/') and path.rsplit('/', 1)[1].isdigit():
                self._run(lambda: api.decide_proposal(int(path.rsplit('/', 1)[1]), body))
            elif path == '/api/labels':
                self._run(lambda: api.set_label(body))
            elif path == '/api/labels/create':
                self._run(lambda: api.create_label(body))
            elif path == '/api/labels/rename':
                self._run(lambda: api.rename_label(body))
            elif path == '/api/labels/delete':
                self._run(lambda: api.delete_label(body))
            elif path == '/api/attention':
                self._run(lambda: api.attention(body))
            elif path == '/api/summaries':
                self._run(api.write_summary)
            elif path == '/api/auth/start':
                self._run(lambda: api.start_signin(body))
            elif path == '/api/subscriptions':
                self._run(lambda: api.set_subscription(body))
            elif path == '/api/subscriptions/bulk':
                self._run(lambda: api.bulk_subscriptions(body))
            elif path == '/api/unsubscribe/one-click':
                self._run(lambda: api.start_unsubscribe(body))
            elif path == '/api/done':
                self._run(lambda: api.set_done(body))
            elif path == '/api/calendar':
                self._run(lambda: api.add_to_calendar(body))
            elif path == '/api/calendar/undo':
                self._run(lambda: api.remove_from_calendar(body))
            elif path == '/api/settings':
                self._run(lambda: api.settings(body))
            else:
                self._send(HTTPStatus.NOT_FOUND, {'error': 'not found'})

        def _signin_callback(self, query: str) -> None:
            """Google sends the browser here after a sign-in; back to the page,
            which reports the result from the fragment."""
            try:
                result = f'ok:{api.signins.finish(query)}'
            except Exception as err:
                log.warning('dashboard sign-in failed: %s: %s', type(err).__name__, err)
                result = f'error:{err}'
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header('Location', f'/#signin={quote(result)}')
            self.send_header('Content-Length', '0')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()

        def _host_ok(self) -> bool:
            return (self.headers.get('Host') or '').lower() in allowed_hosts

        def _run(self, fn) -> None:
            try:
                result = fn()
            except (ValueError, KeyError) as err:
                # str() of a KeyError quotes its message; use the message itself.
                return self._send(HTTPStatus.BAD_REQUEST, {'error': str(err.args[0]) if err.args else str(err)})
            except Exception:
                log.exception('dashboard request failed: %s', self.path)
                return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': 'server error'})
            self._send(HTTPStatus.OK, result)

        def _send(self, status: HTTPStatus, body, content_type: str = 'application/json') -> None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy',
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            log.debug(format, *args)

    return DashboardHandler


def start(account: str, db: Path | str | None = None, host: str = '127.0.0.1',
          port: int = PORT) -> ThreadingHTTPServer:
    """Serve the dashboard on a background thread; returns the server."""
    store = Store(db)
    store.seed_definitions(account, DEFINITIONS)
    names = {f'{host}:{port}', f'localhost:{port}', f'127.0.0.1:{port}'}
    if ts := tailnet_name():
        # tailscale serve passes the name on unchanged, with or without its port.
        names |= {ts.lower(), f'{ts.lower()}:8443'}
    server = ThreadingHTTPServer((host, port), make_handler(Api(store, account), names))
    threading.Thread(target=server.serve_forever, name='dashboard', daemon=True).start()
    log.info('dashboard on http://%s:%d (tailnet: https://%s:8443)', host, port, ts or '?')
    return server
