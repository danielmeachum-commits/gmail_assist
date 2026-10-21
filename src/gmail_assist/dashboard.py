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

It reads and writes only the local store, and never calls Gmail. The one
outside call is "write a summary now", which asks Claude like the schedule does.
"""
import json
import logging
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from gmail_assist.agent import nodes, summary
from gmail_assist.agent.labels import DEFINITIONS, automated, parent
from gmail_assist.agent.run import PROPOSAL_MIN_MESSAGES, PROPOSAL_MIN_SENDERS
from gmail_assist.store import Decision, Message, Store, Summary

log = logging.getLogger(__name__)

PORT = 8090
MAX_BODY = 64 * 1024
CSRF_HEADER = 'X-Gmail-Assist'


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


def _summary_row(s: Summary) -> dict[str, Any]:
    return {'id': s.id, 'kind': s.kind, 'period_start': _iso(s.period_start),
            'period_end': _iso(s.period_end), 'created_at': _iso(s.created_at),
            'model': s.model, **s.content}


class Api:
    """The dashboard's JSON endpoints, over one store and account."""

    def __init__(self, store: Store, account: str):
        self.store = store
        self.account = account

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

    def summaries(self) -> list[dict[str, Any]]:
        return [_summary_row(s) for s in self.store.summaries(self.account)]

    def needs_you(self) -> list[dict[str, Any]]:
        """'act' mail from the last few days, newest first."""
        since = datetime.now(timezone.utc) - timedelta(days=summary.STILL_OPEN_DAYS)
        return [_decision_row(d, m) if d else {
                    'message_id': m.id, 'received': _received(m), 'sender': m.sender,
                    'sender_email': m.sender_email, 'subject': m.subject, 'label': None,
                    'attention': m.attention, 'unread': m.unread}
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
            routes = {
                '/api/summary': api.summary,
                '/api/decisions': lambda: api.decisions(query.get('view', 'all'),
                                                        min(int(query.get('limit', 200)), 1000)),
                '/api/proposals': api.proposals,
                '/api/labels': api.labels,
                '/api/senders': api.senders,
                '/api/summaries': api.summaries,
                '/api/needs-you': api.needs_you,
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
            elif path == '/api/settings':
                self._run(lambda: api.settings(body))
            else:
                self._send(HTTPStatus.NOT_FOUND, {'error': 'not found'})

        def _host_ok(self) -> bool:
            return (self.headers.get('Host') or '').lower() in allowed_hosts

        def _run(self, fn) -> None:
            try:
                result = fn()
            except (ValueError, KeyError) as err:
                return self._send(HTTPStatus.BAD_REQUEST, {'error': str(err)})
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
