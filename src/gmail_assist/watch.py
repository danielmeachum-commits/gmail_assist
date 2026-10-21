"""Keep the store in sync from Gmail push notifications.

Gmail publishes to a Pub/Sub topic whenever the mailbox changes; a push
subscription POSTs those to serve()'s endpoint (exposed through a tunnel such
as Tailscale Funnel or Cloudflare Tunnel). A notification only carries
{emailAddress, historyId}, so all it does is wake the sync worker, which runs
incremental_sync() from the stored cursor -- duplicate or out-of-order
notifications are harmless.

The worker also renews the Gmail watch daily (it expires after 7 days) and
syncs every BACKSTOP_EVERY_S regardless, since Gmail can drop notifications.

With triage on, each sync is followed by a pass of the triage graph over
pending mail that arrived after serve started (the backlog is left for
`gmail-assist process`). The backstop also puts failed messages back in the
queue, up to MAX_ATTEMPTS each, so a model server that was down catches up.

Every request must carry the OIDC token Pub/Sub adds when the subscription
has push auth enabled (--push-auth-service-account); anything else gets a 403.
"""
import base64
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from google.auth.exceptions import GoogleAuthError, RefreshError
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from gmail_assist.gmail.auth import AuthRequired
from gmail_assist.store import Store
from gmail_assist.sync import incremental_sync

log = logging.getLogger(__name__)

RENEW_EVERY_S = 24 * 3600
RENEW_RETRY_S = 5 * 60
BACKSTOP_EVERY_S = 30 * 60
# Gmail often sends several notifications for one change; wait briefly so
# they collapse into a single sync.
COALESCE_S = 2
MAX_BODY = 64 * 1024
# Triage: messages claimed per graph pass, and tries before a message is
# left as failed for good.
TRIAGE_BATCH = 20
MAX_ATTEMPTS = 3


def start_watch(service, topic: str) -> dict:
    """(Re)register the mailbox watch. Safe to call repeatedly."""
    response = service.users().watch(userId='me', body={'topicName': topic}).execute()
    expires = time.strftime('%Y-%m-%d %H:%M', time.localtime(int(response['expiration']) / 1000))
    log.info('watching %s until %s', topic, expires)
    return response


class PushVerifier:
    """Checks the OIDC token Pub/Sub signs each push request with."""

    def __init__(self, audience: str, service_account: str | None = None):
        self.audience = audience
        self.service_account = service_account
        self._request = Request()

    def verify(self, authorization: str | None) -> bool:
        if not authorization or not authorization.startswith('Bearer '):
            return False
        try:
            claims = id_token.verify_oauth2_token(
                authorization.removeprefix('Bearer '), self._request, audience=self.audience
            )
        except (ValueError, GoogleAuthError) as err:
            log.warning('rejected push: %s', err)
            return False
        if self.service_account and (
            claims.get('email') != self.service_account or not claims.get('email_verified')
        ):
            log.warning('rejected push from %s', claims.get('email'))
            return False
        return True


class SyncWorker(threading.Thread):
    """Runs every sync on one thread: the Gmail client isn't thread-safe,
    and serializing syncs means the history cursor only moves forward."""

    def __init__(self, service, account: str, topic: str, db: Path | str | None = None,
                 triage: bool = True, workers: int = 4, on_fatal=None):
        super().__init__(name='sync', daemon=True)
        # Called (from this thread) when the worker can't carry on, e.g.
        # Google rejected the refresh token; serve() uses it to shut down.
        self.on_fatal = on_fatal
        self.fatal: Exception | None = None
        self.service = service
        self.account = account
        self.topic = topic
        self.db = db
        self.triage = triage
        self.workers = workers
        # Only mail received from now on is triaged automatically.
        self.since_ms = int(time.time() * 1000)
        self._wake = threading.Event()

    def notify(self) -> None:
        self._wake.set()

    def run(self) -> None:
        try:
            self._loop()
        except RefreshError as err:
            # Retrying can't fix a revoked or expired refresh token.
            log.error('Google rejected the refresh token: %s', err)
            self.fatal = AuthRequired('read')
        except Exception as err:
            log.exception('sync worker crashed')
            self.fatal = err
        if self.on_fatal:
            self.on_fatal()

    def _loop(self) -> None:
        next_renew = 0.0
        with Store(self.db) as store:
            while True:
                if time.monotonic() >= next_renew:
                    try:
                        start_watch(self.service, self.topic)
                        next_renew = time.monotonic() + RENEW_EVERY_S
                    except RefreshError:
                        raise
                    except Exception:
                        log.exception('watch renewal failed')
                        next_renew = time.monotonic() + RENEW_RETRY_S
                # Clear before syncing, so a notification that lands mid-sync
                # triggers another pass instead of being lost.
                self._wake.clear()
                try:
                    incremental_sync(self.service, store, self.account)
                except RefreshError:
                    raise
                except Exception:
                    log.exception('sync failed')
                if self.triage:
                    self._triage(store)
                if self._wake.wait(BACKSTOP_EVERY_S):
                    time.sleep(COALESCE_S)
                elif self.triage:
                    # Backstop pass: retry what failed (e.g. llama-server was
                    # down) and anything stuck from a crash.
                    n = store.requeue(failed=True, stale_after_s=3600, max_attempts=MAX_ATTEMPTS)
                    if n:
                        log.info('requeued %d failed or stuck messages', n)

    def _triage(self, store: Store) -> None:
        from gmail_assist.agent.run import process

        while store.pending_count(self.account, self.since_ms):
            try:
                counts = process(self.service, store, limit=TRIAGE_BATCH,
                                 workers=self.workers, since_ms=self.since_ms)
            except RefreshError:
                raise
            except Exception:
                log.exception('triage failed')
                return
            log.info('triaged: %s', counts)
            if not sum(counts.values()):
                return  # nothing claimable after all; don't spin


def make_handler(path: str, verifier: PushVerifier, on_notify) -> type[BaseHTTPRequestHandler]:
    class PushHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != path:
                return self._reply(HTTPStatus.NOT_FOUND)
            if not verifier.verify(self.headers.get('Authorization')):
                return self._reply(HTTPStatus.FORBIDDEN)
            length = int(self.headers.get('Content-Length') or 0)
            if length > MAX_BODY:
                return self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            try:
                envelope = json.loads(self.rfile.read(length))
                data = json.loads(base64.b64decode(envelope['message']['data']))
                log.info('push: %s historyId=%s', data.get('emailAddress'), data.get('historyId'))
            except (ValueError, KeyError, TypeError) as err:
                # Still ack: Pub/Sub would redeliver a malformed message forever.
                log.warning('unreadable push body: %s', err)
            on_notify()
            self._reply(HTTPStatus.NO_CONTENT)

        def _reply(self, status: HTTPStatus) -> None:
            self.send_response(status)
            self.send_header('Content-Length', '0')
            self.end_headers()

        def log_message(self, format, *args):
            log.debug(format, *args)

    return PushHandler


def serve(service, topic: str, audience: str, service_account: str | None = None, *,
          host: str = '127.0.0.1', port: int = 8080, path: str = '/gmail/push',
          db: Path | str | None = None, triage: bool = True, workers: int = 4) -> None:
    """Runs until interrupted. If the sync worker stops (e.g. the Gmail token
    was revoked), shuts the server down and raises its error, so a service
    manager sees the failure instead of a server that silently stopped syncing."""
    account = service.users().getProfile(userId='me').execute()['emailAddress']
    server = None
    worker = SyncWorker(service, account, topic, db, triage=triage, workers=workers,
                        on_fatal=lambda: server and server.shutdown())
    server = ThreadingHTTPServer(
        (host, port), make_handler(path, PushVerifier(audience, service_account), worker.notify)
    )
    worker.start()
    log.info('listening on http://%s:%d%s for %s', host, port, path, account)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    if worker.fatal:
        raise worker.fatal
