"""Keep the store in sync from Gmail push notifications.

Gmail publishes to a Pub/Sub topic whenever the mailbox changes; a push
subscription POSTs those to serve()'s endpoint (exposed through a tunnel such
as Tailscale Funnel or Cloudflare Tunnel). A notification only carries
{emailAddress, historyId}, so all it does is wake the sync worker, which runs
incremental_sync() from the stored cursor -- duplicate or out-of-order
notifications are harmless.

The worker also renews the Gmail watch daily (it expires after 7 days) and
syncs every BACKSTOP_EVERY_S regardless, since Gmail can drop notifications.

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

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2 import id_token

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

    def __init__(self, service, account: str, topic: str, db: Path | str | None = None):
        super().__init__(name='sync', daemon=True)
        self.service = service
        self.account = account
        self.topic = topic
        self.db = db
        self._wake = threading.Event()

    def notify(self) -> None:
        self._wake.set()

    def run(self) -> None:
        next_renew = 0.0
        with Store(self.db) as store:
            while True:
                if time.monotonic() >= next_renew:
                    try:
                        start_watch(self.service, self.topic)
                        next_renew = time.monotonic() + RENEW_EVERY_S
                    except Exception:
                        log.exception('watch renewal failed')
                        next_renew = time.monotonic() + RENEW_RETRY_S
                # Clear before syncing, so a notification that lands mid-sync
                # triggers another pass instead of being lost.
                self._wake.clear()
                try:
                    incremental_sync(self.service, store, self.account)
                except Exception:
                    log.exception('sync failed')
                if self._wake.wait(BACKSTOP_EVERY_S):
                    time.sleep(COALESCE_S)


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
          db: Path | str | None = None) -> None:
    account = service.users().getProfile(userId='me').execute()['emailAddress']
    worker = SyncWorker(service, account, topic, db)
    worker.start()

    server = ThreadingHTTPServer(
        (host, port), make_handler(path, PushVerifier(audience, service_account), worker.notify)
    )
    log.info('listening on http://%s:%d%s for %s', host, port, path, account)
    try:
        server.serve_forever()
    finally:
        server.server_close()
