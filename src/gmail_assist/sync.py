"""Pull Gmail changes into the local store.

Messages are fetched headers-first (format='metadata'). Anything with a
List-Unsubscribe header is bulk mail and stops there; everything else is
re-fetched with format='full' so the body is available for classification.

full_sync() walks messages.list for a query window; incremental_sync() replays
users.history.list from the stored historyId and falls back to a full sync
when Gmail has expired that history (roughly a week).
"""
import logging
import random
import time
from collections.abc import Iterable

from googleapiclient.errors import HttpError

from gmail_assist.gmail import parse
from gmail_assist.store import Store

log = logging.getLogger(__name__)

METADATA_HEADERS = [
    'From', 'To', 'Subject', 'Date', 'List-Id', 'List-Unsubscribe', 'List-Unsubscribe-Post',
]
# Gmail allows up to 100 calls per batch, but large batches trip the
# per-user rate limit quickly; 25 is a reasonable middle ground.
BATCH_SIZE = 25
# Gmail's per-user quota is 15,000 units/min (250/s) and messages.get costs 5.
# Batches start paced under that and slow down whenever Gmail pushes back.
QUOTA_UNITS_PER_S = 200
GET_COST = 5
MAX_RETRIES = 5
# The quota is per minute, so retries must wait long enough to reach a fresh
# window: 30s, 60s, then 120s.
RETRY_BASE_S = 30
RETRY_MAX_S = 120
# Never sync these: they aren't mail you received.
SKIP_LABELS = {'DRAFT', 'CHAT'}


def _chunks(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _is_retryable(err: HttpError) -> bool:
    # 403s cover both rateLimitExceeded and userRateLimitExceeded.
    return err.resp.status in (429, 500, 502, 503, 504) or (
        err.resp.status == 403 and b'ratelimitexceeded' in (err.content or b'').lower()
    )


class _Pacer:
    """Spaces batches to a units/s rate that halves when Gmail throttles
    and creeps back up while it doesn't."""

    def __init__(self, max_rate: float):
        self.max_rate = self.rate = max_rate

    def wait(self, units: int, started: float) -> None:
        time.sleep(max(0.0, units / self.rate - (time.monotonic() - started)))

    def throttled(self) -> None:
        self.rate = max(self.max_rate / 16, self.rate / 2)
        log.info('throttled; pacing at %.0f units/s', self.rate)

    def ok(self) -> None:
        self.rate = min(self.max_rate, self.rate * 1.05)


# Module-level so the learned rate carries across calls within a process.
_pacer = _Pacer(QUOTA_UNITS_PER_S)


def _batch_get(service, ids: list[str], fmt: str) -> list[dict]:
    """messages.get for many ids via batch HTTP, retrying rate-limited ones.
    Ids that 404 (deleted since they were listed) are dropped."""
    results: dict[str, dict] = {}
    todo = list(ids)
    for attempt in range(MAX_RETRIES + 1):
        retry: list[str] = []
        chunks = list(_chunks(todo, BATCH_SIZE))
        for n, chunk in enumerate(chunks):
            def callback(request_id, response, exception):
                if exception is None:
                    results[request_id] = response
                elif isinstance(exception, HttpError) and exception.resp.status == 404:
                    pass
                elif isinstance(exception, HttpError) and _is_retryable(exception):
                    retry.append(request_id)
                else:
                    raise exception

            batch = service.new_batch_http_request(callback=callback)
            for id in chunk:
                kwargs = {'userId': 'me', 'id': id, 'format': fmt}
                if fmt == 'metadata':
                    kwargs['metadataHeaders'] = METADATA_HEADERS
                batch.add(service.users().messages().get(**kwargs), request_id=id)
            started = time.monotonic()
            batch.execute()
            _pacer.wait(len(chunk) * GET_COST, started)
            if retry:
                # The quota window is spent: stop sending (every call made
                # now just fails) and retry the rest after the backoff.
                _pacer.throttled()
                retry += [id for rest in chunks[n + 1:] for id in rest]
                break
            _pacer.ok()
        if not retry:
            break
        if attempt == MAX_RETRIES:
            raise RuntimeError(f'gave up on {len(retry)} messages after {MAX_RETRIES} retries')
        delay = min(RETRY_BASE_S * 2 ** attempt, RETRY_MAX_S) + random.uniform(0, 5)
        log.info('rate limited on %d messages, retrying in %.1fs', len(retry), delay)
        time.sleep(delay)
        todo = retry
    return [results[id] for id in ids if id in results]


def fetch_messages(service, ids: list[str]) -> list[dict]:
    """Headers for everything; full bodies only for non-bulk mail."""
    metadata = [
        m for m in _batch_get(service, ids, 'metadata')
        if not SKIP_LABELS & set(m.get('labelIds', []))
    ]
    need_body = [m['id'] for m in metadata if parse.unsubscribe(m) is None]
    full = {m['id']: m for m in _batch_get(service, need_body, 'full')} if need_body else {}
    return [full.get(m['id'], m) for m in metadata]


def _store_new(service, store: Store, account: str, ids: list[str]) -> int:
    n = 0
    for i, chunk in enumerate(_chunks(ids, 100)):
        n += store.upsert_messages(account, fetch_messages(service, chunk))
        if len(ids) > 100:
            log.info('fetched %d/%d', min((i + 1) * 100, len(ids)), len(ids))
    return n


def full_sync(service, store: Store, account: str, query: str = 'newer_than:90d',
              limit: int | None = None) -> int:
    """Sync every message matching `query` (Gmail search syntax)."""
    # Take the cursor before listing so anything arriving mid-sync is picked
    # up by the next incremental sync rather than missed.
    history_id = service.users().getProfile(userId='me').execute()['historyId']

    ids: list[str] = []
    request = service.users().messages().list(userId='me', q=query, maxResults=500)
    while request is not None:
        response = request.execute()
        ids += [m['id'] for m in response.get('messages', [])]
        if limit and len(ids) >= limit:
            ids = ids[:limit]
            break
        request = service.users().messages().list_next(request, response)

    # Skip what's already stored, so an interrupted full sync resumes where
    # it stopped instead of refetching everything.
    known: set[str] = set()
    for chunk in _chunks(ids, 10_000):
        known |= store.known_ids(account, chunk)
    log.info('full sync: %d messages match %r, %d already stored', len(ids), query, len(known))
    n = _store_new(service, store, account, [i for i in ids if i not in known])
    store.set_history_id(account, history_id)
    return n


def incremental_sync(service, store: Store, account: str, **full_sync_kwargs) -> int:
    """Apply changes since the stored historyId. Returns messages added."""
    start = store.get_history_id(account)
    if start is None:
        return full_sync(service, store, account, **full_sync_kwargs)

    added: list[str] = []
    deleted: set[str] = set()
    labels: dict[str, list[str]] = {}  # latest labelIds per message
    history_id = start
    request = service.users().history().list(userId='me', startHistoryId=start)
    try:
        while request is not None:
            response = request.execute()
            for record in response.get('history', []):
                for item in record.get('messagesAdded', []):
                    added.append(item['message']['id'])
                for item in record.get('messagesDeleted', []):
                    deleted.add(item['message']['id'])
                for key in ('labelsAdded', 'labelsRemoved'):
                    for item in record.get(key, []):
                        labels[item['message']['id']] = item['message'].get('labelIds', [])
            history_id = response.get('historyId', history_id)
            request = service.users().history().list_next(request, response)
    except HttpError as err:
        if err.resp.status != 404:
            raise
        log.warning('history %s expired; falling back to full sync', start)
        return full_sync(service, store, account, **full_sync_kwargs)

    added = list(dict.fromkeys(i for i in added if i not in deleted))
    # A new message's labels come from fetching it, not the history entry.
    for id in added:
        labels.pop(id, None)

    n = _store_new(service, store, account, added)
    store.set_labels(account, {i: l for i, l in labels.items() if i not in deleted})
    store.delete_messages(account, deleted)
    store.set_history_id(account, history_id)
    log.info('incremental sync: +%d new, %d label changes, -%d deleted',
             n, len(labels), len(deleted))
    return n


def sync(service, store: Store, full: bool = False, **full_sync_kwargs) -> int:
    account = service.users().getProfile(userId='me').execute()['emailAddress']
    if full:
        return full_sync(service, store, account, **full_sync_kwargs)
    return incremental_sync(service, store, account, **full_sync_kwargs)
