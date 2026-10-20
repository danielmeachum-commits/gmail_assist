"""Local SQLite store (via SQLAlchemy) for synced Gmail data, queued for batch processing.

Lives at ~/.local/share/gmail_assist/store.db (override with GMAIL_ASSIST_DB).

Tables:
  - sync_state:    per-account sync cursor (Gmail historyId) so the next sync
                   can be incremental via users.history.list.
  - messages:      one row per synced message plus a processing status, used as
                   a work queue: sync inserts rows as 'pending', a LangGraph batch
                   job claims them with claim_batch() and marks each done or failed.
  - decisions:     append-only log of what was decided about each message (by a
                   rule or the LLM) and whether/when it was applied to Gmail.
  - subscriptions: one row per mailing list (List-Id, else sender address) with
                   its List-Unsubscribe targets and a status used to flag lists
                   for unsubscribing.
"""
import os
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import (
    JSON, BigInteger, CheckConstraint, DateTime, Engine, Index, String, Text,
    case, create_engine, delete, event, func, or_, select, tuple_, update,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from gmail_assist.gmail import parse

Status = Literal['pending', 'processing', 'done', 'failed']
Kind = Literal['bulk', 'personal']
# 'label':  add `label` (creating it if new_label) -- the main categorization
# 'expire': login code / sign-in link; trash once read or a day old
# 'trash', 'archive': act now
# 'none':   looked at, deliberately left alone
Action = Literal['label', 'expire', 'trash', 'archive', 'none']
SubscriptionStatus = Literal['active', 'flagged', 'keep', 'unsubscribed']


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSON,
        list[str]: JSON,
    }


class SyncState(Base):
    __tablename__ = 'sync_state'

    account: Mapped[str] = mapped_column(String, primary_key=True)
    history_id: Mapped[str | None]
    last_sync_at: Mapped[datetime | None]


class Message(Base):
    __tablename__ = 'messages'
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'done', 'failed')", name='status_valid'
        ),
        Index('messages_status', 'status', 'internal_date'),
        Index('messages_thread', 'account', 'thread_id'),
        Index('messages_subscription', 'account', 'subscription_key', 'internal_date'),
    )

    account: Mapped[str] = mapped_column(primary_key=True)
    id: Mapped[str] = mapped_column(primary_key=True)
    thread_id: Mapped[str]
    history_id: Mapped[str | None]
    internal_date: Mapped[int | None] = mapped_column(BigInteger)  # ms since epoch, as Gmail returns it
    sender: Mapped[str | None] = mapped_column(Text)
    sender_email: Mapped[str | None]
    recipients: Mapped[str | None] = mapped_column(Text)
    subject: Mapped[str | None] = mapped_column(Text)
    snippet: Mapped[str | None] = mapped_column(Text)
    label_ids: Mapped[list[str]] = mapped_column(default=list)
    unread: Mapped[bool] = mapped_column(default=False)
    # 'bulk' = has List-Unsubscribe; only headers are fetched for these.
    kind: Mapped[str] = mapped_column(String(16))
    subscription_key: Mapped[str | None]
    has_body: Mapped[bool] = mapped_column(default=False)
    payload: Mapped[dict[str, Any] | None]  # raw Gmail message resource
    synced_at: Mapped[datetime]

    status: Mapped[str] = mapped_column(String(16), default='pending', server_default='pending')
    attempts: Mapped[int] = mapped_column(default=0, server_default='0')
    claimed_at: Mapped[datetime | None]
    processed_at: Mapped[datetime | None]
    result: Mapped[dict[str, Any] | None]  # output of processing
    error: Mapped[str | None] = mapped_column(Text)

    def body_text(self) -> str | None:
        return parse.body_text(self.payload) if self.has_body and self.payload else None


class Decision(Base):
    __tablename__ = 'decisions'
    __table_args__ = (
        CheckConstraint(
            "action IN ('label', 'expire', 'trash', 'archive', 'none')", name='action_valid'
        ),
        Index('decisions_message', 'account', 'message_id'),
        Index('decisions_unapplied', 'applied_at'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account: Mapped[str]
    message_id: Mapped[str]
    action: Mapped[str] = mapped_column(String(16))
    label: Mapped[str | None]                 # e.g. 'Finance/Receipts'
    new_label: Mapped[bool] = mapped_column(default=False)
    reason: Mapped[str | None] = mapped_column(Text)
    decided_by: Mapped[str]                   # 'rule:<name>' or 'llm'
    model: Mapped[str | None]
    confidence: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(default=_now)
    applied_at: Mapped[datetime | None]
    apply_error: Mapped[str | None] = mapped_column(Text)


class Subscription(Base):
    __tablename__ = 'subscriptions'
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'flagged', 'keep', 'unsubscribed')", name='status_valid'
        ),
    )

    account: Mapped[str] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(primary_key=True)  # List-Id, else sender address
    sender: Mapped[str | None] = mapped_column(Text)
    sender_email: Mapped[str | None]
    unsubscribe_urls: Mapped[list[str]] = mapped_column(default=list)
    unsubscribe_mailtos: Mapped[list[str]] = mapped_column(default=list)
    one_click: Mapped[bool] = mapped_column(default=False)
    first_seen: Mapped[datetime]
    last_seen: Mapped[datetime]
    status: Mapped[str] = mapped_column(String(16), default='active', server_default='active')
    status_reason: Mapped[str | None] = mapped_column(Text)
    status_by: Mapped[str | None]             # 'user', 'llm', 'rule:<name>'
    status_at: Mapped[datetime | None]


def _data_dir() -> Path:
    base = os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share'
    return Path(base) / 'gmail_assist'


def default_db_path() -> Path:
    return Path(os.environ.get('GMAIL_ASSIST_DB') or _data_dir() / 'store.db')


def make_engine(path: Path) -> Engine:
    engine = create_engine(f'sqlite:///{path}')

    # pysqlite's own transaction handling defers BEGIN and ignores the lock
    # mode; take it over (SQLAlchemy's documented recipe) and use IMMEDIATE so
    # two workers can't claim the same pending rows.
    @event.listens_for(engine, 'connect')
    def _on_connect(dbapi_conn, _record):
        dbapi_conn.isolation_level = None
        dbapi_conn.execute('PRAGMA journal_mode=WAL')
        dbapi_conn.execute('PRAGMA busy_timeout=5000')

    @event.listens_for(engine, 'begin')
    def _on_begin(conn):
        conn.exec_driver_sql('BEGIN IMMEDIATE')

    return engine


def _message_row(account: str, m: dict, now: datetime) -> dict[str, Any]:
    unsub = parse.unsubscribe(m)
    label_ids = m.get('labelIds', [])
    return {
        'account': account,
        'id': m['id'],
        'thread_id': m['threadId'],
        'history_id': m.get('historyId'),
        'internal_date': int(m['internalDate']) if 'internalDate' in m else None,
        'sender': parse.header(m, 'From'),
        'sender_email': parse.sender_email(m),
        'recipients': parse.header(m, 'To'),
        'subject': parse.header(m, 'Subject'),
        'snippet': m.get('snippet'),
        'label_ids': label_ids,
        'unread': 'UNREAD' in label_ids,
        'kind': 'bulk' if unsub else 'personal',
        'subscription_key': (parse.list_id(m) or parse.sender_email(m)) if unsub else None,
        # format='metadata' responses have no body parts at all.
        'has_body': 'parts' in m.get('payload', {}) or 'data' in m.get('payload', {}).get('body', {}),
        'payload': m,
        'synced_at': now,
    }


class Store:
    def __init__(self, path: Path | str | None = None):
        path = Path(path) if path else default_db_path()
        # Message bodies are private, so keep the directory user-only.
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.engine = make_engine(path)
        Base.metadata.create_all(self.engine)
        os.chmod(path, 0o600)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> 'Store':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- sync state ---------------------------------------------------------

    def get_history_id(self, account: str) -> str | None:
        with self.Session() as s:
            state = s.get(SyncState, account)
            return state.history_id if state else None

    def set_history_id(self, account: str, history_id: str) -> None:
        with self.Session.begin() as s:
            s.merge(SyncState(account=account, history_id=history_id, last_sync_at=_now()))

    # -- writing synced messages --------------------------------------------

    def upsert_messages(self, account: str, messages: Iterable[dict]) -> int:
        """Store Gmail message resources (from users.messages.get).

        New messages are queued as 'pending'. A message that was already
        stored has its data refreshed but keeps its processing status, so
        re-syncing doesn't reprocess everything. Bulk mail also updates its
        subscription row. Returns rows written.
        """
        now = _now()
        messages = list(messages)
        rows = [_message_row(account, m, now) for m in messages]
        if not rows:
            return 0
        stmt = insert(Message)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Message.account, Message.id],
            set_={
                col: getattr(stmt.excluded, col)
                for col in ('history_id', 'label_ids', 'unread', 'payload', 'has_body', 'synced_at')
            },
        )
        with self.Session.begin() as s:
            s.execute(stmt, rows)
            for m, row in zip(messages, rows):
                if row['subscription_key']:
                    self._touch_subscription(s, account, row, parse.unsubscribe(m))
        return len(rows)

    @staticmethod
    def _touch_subscription(s, account: str, row: dict, unsub: parse.Unsubscribe) -> None:
        seen = (
            datetime.fromtimestamp(row['internal_date'] / 1000, timezone.utc)
            if row['internal_date'] else row['synced_at']
        )
        stmt = insert(Subscription).values(
            account=account,
            key=row['subscription_key'],
            sender=row['sender'],
            sender_email=row['sender_email'],
            unsubscribe_urls=unsub.urls,
            unsubscribe_mailtos=unsub.mailtos,
            one_click=unsub.one_click,
            first_seen=seen,
            last_seen=seen,
        )
        newer = stmt.excluded.last_seen > Subscription.last_seen
        stmt = stmt.on_conflict_do_update(
            index_elements=[Subscription.account, Subscription.key],
            set_={
                # Only the newest message's headers describe the list now.
                **{
                    col: case((newer, getattr(stmt.excluded, col)), else_=getattr(Subscription, col))
                    for col in ('sender', 'sender_email', 'unsubscribe_urls',
                                'unsubscribe_mailtos', 'one_click', 'last_seen')
                },
                'first_seen': func.min(Subscription.first_seen, stmt.excluded.first_seen),
            },
        )
        s.execute(stmt)

    def set_labels(self, account: str, labels: dict[str, list[str]]) -> int:
        """Update label_ids for already-stored messages ({message_id: labelIds}),
        e.g. from history labelsAdded/labelsRemoved. Unknown ids are ignored."""
        with self.Session.begin() as s:
            n = 0
            for id, label_ids in labels.items():
                n += s.execute(
                    update(Message)
                    .where(Message.account == account, Message.id == id)
                    .values(label_ids=label_ids, unread='UNREAD' in label_ids)
                ).rowcount
        return n

    def delete_messages(self, account: str, ids: Iterable[str]) -> None:
        """Drop messages deleted upstream (history 'messagesDeleted')."""
        with self.Session.begin() as s:
            s.execute(delete(Message).where(Message.account == account, Message.id.in_(list(ids))))

    def known_ids(self, account: str, ids: Iterable[str]) -> set[str]:
        ids = list(ids)
        with self.Session() as s:
            return set(s.scalars(
                select(Message.id).where(Message.account == account, Message.id.in_(ids))
            ))

    # -- batch processing queue ---------------------------------------------

    def claim_batch(self, limit: int = 50, account: str | None = None,
                    kind: Kind | None = None) -> list[Message]:
        """Atomically move up to `limit` pending messages to 'processing'."""
        pending = select(Message.account, Message.id).where(Message.status == 'pending')
        if account:
            pending = pending.where(Message.account == account)
        if kind:
            pending = pending.where(Message.kind == kind)
        pending = pending.order_by(Message.internal_date).limit(limit)

        with self.Session.begin() as s:
            claimed = s.scalars(
                update(Message)
                .where(tuple_(Message.account, Message.id).in_(pending))
                .values(status='processing', claimed_at=_now(), attempts=Message.attempts + 1)
                .returning(Message),
                execution_options={'synchronize_session': False},
            ).all()
        # RETURNING order is unspecified; oldest first.
        return sorted(claimed, key=lambda m: m.internal_date or 0)

    def mark_done(self, account: str, id: str, result: dict[str, Any] | None = None) -> None:
        with self.Session.begin() as s:
            s.execute(
                update(Message)
                .where(Message.account == account, Message.id == id)
                .values(status='done', processed_at=_now(), result=result, error=None)
            )

    def mark_failed(self, account: str, id: str, error: str) -> None:
        with self.Session.begin() as s:
            s.execute(
                update(Message)
                .where(Message.account == account, Message.id == id)
                .values(status='failed', processed_at=_now(), error=error)
            )

    def requeue(self, *, failed: bool = True, stale_after_s: int | None = 3600,
                max_attempts: int = 3) -> int:
        """Put work back to 'pending': failed rows, and rows stuck in
        'processing' longer than stale_after_s (e.g. a crashed run).
        Rows that already hit max_attempts are left alone."""
        conds = []
        if failed:
            conds.append(Message.status == 'failed')
        if stale_after_s is not None:
            cutoff = _now() - timedelta(seconds=stale_after_s)
            conds.append((Message.status == 'processing') & (Message.claimed_at < cutoff))
        if not conds:
            return 0
        with self.Session.begin() as s:
            res = s.execute(
                update(Message)
                .where(or_(*conds), Message.attempts < max_attempts)
                .values(status='pending', claimed_at=None)
            )
        return res.rowcount

    def counts(self) -> dict[str, int]:
        with self.Session() as s:
            rows = s.execute(select(Message.status, func.count()).group_by(Message.status))
            return {status: n for status, n in rows}

    def get_message(self, account: str, id: str) -> Message | None:
        with self.Session() as s:
            return s.get(Message, (account, id))

    # -- decisions ----------------------------------------------------------

    def record_decision(self, account: str, message_id: str, action: Action, *,
                        decided_by: str, label: str | None = None, new_label: bool = False,
                        reason: str | None = None, model: str | None = None,
                        confidence: float | None = None) -> Decision:
        """Append a decision. Nothing touches Gmail here; the applier reads
        unapplied decisions later (and a dry run just never applies them)."""
        if action == 'label' and not label:
            raise ValueError("action='label' needs a label")
        decision = Decision(
            account=account, message_id=message_id, action=action, label=label,
            new_label=new_label, reason=reason, decided_by=decided_by, model=model,
            confidence=confidence,
        )
        with self.Session.begin() as s:
            s.add(decision)
        return decision

    def decisions_for(self, account: str, message_id: str) -> list[Decision]:
        with self.Session() as s:
            return list(s.scalars(
                select(Decision)
                .where(Decision.account == account, Decision.message_id == message_id)
                .order_by(Decision.created_at)
            ))

    def unapplied_decisions(self, account: str | None = None) -> list[Decision]:
        q = select(Decision).where(Decision.applied_at.is_(None), Decision.apply_error.is_(None))
        if account:
            q = q.where(Decision.account == account)
        with self.Session() as s:
            return list(s.scalars(q.order_by(Decision.created_at)))

    def mark_decision_applied(self, decision_id: int, error: str | None = None) -> None:
        with self.Session.begin() as s:
            s.execute(
                update(Decision)
                .where(Decision.id == decision_id)
                .values(applied_at=None if error else _now(), apply_error=error)
            )

    # -- subscriptions ------------------------------------------------------

    def subscription_stats(self, account: str, days: int = 90) -> list[dict[str, Any]]:
        """Per-list volume and read rate over the last `days`, busiest first.

        'Read' means the UNREAD label is gone, which is the only signal Gmail
        gives -- opening it, or marking it read, both count.
        """
        since_ms = int((_now() - timedelta(days=days)).timestamp() * 1000)
        q = (
            select(
                Subscription,
                func.count(Message.id).label('received'),
                func.coalesce(func.sum(case((Message.unread, 0), else_=1)), 0).label('read'),
                func.max(Message.internal_date).label('latest_ms'),
            )
            .outerjoin(Message, (Message.account == Subscription.account)
                       & (Message.subscription_key == Subscription.key)
                       & (Message.internal_date >= since_ms))
            .where(Subscription.account == account)
            .group_by(Subscription.account, Subscription.key)
            .order_by(func.count(Message.id).desc())
        )
        with self.Session() as s:
            return [
                {
                    'subscription': sub,
                    'received': received,
                    'read': read,
                    'read_rate': read / received if received else None,
                    'per_week': received / (days / 7),
                    'latest_ms': latest_ms,
                }
                for sub, received, read, latest_ms in s.execute(q)
            ]

    def set_subscription_status(self, account: str, key: str, status: SubscriptionStatus, *,
                                by: str, reason: str | None = None) -> None:
        with self.Session.begin() as s:
            res = s.execute(
                update(Subscription)
                .where(Subscription.account == account, Subscription.key == key)
                .values(status=status, status_reason=reason, status_by=by, status_at=_now())
            )
            if not res.rowcount:
                raise KeyError(f'no subscription {key!r} for {account}')

    def subscriptions(self, account: str, status: SubscriptionStatus | None = None) -> list[Subscription]:
        q = select(Subscription).where(Subscription.account == account)
        if status:
            q = q.where(Subscription.status == status)
        with self.Session() as s:
            return list(s.scalars(q.order_by(Subscription.key)))
