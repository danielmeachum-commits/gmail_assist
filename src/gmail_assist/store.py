"""Local SQLite store (via SQLAlchemy) for synced Gmail data, queued for batch processing.

Lives at ~/.local/share/gmail_assist/store.db (override with GMAIL_ASSIST_DB).

Two tables:
  - sync_state: per-account sync cursor (Gmail historyId) so the next sync
    can be incremental via users.history.list.
  - messages:   one row per synced message plus a processing status, used as
    a work queue: sync inserts rows as 'pending', a LangGraph batch job
    claims them with claim_batch() and marks each done or failed.
"""
import os
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import (
    JSON, BigInteger, CheckConstraint, DateTime, Engine, Index, String, Text,
    create_engine, delete, event, func, or_, select, tuple_, update,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

Status = Literal['pending', 'processing', 'done', 'failed']


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
    )

    account: Mapped[str] = mapped_column(primary_key=True)
    id: Mapped[str] = mapped_column(primary_key=True)
    thread_id: Mapped[str]
    history_id: Mapped[str | None]
    internal_date: Mapped[int | None] = mapped_column(BigInteger)  # ms since epoch, as Gmail returns it
    sender: Mapped[str | None] = mapped_column(Text)
    recipients: Mapped[str | None] = mapped_column(Text)
    subject: Mapped[str | None] = mapped_column(Text)
    snippet: Mapped[str | None] = mapped_column(Text)
    label_ids: Mapped[list[str]] = mapped_column(default=list)
    payload: Mapped[dict[str, Any] | None]  # raw Gmail message resource
    synced_at: Mapped[datetime]

    status: Mapped[str] = mapped_column(String(16), default='pending', server_default='pending')
    attempts: Mapped[int] = mapped_column(default=0, server_default='0')
    claimed_at: Mapped[datetime | None]
    processed_at: Mapped[datetime | None]
    result: Mapped[dict[str, Any] | None]  # output of processing
    error: Mapped[str | None] = mapped_column(Text)


def _data_dir() -> Path:
    base = os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share'
    return Path(base) / 'gmail_assist'


def default_db_path() -> Path:
    return Path(os.environ.get('GMAIL_ASSIST_DB') or _data_dir() / 'store.db')


def _header(message: dict, name: str) -> str | None:
    for h in message.get('payload', {}).get('headers', []):
        if h['name'].lower() == name.lower():
            return h['value']
    return None


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
        re-syncing doesn't reprocess everything. Returns rows written.
        """
        now = _now()
        rows = [
            {
                'account': account,
                'id': m['id'],
                'thread_id': m['threadId'],
                'history_id': m.get('historyId'),
                'internal_date': int(m['internalDate']) if 'internalDate' in m else None,
                'sender': _header(m, 'From'),
                'recipients': _header(m, 'To'),
                'subject': _header(m, 'Subject'),
                'snippet': m.get('snippet'),
                'label_ids': m.get('labelIds', []),
                'payload': m,
                'synced_at': now,
            }
            for m in messages
        ]
        if not rows:
            return 0
        stmt = insert(Message)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Message.account, Message.id],
            set_={
                'history_id': stmt.excluded.history_id,
                'label_ids': stmt.excluded.label_ids,
                'payload': stmt.excluded.payload,
                'synced_at': stmt.excluded.synced_at,
            },
        )
        with self.Session.begin() as s:
            s.execute(stmt, rows)
        return len(rows)

    def delete_messages(self, account: str, ids: Iterable[str]) -> None:
        """Drop messages deleted upstream (history 'messagesDeleted')."""
        with self.Session.begin() as s:
            s.execute(delete(Message).where(Message.account == account, Message.id.in_(list(ids))))

    # -- batch processing queue ---------------------------------------------

    def claim_batch(self, limit: int = 50, account: str | None = None) -> list[Message]:
        """Atomically move up to `limit` pending messages to 'processing'."""
        pending = select(Message.account, Message.id).where(Message.status == 'pending')
        if account:
            pending = pending.where(Message.account == account)
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
