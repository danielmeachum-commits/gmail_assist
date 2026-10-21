"""Claim pending messages, run each through the graph, record what it decided.

Dry run by design: this only writes to the local store (decisions, list
flags, message status). Nothing touches Gmail until the applier runs.

Each message is also traced to LangSmith (project 'gmail-assist') when
LANGSMITH_API_KEY is set in .env. Traces include the email contents the
model sees; set LANGSMITH_TRACING=false to turn tracing off with the key set.
"""
import contextvars
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import langsmith as ls
from langchain_core.tracers.langchain import wait_for_all_tracers

from gmail_assist.agent.graph import build_graph
from gmail_assist.agent.labels import DEFINITIONS
from gmail_assist.agent.tracing import LangSmithLogHandler
from gmail_assist.agent.state import Context, EmailState, initial_state
from gmail_assist.store import Kind, Message, Store

log = logging.getLogger(__name__)
# Per-node progress, shown with `gmail-assist -v process`.
trace = logging.getLogger('gmail_assist.agent.trace')

# A proposed sub-label is approved automatically (when the label_creation
# setting is 'auto') once this many messages from this many senders support it.
PROPOSAL_MIN_MESSAGES = 3
PROPOSAL_MIN_SENDERS = 2

# The longest legitimate path is 10 steps (with one escalation). A
# tight limit means a routing bug that loops classify -> escalate fails the
# message after a few model calls instead of LangGraph's default 25 steps.
RECURSION_LIMIT = 12

LANGSMITH_PROJECT = 'gmail-assist'


def tracing_enabled() -> bool:
    return (bool(os.environ.get('LANGSMITH_API_KEY'))
            and os.environ.get('LANGSMITH_TRACING', '').lower() != 'false')


def user_labels(service) -> list[str]:
    labels = service.users().labels().list(userId='me').execute().get('labels', [])
    return sorted(l['name'] for l in labels if l['type'] == 'user')


def load_definitions(store: Store, account: str, gmail_labels: list[str]) -> dict[str, str]:
    """The labels models may choose (see Context.definitions), after adding
    any new or changed definitions from labels.py to the store."""
    store.seed_definitions(account, DEFINITIONS)
    gmail = set(gmail_labels)
    # Labels added here (an approved proposal, or one you created in the
    # dashboard) may not be in Gmail yet: the applier creates them.
    return {d.name: d.description for d in store.definitions(account)
            if d.name in gmail or d.source in ('proposal', 'user')}


def approve_ready_proposals(store: Store, account: str) -> None:
    """With label_creation 'auto', approve proposals with enough support."""
    if store.get_setting(account, 'label_creation') != 'auto':
        return
    for p in store.proposals(account, status='pending'):
        if p['messages'] >= PROPOSAL_MIN_MESSAGES and p['senders'] >= PROPOSAL_MIN_SENDERS:
            n = store.decide_proposal(account, p['proposal'].id, 'approved', by='auto')
            log.info('approved sub-label %r; relabelled %d messages', p['proposal'].name, n)


def _summarize(update: dict[str, Any]) -> str:
    """One line describing what a node changed."""
    parts = []
    for key, value in update.items():
        if key == 'attempts':
            parts += [f"{a['model']} -> {a['classification']['label']!r} "
                      f"@ {a['classification']['confidence']:.2f}" for a in value]
        elif key == 'decisions':
            parts += [f"{d['action']}" + (f" {d['label']!r}" if d.get('label') else '')
                      + f" ({d['decided_by']})" for d in value]
        elif isinstance(value, dict):
            parts.append(f'{key}=' + ', '.join(
                f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in value.items()))
        else:
            parts.append(f'{key}={value}')
    return '; '.join(parts) or '(no changes)'


def run_graph(graph, msg: Message, ctx: Context, live: bool = True) -> EmailState:
    """Run one message through the graph, logging each node as it finishes.
    Returns the final state, like graph.invoke().

    With live=False the trace lines are held and printed together at the
    end, so messages running in parallel don't interleave their output.
    """
    lines: list[str] = []

    def emit(fmt: str, *args) -> None:
        if live:
            trace.info(fmt, *args)
        else:
            lines.append(fmt % args)

    emit('▶ %s [%s] %s', msg.id, msg.kind, (msg.subject or '')[:70])
    started = last = time.monotonic()
    final: EmailState = {}
    config = {
        'recursion_limit': RECURSION_LIMIT,
        # How the run shows up in LangSmith: filter by tag or metadata there.
        'run_name': f'triage {msg.kind}',
        'tags': [msg.kind],
        'metadata': {'message_id': msg.id, 'kind': msg.kind,
                     'subscription_key': msg.subscription_key},
    }
    try:
        for mode, chunk in graph.stream(
            initial_state(msg), context=ctx, config=config, stream_mode=['updates', 'values'],
        ):
            if mode == 'values':
                final = chunk
                continue
            now = time.monotonic()
            for node, update in chunk.items():
                emit('  %-20s %5.1fs  %s', node, now - last, _summarize(update or {}))
            last = now
        emit('  done in %.1fs', time.monotonic() - started)
    finally:
        # On failure too, so the steps that did run are visible.
        if lines:
            trace.info('\n'.join(lines))
    return final


def process(service, store: Store, limit: int = 20, kind: Kind | None = None,
            workers: int = 4, synced_since: datetime | None = None) -> dict[str, int]:
    """`workers` messages run through the graph at once. Match it to the
    local server's parallel slots (llama-server -np N); more workers than
    slots just queue at the server. `synced_since` only takes messages synced
    at or after that time (serve uses it to skip the backlog)."""
    # The Gmail client isn't thread-safe: use it here, before any workers.
    account = service.users().getProfile(userId='me').execute()['emailAddress']
    ctx = Context(store=store, account=account,
                  definitions=load_definitions(store, account, user_labels(service)))
    graph = build_graph()

    traced = tracing_enabled()
    app_logger = logging.getLogger('gmail_assist')
    log_handler = LangSmithLogHandler()
    if traced:
        log.info('tracing to LangSmith project %r', LANGSMITH_PROJECT)
        app_logger.addHandler(log_handler)
    try:
        with ls.tracing_context(enabled=traced, project_name=LANGSMITH_PROJECT):
            messages = store.claim_batch(limit, account=account, kind=kind,
                                         synced_since=synced_since)
            if workers <= 1:
                outcomes = [_process_one(graph, store, ctx, msg, live=True) for msg in messages]
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='triage') as pool:
                    # Threads don't inherit context variables, which is where
                    # tracing_context keeps its settings: give each task its
                    # own copy so traces still go to the right project.
                    futures = [
                        pool.submit(contextvars.copy_context().run,
                                    _process_one, graph, store, ctx, msg, False)
                        for msg in messages
                    ]
                    outcomes = [f.result() for f in futures]
    finally:
        if traced:
            app_logger.removeHandler(log_handler)
            # Traces upload in the background; don't let the CLI exit first.
            wait_for_all_tracers()
    return {'done': outcomes.count('done'), 'failed': outcomes.count('failed')}


def _process_one(graph, store: Store, ctx: Context, msg: Message, live: bool) -> str:
    """Run one message and record the outcome. Returns 'done' or 'failed'."""
    account = ctx.account
    try:
        out = run_graph(graph, msg, ctx, live=live)
        for d in out['decisions']:
            store.record_decision(account, msg.id, **d)
        if any(d.get('proposal') for d in out['decisions']):
            approve_ready_proposals(store, account)
        if attention := out.get('attention'):
            store.set_attention(account, msg.id, attention['level'], attention['by'])
        if flag := out.get('subscription_flag'):
            store.set_subscription_status(
                account, msg.subscription_key, flag['status'],
                by=flag['by'], reason=flag['reason'], only_from='active',
            )
        store.mark_done(account, msg.id, result={
            'decisions': out['decisions'], 'attempts': out.get('attempts', []),
            'refinement': out.get('refinement'), 'attention': out.get('attention'),
        })
        return 'done'
    except Exception as err:
        # One line per failure: when the model server is down every message
        # fails the same way. The traceback is in LangSmith, or with DEBUG.
        log.error('message %s failed: %s: %s', msg.id, type(err).__name__, err)
        log.debug('traceback for %s', msg.id, exc_info=True)
        store.mark_failed(account, msg.id, f'{type(err).__name__}: {err}')
        return 'failed'
