"""Claim pending messages, run each through the graph, record what it decided.

Dry run by design: this only writes to the local store (decisions, list
flags, message status). Nothing touches Gmail until the applier runs.

Each message is also traced to LangSmith (project 'gmail-assist') when
LANGSMITH_API_KEY is set in .env. Traces include the email contents the
model sees; set LANGSMITH_TRACING=false to turn tracing off with the key set.
"""
import logging
import os
import time
from typing import Any

import langsmith as ls
from langchain_core.tracers.langchain import wait_for_all_tracers

from gmail_assist.agent.graph import build_graph
from gmail_assist.agent.tracing import LangSmithLogHandler
from gmail_assist.agent.state import Context, EmailState, initial_state
from gmail_assist.store import Kind, Message, Store

log = logging.getLogger(__name__)
# Per-node progress, shown with `gmail-assist -v process`.
trace = logging.getLogger('gmail_assist.agent.trace')

# The longest legitimate path is about 7 steps (with one escalation). A
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


def run_graph(graph, msg: Message, ctx: Context) -> EmailState:
    """Run one message through the graph, logging each node as it finishes.
    Returns the final state, like graph.invoke()."""
    trace.info('▶ %s [%s] %s', msg.id, msg.kind, (msg.subject or '')[:70])
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
    for mode, chunk in graph.stream(
        initial_state(msg), context=ctx, config=config, stream_mode=['updates', 'values'],
    ):
        if mode == 'values':
            final = chunk
            continue
        now = time.monotonic()
        for node, update in chunk.items():
            trace.info('  %-20s %5.1fs  %s', node, now - last, _summarize(update or {}))
        last = now
    trace.info('  done in %.1fs', time.monotonic() - started)
    return final


def process(service, store: Store, limit: int = 20, kind: Kind | None = None) -> dict[str, int]:
    account = service.users().getProfile(userId='me').execute()['emailAddress']
    ctx = Context(store=store, account=account, labels=user_labels(service))
    graph = build_graph()

    traced = tracing_enabled()
    app_logger = logging.getLogger('gmail_assist')
    log_handler = LangSmithLogHandler()
    if traced:
        log.info('tracing to LangSmith project %r', LANGSMITH_PROJECT)
        app_logger.addHandler(log_handler)
    try:
        with ls.tracing_context(enabled=traced, project_name=LANGSMITH_PROJECT):
            return _process_batch(graph, store, ctx, limit, kind)
    finally:
        if traced:
            app_logger.removeHandler(log_handler)
            # Traces upload in the background; don't let the CLI exit first.
            wait_for_all_tracers()


def _process_batch(graph, store: Store, ctx: Context, limit: int,
                   kind: Kind | None) -> dict[str, int]:
    account = ctx.account
    counts = {'done': 0, 'failed': 0}
    for msg in store.claim_batch(limit, account=account, kind=kind):
        try:
            out = run_graph(graph, msg, ctx)
            for d in out['decisions']:
                store.record_decision(account, msg.id, **d)
            if flag := out.get('subscription_flag'):
                store.set_subscription_status(
                    account, msg.subscription_key, flag['status'],
                    by=flag['by'], reason=flag['reason'], only_from='active',
                )
            store.mark_done(account, msg.id, result={
                'decisions': out['decisions'], 'attempts': out.get('attempts', []),
            })
            counts['done'] += 1
        except Exception as err:
            log.exception('message %s failed', msg.id)
            store.mark_failed(account, msg.id, f'{type(err).__name__}: {err}')
            counts['failed'] += 1
    return counts
