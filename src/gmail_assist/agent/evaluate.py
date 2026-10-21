"""Reference set for measuring triage accuracy.

Your earlier project left ~3,000 labelled emails in Gmail, older than the
main store's sync window. `fetch` copies a sample of them per label into a
separate store (never the main one: serve would triage anything newly synced
there). `reference` has Claude Opus label each under the current DEFINITIONS
and writes reference.json next to it, recording the old label and Opus's
answer. Where they agree the label is trusted; where they don't, you decide.

Everything lives in ~/.local/share/gmail_assist/eval/ (it holds email
contents, so it stays out of the repo).
"""
import contextvars
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import langsmith as ls
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from pydantic import BaseModel, Field, create_model
from sqlalchemy import select

from gmail_assist.agent.labels import DEFINITIONS, EXAMPLES, GUIDANCE, automated, choices
from gmail_assist.agent.llm import Models
from gmail_assist.agent.nodes import render_email
from gmail_assist.agent.run import tracing_enabled
from gmail_assist.agent.state import initial_state
from gmail_assist.store import Message, Store, _data_dir
from gmail_assist.sync import fetch_messages

log = logging.getLogger(__name__)

REFERENCE_MODEL = 'anthropic:claude-opus-5-5'
REFERENCE_EFFORT = 'high'
NONE = 'NONE'


def eval_dir() -> Path:
    path = _data_dir() / 'eval'
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def eval_store() -> Store:
    return Store(eval_dir() / 'reference.db')


def _gmail_labels(service) -> dict[str, str]:
    """User label id -> name."""
    return {l['id']: l['name'] for l in service.users().labels().list(userId='me').execute()['labels']
            if l['type'] == 'user'}


def old_label(msg: Message, names: dict[str, str]) -> str | None:
    """The most specific defined label the earlier project gave this message."""
    labels = [names[i] for i in msg.label_ids if names.get(i) in DEFINITIONS]
    return max(labels, key=lambda l: l.count('/'), default=None)


def fetch(service, per_label: int = 25) -> dict[str, int]:
    """Copy the `per_label` most recent messages of each defined label into
    the eval store. Returns messages stored per label."""
    account = service.users().getProfile(userId='me').execute()['emailAddress']
    names = _gmail_labels(service)
    store = eval_store()
    counts = {}
    for label_id, name in sorted(names.items(), key=lambda kv: kv[1]):
        if name not in DEFINITIONS:
            continue
        ids = [m['id'] for m in service.users().messages().list(
            userId='me', labelIds=[label_id], maxResults=per_label).execute().get('messages', [])]
        counts[name] = store.upsert_messages(account, fetch_messages(service, ids)) if ids else 0
        log.info('%-45s %d', name, counts[name])
    return counts


def _reference_schema(options: list[str]) -> type[BaseModel]:
    return create_model(
        'ReferenceLabel',
        label=(Literal[tuple(options + [NONE])],
               Field(description=f'The single best label, or {NONE} if none fits.')),
        confidence=(float, Field(ge=0, le=1, description='How sure you are, from 0 to 1.')),
        reason=(str, Field(description='One short sentence explaining the choice.')),
    )


def reference_prompt(options: list[str]) -> str:
    listed = '\n'.join(f'- {label}: {DEFINITIONS[label]}' for label in options)
    return f'You sort emails into Gmail labels.\n\n{GUIDANCE}\n\nLabels:\n{listed}'


def reference(service, workers: int = 4, relabel: list[str] | None = None) -> Path:
    """Label every eval message with Opus; write reference.json. Messages
    already labelled are kept, except those whose reference is in `relabel`
    (e.g. after changing those labels' definitions)."""
    names = _gmail_labels(service)
    definitions = {k: v for k, v in DEFINITIONS.items() if k in set(names.values())}
    store = eval_store()
    models = Models()
    llm = models.get(REFERENCE_MODEL, effort=REFERENCE_EFFORT)
    with store.ReadSession() as s:
        messages = list(s.scalars(select(Message)))

    def label_one(msg: Message) -> tuple[str, dict | None]:
        options = choices(definitions, automated(msg.kind, msg.sender_email))
        try:
            result = llm.structured(_reference_schema(options)).invoke([
                SystemMessage(reference_prompt(options)),
                HumanMessage(render_email(initial_state(msg))),
            ])
        except Exception as err:
            # Skipped, not recorded: the next run retries it.
            log.error('reference label failed for %s: %s: %s', msg.id, type(err).__name__, err)
            return msg.id, None
        return msg.id, {
            'old_label': old_label(msg, names),
            'reference': result.label,
            'confidence': result.confidence,
            'reason': result.reason,
            'kind': msg.kind,
            'sender': msg.sender_email,
            'subject': msg.subject,
            'thread_id': msg.thread_id,
        }

    out = eval_dir() / 'reference.json'
    results: dict[str, dict] = json.loads(out.read_text()) if out.exists() else {}
    if relabel:
        results = {k: v for k, v in results.items() if v['reference'] not in relabel}
    todo = [m for m in messages if m.id not in results]
    log.info('labelling %d messages with %s (%d already done)', len(todo), REFERENCE_MODEL, len(results))
    with ls.tracing_context(enabled=tracing_enabled(), project_name='gmail-assist-eval'):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Each task gets a copy of the context so tracing settings reach it.
            futures = [pool.submit(contextvars.copy_context().run, label_one, m) for m in todo]
            for i, future in enumerate(as_completed(futures), 1):
                msg_id, row = future.result()
                if row is not None:
                    results[msg_id] = row
                if i % 10 == 0 or i == len(todo):
                    # Save as we go: a crash or Ctrl-C keeps what's paid for.
                    _write_private(out, json.dumps(results, indent=1))
                    log.info('labelled %d/%d', i, len(todo))
    wait_for_all_tracers()
    return out


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(text)


def score(service, workers: int = 4) -> dict:
    """Run the local model's vote on every reference email and compare with
    the reference labels. Writes score.json (per-email results) and returns
    the summary. Local model only: costs nothing."""
    from gmail_assist.agent import nodes

    gmail_names = set(_gmail_labels(service).values())
    definitions = {k: v for k, v in DEFINITIONS.items() if k in gmail_names}
    ref = json.loads((eval_dir() / 'reference.json').read_text())
    store = eval_store()
    # Emails shown to the model as examples would score themselves.
    shown = {tuple(ex) for exs in EXAMPLES.values() for ex in exs}
    with store.ReadSession() as s:
        messages = [m for m in s.scalars(select(Message))
                    if m.id in ref and (m.sender_email, (m.subject or '').strip()) not in shown]
    llm = Models().get(nodes.LOCAL_MODEL)

    def score_one(msg: Message) -> dict:
        options = choices(definitions, automated(msg.kind, msg.sender_email))
        try:
            attempt = nodes._vote(llm, initial_state(msg), options, definitions)
        except Exception as err:
            log.error('vote failed for %s: %s: %s', msg.id, type(err).__name__, err)
            return {'id': msg.id, 'error': str(err)}
        c = attempt['classification']
        decided = c['label'] != nodes.NONE and c['confidence'] >= nodes.LOCAL_MIN_AGREEMENT
        return {'id': msg.id, 'reference': ref[msg.id]['reference'], 'local': c['label'],
                'votes': attempt['votes'], 'decided': decided,
                'correct': c['label'] == ref[msg.id]['reference'],
                'sender': msg.sender_email, 'subject': msg.subject}

    with ls.tracing_context(enabled=tracing_enabled(), project_name='gmail-assist-eval'):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(contextvars.copy_context().run, score_one, m) for m in messages]
            rows = []
            for i, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if i % 25 == 0 or i == len(futures):
                    log.info('scored %d/%d', i, len(futures))
    wait_for_all_tracers()
    _write_private(eval_dir() / 'score.json', json.dumps(rows, indent=1))

    ok = [r for r in rows if 'error' not in r]
    decided = [r for r in ok if r['decided']]
    return {
        'emails': len(ok),
        'errors': len(rows) - len(ok),
        'decided_locally': len(decided),
        'accuracy_when_decided': sum(r['correct'] for r in decided) / len(decided) if decided else None,
        'escalated': len(ok) - len(decided),
        'accuracy_if_forced': sum(r['correct'] for r in ok) / len(ok) if ok else None,
    }
