import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from gmail_assist.gmail.auth import AuthRequired, get_credentials, get_service

# sysexits.h EX_CONFIG. The systemd unit won't restart on it: restarting
# can't fix a missing sign-in.
EXIT_AUTH = 78


def labels() -> None:
    service = get_service('read')

    profile = service.users().getProfile(userId='me').execute()
    print(f"Signed in as {profile['emailAddress']}")

    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        print('No labels found.')
    else:
        print('Labels:')
        for label in labels:
            print(f"- {label['name']}")


def login(args: argparse.Namespace) -> None:
    get_credentials(args.access, interactive=True)
    print(f'Signed in: {args.access} access is ready.')


def sync(args: argparse.Namespace) -> None:
    from gmail_assist.store import Store
    from gmail_assist.sync import sync

    with Store(args.db) as store:
        n = sync(get_service('read'), store, full=args.full, query=args.query, limit=args.limit)
        print(f'Synced {n} new messages. Queue: {store.counts()}')


def serve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from gmail_assist.watch import serve

    if not args.topic or not args.audience:
        parser.error('serve needs --topic and --audience (or GMAIL_ASSIST_TOPIC / GMAIL_ASSIST_PUSH_AUDIENCE)')
    serve(get_service('read'), args.topic, args.audience, args.service_account,
          host=args.host, port=args.port, path=args.path, db=args.db,
          triage=not args.no_triage, workers=args.workers, dashboard_port=args.dashboard_port)


def dashboard_cmd(args: argparse.Namespace) -> None:
    import threading

    from gmail_assist import dashboard

    service = get_service('read')
    account = service.users().getProfile(userId='me').execute()['emailAddress']
    dashboard.start(account, args.db, port=args.port)
    print(f'Dashboard on http://127.0.0.1:{args.port} (Ctrl-C to stop)')
    threading.Event().wait()


def process(args: argparse.Namespace) -> None:
    from gmail_assist.agent.run import process
    from gmail_assist.store import Store

    with Store(args.db) as store:
        counts = process(get_service('read'), store, limit=args.limit, kind=args.kind,
                         workers=args.workers)
        print(f'Processed: {counts}. Queue: {store.counts()}')


def eval_cmd(args: argparse.Namespace) -> None:
    from gmail_assist.agent import evaluate

    service = get_service('read')
    if args.step == 'fetch':
        counts = evaluate.fetch(service, per_label=args.per_label)
        print(f'Stored {sum(counts.values())} reference emails in {evaluate.eval_dir()}')
    elif args.step == 'score':
        summary = evaluate.score(service, workers=args.workers)
        for key, value in summary.items():
            print(f'{key:22s} {value:.1%}' if isinstance(value, float) else f'{key:22s} {value}')
    elif args.step == 'reference':
        path = evaluate.reference(service, workers=args.workers, relabel=args.relabel)
        print(f'Reference labels: {path}')


def setup_logging(verbose: bool) -> None:
    console_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=console_level,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')
    for handler in logging.getLogger().handlers:
        handler.setLevel(console_level)
    # Our own INFO logs are always produced, so LangSmith gets them even
    # without -v; the console handler above still filters them.
    logging.getLogger('gmail_assist').setLevel(logging.INFO)
    # Per-request lines from the HTTP clients drown out everything else.
    for noisy in ('httpx', 'httpx2', 'httpcore', 'googleapiclient'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # The graph trace is already formatted as a readable tree: print it bare.
    trace = logging.getLogger('gmail_assist.agent.trace')
    trace.setLevel(logging.INFO if verbose else logging.WARNING)
    if not trace.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        trace.addHandler(handler)
    trace.propagate = False


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog='gmail-assist')
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('labels', help='list Gmail labels')
    p = sub.add_parser('login', help='sign in to Google for one grant (opens a browser)')
    p.add_argument('access', choices=['read', 'modify', 'calendar'],
                   help='read: Gmail read-only; modify: change Gmail; calendar: add calendar events')
    p = sub.add_parser('sync', help='pull new mail into the local store')
    p.add_argument('--full', action='store_true', help='ignore the saved cursor and resync the window')
    p.add_argument('--query', default='newer_than:90d', help='Gmail search for full syncs')
    p.add_argument('--limit', type=int, help='max messages for a full sync')
    p.add_argument('--db', help='store path (default: ~/.local/share/gmail_assist/store.db)')
    p = sub.add_parser('serve', help='receive Gmail push notifications and sync on each one')
    p.add_argument('--topic', default=os.environ.get('GMAIL_ASSIST_TOPIC'),
                   help='Pub/Sub topic Gmail publishes to (projects/<p>/topics/<t>)')
    p.add_argument('--audience', default=os.environ.get('GMAIL_ASSIST_PUSH_AUDIENCE'),
                   help="the subscription's push auth audience (defaults to the push endpoint URL)")
    p.add_argument('--service-account', default=os.environ.get('GMAIL_ASSIST_PUSH_SA'),
                   help='only accept pushes signed as this service account')
    p.add_argument('--host', default='127.0.0.1', help='bind address; the tunnel connects here')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--path', default='/gmail/push')
    p.add_argument('--no-triage', action='store_true',
                   help="only sync; don't run new mail through the triage graph")
    p.add_argument('--workers', type=int, default=4,
                   help='messages triaged in parallel (default 4); match llama-server -np')
    p.add_argument('--dashboard-port', type=int, default=8090,
                   help='serve the dashboard on 127.0.0.1 at this port (default 8090; 0 = off)')
    p.add_argument('--db', help='store path (default: ~/.local/share/gmail_assist/store.db)')
    p = sub.add_parser('dashboard', help='serve only the dashboard (serve already includes it)')
    p.add_argument('--port', type=int, default=8090)
    p.add_argument('--db', help='store path (default: ~/.local/share/gmail_assist/store.db)')
    p = sub.add_parser('process', help='run pending messages through the triage graph (dry run)')
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--kind', choices=['personal', 'bulk'])
    p.add_argument('--workers', type=int, default=4,
                   help='messages processed in parallel (default 4); match llama-server -np')
    p.add_argument('--db', help='store path (default: ~/.local/share/gmail_assist/store.db)')
    p = sub.add_parser('eval', help='build and score the triage reference set')
    p.add_argument('step', choices=['fetch', 'reference', 'score'],
                   help='fetch: copy labelled emails into the eval store; '
                        'reference: label them with Claude Opus; '
                        'score: measure the local model against the reference')
    p.add_argument('--per-label', type=int, default=25)
    p.add_argument('--relabel', action='append', metavar='LABEL',
                   help='reference: redo emails currently referenced as LABEL (repeatable)')
    p.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    setup_logging(args.verbose)
    try:
        run_command(args, parser)
    except AuthRequired as err:
        print(f'gmail-assist: {err}', file=sys.stderr)
        sys.exit(EXIT_AUTH)


def run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command == 'login':
        login(args)
    elif args.command == 'sync':
        sync(args)
    elif args.command == 'serve':
        serve(args, parser)
    elif args.command == 'process':
        process(args)
    elif args.command == 'eval':
        eval_cmd(args)
    elif args.command == 'dashboard':
        dashboard_cmd(args)
    else:
        labels()
