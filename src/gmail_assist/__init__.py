import argparse
import logging
import os

from dotenv import load_dotenv

from gmail_assist.gmail.auth import get_service


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
          host=args.host, port=args.port, path=args.path, db=args.db)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog='gmail-assist')
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('labels', help='list Gmail labels')
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
    p.add_argument('--db', help='store path (default: ~/.local/share/gmail_assist/store.db)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if args.command == 'sync':
        sync(args)
    elif args.command == 'serve':
        serve(args, parser)
    else:
        labels()
