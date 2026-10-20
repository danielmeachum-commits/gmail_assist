import argparse
import logging

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


def main() -> None:
    parser = argparse.ArgumentParser(prog='gmail-assist')
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('labels', help='list Gmail labels')
    p = sub.add_parser('sync', help='pull new mail into the local store')
    p.add_argument('--full', action='store_true', help='ignore the saved cursor and resync the window')
    p.add_argument('--query', default='newer_than:90d', help='Gmail search for full syncs')
    p.add_argument('--limit', type=int, help='max messages for a full sync')
    p.add_argument('--db', help='store path (default: ~/.local/share/gmail_assist/store.db)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if args.command == 'sync':
        sync(args)
    else:
        labels()
