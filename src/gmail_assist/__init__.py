from gmail_assist.gmail.auth import get_service

def main() -> None:
    print("Hello from gmail-assist!")
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
