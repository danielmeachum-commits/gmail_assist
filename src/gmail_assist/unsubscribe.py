"""One-click unsubscribe (RFC 8058).

A list that sends both List-Unsubscribe (with an https URL) and
List-Unsubscribe-Post: List-Unsubscribe=One-Click promises that a single
POST to that URL unsubscribes you, with no page to fill in. The URL comes
from the email, so before posting to it:

- the list's latest email must have passed DKIM, by Gmail's own verdict
  (Authentication-Results): RFC 8058 requires the headers to be signed,
  so a spoofed email can't aim the request somewhere;
- it must be https on the standard port, and every address the host
  resolves to must be public: never this machine, your network or the
  tailnet (otherwise an email could point it at the dashboard or the router);
- redirects aren't followed, and it gives up after TIMEOUT_S.

(The host is resolved once for the check and again by the HTTP client; a
DNS answer that changes in between isn't caught. The DKIM requirement means
only a domain that signs mail to you could try that.)
"""
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from googleapiclient.discovery import build

from gmail_assist.gmail.auth import get_credentials

TIMEOUT_S = 15
USER_AGENT = 'gmail-assist (RFC 8058 one-click unsubscribe)'


class Refused(Exception):
    """Not sent, or the list didn't accept it; the message says why."""


def dkim_passed(message_id: str) -> bool:
    """Whether Gmail recorded dkim=pass for this message. A client per call:
    this runs on several threads, and Gmail clients aren't thread-safe."""
    gmail = build('gmail', 'v1', credentials=get_credentials('read', interactive=False),
                  cache_discovery=False)
    msg = gmail.users().messages().get(userId='me', id=message_id, format='metadata',
                                       metadataHeaders=['Authentication-Results']).execute()
    results = [h['value'] for h in msg.get('payload', {}).get('headers', [])
               if h['name'].lower() == 'authentication-results']
    return any('dkim=pass' in r.lower() for r in results)


def check_public_https(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise Refused('not an https link')
    if parsed.port not in (None, 443):
        raise Refused('unusual port')
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, 443, proto=socket.IPPROTO_TCP)}
    except socket.gaierror:
        raise Refused(f"can't resolve {parsed.hostname}") from None
    private = [a for a in addresses if not ipaddress.ip_address(a.split('%')[0]).is_global]
    if private or not addresses:
        raise Refused(f'{parsed.hostname} points at a private address')


def one_click(url: str, latest_message_id: str) -> str:
    """Send the unsubscribe request. Returns what the list answered
    (e.g. 'HTTP 200'); raises Refused if it wasn't sent or was rejected."""
    if not dkim_passed(latest_message_id):
        raise Refused("the list's latest email didn't pass DKIM")
    check_public_https(url)
    try:
        response = httpx.post(url, data={'List-Unsubscribe': 'One-Click'}, timeout=TIMEOUT_S,
                              follow_redirects=False, headers={'User-Agent': USER_AGENT})
    except httpx.HTTPError as err:
        raise Refused(f'request failed: {type(err).__name__}') from None
    if not 200 <= response.status_code < 300:
        raise Refused(f'the list answered HTTP {response.status_code}')
    return f'HTTP {response.status_code}'
