"""Helpers for pulling fields out of Gmail message resources (users.messages.get)."""
import base64
import re
from dataclasses import dataclass
from email.utils import parseaddr
from html.parser import HTMLParser


def header(message: dict, name: str) -> str | None:
    for h in message.get('payload', {}).get('headers', []):
        if h['name'].lower() == name.lower():
            return h['value']
    return None


def sender_email(message: dict) -> str | None:
    addr = parseaddr(header(message, 'From') or '')[1]
    return addr.lower() or None


def list_id(message: dict) -> str | None:
    # "Some Newsletter <news.example.com>" -> "news.example.com"
    value = header(message, 'List-Id')
    if not value:
        return None
    m = re.search(r'<([^>]+)>', value)
    return (m.group(1) if m else value).strip().lower()


@dataclass
class Unsubscribe:
    urls: list[str]          # https:// targets from List-Unsubscribe
    mailtos: list[str]       # mailto: targets from List-Unsubscribe
    one_click: bool          # RFC 8058: POST to an https url, no page visit


def unsubscribe(message: dict) -> Unsubscribe | None:
    value = header(message, 'List-Unsubscribe')
    if not value:
        return None
    targets = re.findall(r'<([^>]+)>', value)
    urls = [t for t in targets if t.lower().startswith('https://')]
    mailtos = [t for t in targets if t.lower().startswith('mailto:')]
    post = (header(message, 'List-Unsubscribe-Post') or '').replace(' ', '').lower()
    return Unsubscribe(urls, mailtos, one_click=bool(urls) and post == 'list-unsubscribe=one-click')


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _decode(part: dict) -> str:
    data = part.get('body', {}).get('data')
    if not data:
        return ''
    return base64.urlsafe_b64decode(data + '=' * (-len(data) % 4)).decode('utf-8', 'replace')


def _walk(part: dict):
    yield part
    for sub in part.get('parts', []):
        yield from _walk(sub)


def body_text(message: dict) -> str | None:
    """Plain-text body, falling back to tag-stripped HTML. None if the
    message was fetched without a body (format='metadata')."""
    parts = list(_walk(message.get('payload', {})))
    plain = [_decode(p) for p in parts if p.get('mimeType') == 'text/plain']
    if any(plain):
        return '\n'.join(plain).strip()
    html = [_decode(p) for p in parts if p.get('mimeType') == 'text/html']
    if any(html):
        extractor = _TextExtractor()
        extractor.feed('\n'.join(html))
        return re.sub(r'\n\s*\n+', '\n\n', ''.join(extractor.parts)).strip()
    return None
