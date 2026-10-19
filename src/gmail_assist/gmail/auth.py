"""Gmail OAuth credentials, stored as 0600 files in ~/.config/gmail_assist/.

Two grants are kept separately so most work runs with a read-only token:
  - 'read':   gmail.readonly
  - 'modify': gmail.modify (apply labels, archive, trash -- and send mail)
Only ask for 'modify' at the point a write is actually needed.
"""
import json
import os
from pathlib import Path
from typing import Literal

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

Access = Literal['read', 'modify']

SCOPES: dict[Access, list[str]] = {
    'read': ['https://www.googleapis.com/auth/gmail.readonly'],
    'modify': ['https://www.googleapis.com/auth/gmail.modify'],
}

OAUTH_PORT = 38991


def _config_dir() -> Path:
    base = os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config'
    return Path(base) / 'gmail_assist'


def _client_secrets_path() -> Path:
    # The OAuth client file (Desktop app type) downloaded from Google Cloud Console.
    path = Path(os.environ.get('GMAIL_ASSIST_CREDENTIALS') or _config_dir() / 'credentials.json')
    if not path.exists():
        raise FileNotFoundError(
            f'OAuth client file not found at {path}. Download a Desktop-app OAuth client '
            'from Google Cloud Console and save it there, or set GMAIL_ASSIST_CREDENTIALS.'
        )
    return path


def _token_path(access: Access) -> Path:
    return _config_dir() / f'token_{access}.json'


def _load_token(access: Access) -> Credentials | None:
    path = _token_path(access)
    if not path.exists():
        return None
    # Don't pass scopes here: that would overwrite the stored scopes and make
    # the has_scopes() check below meaningless.
    return Credentials.from_authorized_user_info(json.loads(path.read_text()))


def _save_token(access: Access, creds: Credentials) -> None:
    # Holds a long-lived refresh token, so keep it readable by this user only.
    _config_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _token_path(access)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(creds.to_json())
    os.chmod(path, 0o600)


def get_credentials(access: Access = 'read') -> Credentials:
    scopes = SCOPES[access]
    creds = _load_token(access)

    # Scopes changed since the token was issued -- re-authorize.
    if creds and not creds.has_scopes(scopes):
        creds = None

    if creds and not creds.valid:
        if creds.refresh_token:
            try:
                creds.refresh(Request())
                _save_token(access, creds)
            except RefreshError:
                # Refresh token expired or revoked (e.g. 7-day expiry in Testing mode).
                creds = None
        else:
            creds = None

    if creds is None:
        flow = InstalledAppFlow.from_client_secrets_file(str(_client_secrets_path()), scopes)
        creds = flow.run_local_server(port=OAUTH_PORT)
        _save_token(access, creds)

    return creds


def get_service(access: Access = 'read'):
    return build('gmail', 'v1', credentials=get_credentials(access), cache_discovery=False)


def clear_credentials(access: Access | None = None) -> None:
    """Remove stored tokens (both if access is None)."""
    for name in [access] if access else SCOPES:
        _token_path(name).unlink(missing_ok=True)
