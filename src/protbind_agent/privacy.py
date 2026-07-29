"""Small redaction and network-approval helpers shared by reports and workers."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

_SECRET_NAME = (
    r"(?:[A-Z0-9_.-]*(?:API[_-]?KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL)"
    r"[A-Z0-9_.-]*)"
)
_QUOTED_SECRET_ASSIGNMENT = re.compile(
    rf"(?i)(?P<key>[\"']?{_SECRET_NAME}[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)"
)
_UNQUOTED_SECRET_ASSIGNMENT = re.compile(
    rf"(?i)(?P<key>{_SECRET_NAME}\s*[:=]\s*)(?P<value>[^\s,;}}]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_UNIX_ABSOLUTE_PATH = re.compile(r"(?<![\w.])/(?:[^\s/:]+/)+[^\s,;:]+")
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)\b[A-Z]:[\\/](?:[^\s\\/:]+[\\/])+[^\s,;:]+"
)


def redact_text(text: str) -> str:
    value = _QUOTED_SECRET_ASSIGNMENT.sub(
        lambda match: (
            f"{match.group('key')}{match.group('quote')}[REDACTED]"
            f"{match.group('quote')}"
        ),
        text,
    )
    value = _UNQUOTED_SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group('key')}[REDACTED]", value
    )
    value = _BEARER_TOKEN.sub("Bearer [REDACTED]", value)
    value = _UNIX_ABSOLUTE_PATH.sub("[INTERNAL_PATH]", value)
    return _WINDOWS_ABSOLUTE_PATH.sub("[INTERNAL_PATH]", value)


def safe_local_source(path: Path) -> str:
    return f"local-import:{path.name}"


def require_network_approval(url: str, approved_domains: tuple[str, ...]) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("network imports require an HTTPS URL")
    hostname = parsed.hostname.lower().rstrip(".")
    approved = tuple(domain.lower().rstrip(".") for domain in approved_domains)
    if hostname not in approved:
        raise PermissionError(
            f"network target {hostname!r} was not explicitly approved; "
            "approve the exact source domain first"
        )
    return hostname
