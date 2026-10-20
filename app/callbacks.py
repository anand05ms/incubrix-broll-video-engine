import hashlib
import hmac
import ipaddress
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, file_pointer: Any, code: int, message: str, headers: Any, new_url: str) -> None:
        return None


def validate_callback_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError("callback_url must be a public HTTPS URL on port 443")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, 443)
    except socket.gaierror as exc:
        raise ValueError("callback_url host could not be resolved") from exc
    if not addresses or any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
        raise ValueError("callback_url must resolve only to public IP addresses")


def send_signed_callback(
    url: str,
    secret: str,
    payload: dict[str, Any],
    correlation_id: str,
) -> None:
    validate_callback_url(url)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    signature = hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("ascii") + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-IncuBrix-Timestamp": timestamp,
            "X-IncuBrix-Signature": f"sha256={signature}",
            "X-Correlation-Id": correlation_id,
        },
        method="POST",
    )
    opener = urllib.request.build_opener(_NoRedirect())
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with opener.open(request, timeout=10) as response:
                if response.status < 200 or response.status >= 300:
                    raise urllib.error.HTTPError(
                        url,
                        response.status,
                        "Callback endpoint returned a non-success status",
                        response.headers,
                        None,
                    )
                return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"Signed callback delivery failed after three attempts: {last_error}") from last_error
