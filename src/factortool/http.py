# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Shared HTTP client with retry and backoff behavior for remote factoring services."""

from __future__ import annotations

import datetime
import time

from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from collections.abc import Mapping

import requests

from loguru import logger

from factortool.__about__ import PROJECT_URL, __version__
from factortool.interrupt import Interrupted, InterruptState

MAX_DELAY = 3600.0
TRANSIENT_STATUS_CODES = frozenset({502, 503, 504})

CLIENT_ERROR_STATUS_CODES = range(400, 500)
RETRYABLE_CLIENT_STATUS_CODES = frozenset({408, 429})


def build_user_agent(identity: str = "", override: str = "") -> str:
    """Compose the User-Agent to identify this client to a remote service.

    Returns:
        str: The User-Agent header value.
    """
    if override:
        return override

    detail = f"+{PROJECT_URL}"

    if identity:
        safe_identity = quote(identity, safe="", encoding="utf-8")
        detail = f"{safe_identity}; {detail}"

    return f"factortool/{__version__} ({detail})"


def get_too_many_requests_delay(response: requests.Response, default_delay: float = 3600.0) -> float:
    """Get the delay time from a 429 Too Many Requests response.

    Returns:
        float: Delay time in seconds.
    """
    delay = default_delay
    retry_after = response.headers.get("Retry-After")

    if retry_after:
        try:
            delay = int(retry_after)
        except ValueError:
            try:
                retry_date = datetime.datetime.strptime(retry_after, "%a, %d %b %Y %H:%M:%S GMT").replace(
                    tzinfo=datetime.UTC
                )
                delay = (retry_date - datetime.datetime.now(tz=datetime.UTC)).total_seconds()
                delay = max(1.0, delay)
            except ValueError:
                logger.warning("Failed to parse Retry-After header: {}", retry_after)

    return delay


def is_permanent_client_error(status_code: int | None) -> bool:
    """Determine whether an HTTP status code is a client error that retrying cannot fix.

    Returns:
        bool: True for 4xx status codes other than 408 Request Timeout and 429 Too Many Requests.
    """
    return status_code in CLIENT_ERROR_STATUS_CODES and status_code not in RETRYABLE_CLIENT_STATUS_CODES


class PermanentHttpError(requests.RequestException):
    """A permanent HTTP error indicating that the request cannot be retried."""


class HttpClient:
    """HTTP client that retries transient failures with exponential backoff.

    Client errors (4xx) other than 408 and 429 are treated as permanent and fail immediately regardless of max_attempts.
    """

    def __init__(
        self,
        service_name: str,
        cooldown_period: float,
        user_agent: str | None = None,
        interrupts: InterruptState | None = None,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            service_name (str): Name of the remote service. Only used for logging purposes.
            cooldown_period (float): Cooldown period between requests in seconds.
            user_agent (str | None): Custom User-Agent header value. If None, a default User-Agent will be used.
            interrupts (InterruptState | None): Interrupt state for handling external interruptions.
        """
        self._service_name = service_name
        self._cooldown_period = cooldown_period
        self._interrupts = interrupts
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent if user_agent is not None else build_user_agent()

    def _backoff(self, delay: float, *, interruptible: bool) -> None:
        """Wait between attempts, abandoning the wait if an interrupt arrives.

        Raises:
            Interrupted: If an interrupt is received while waiting.
        """
        if interruptible and self._interrupts is not None:
            if self._interrupts.wait(delay):
                msg = f"Interrupted while waiting to retry a request to {self._service_name}"
                raise Interrupted(msg)

            return

        time.sleep(delay)

    def request(  # ruff: ignore[too-many-arguments] (Matching the signature of requests.Session.request)
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, int | str] | None = None,
        data: Mapping[str, str] | None = None,
        files: Mapping[str, tuple[None, str]] | None = None,
        json: Mapping[str, str] | None = None,
        max_attempts: int | None = 5,
        timeout: float = 3.0,
        interruptible: bool = False,
    ) -> requests.Response:
        """Perform an HTTP request, retrying transient failures with exponential backoff.

        Args:
            method (str): HTTP method (e.g., "GET", "POST").
            url (str): URL of the request.
            params (Mapping[str, int | str] | None): Query parameters for the request.
            data (Mapping[str, str] | None): Form data for the request.
            files (Mapping[str, tuple[None, str]] | None): Files to upload with the request.
            json (Mapping[str, str] | None): JSON payload for the request.
            max_attempts (int | None): Maximum number of attempts, or None for unlimited retries.
            timeout (float): Timeout for the request in seconds.
            interruptible (bool): Whether the request can be interrupted during backoff.

        Returns:
            requests.Response: The HTTP response.

        Raises:
            PermanentHttpError: Immediately on a permanent client error.
            requests.RequestException: If the request still fails after the maximum number of attempts.
        """
        delay = max(0.1, self._cooldown_period)
        attempts = 0

        while True:
            attempts += 1
            wait = delay
            rate_limited = False
            error: requests.RequestException | None = None

            try:  # ruff: ignore[too-many-statements-in-try-clause]
                response = self.session.request(
                    method, url, params=params, data=data, files=files, json=json, timeout=timeout
                )

                if response.status_code == requests.codes.too_many_requests:
                    wait = get_too_many_requests_delay(response)
                    rate_limited = True
                    reason = f"Rate limited by {self._service_name} (429)"
                elif response.status_code in TRANSIENT_STATUS_CODES:
                    reason = f"Transient HTTP {response.status_code} from {self._service_name}"
                else:
                    response.raise_for_status()
                    return response
            except requests.Timeout as e:
                error = e
                reason = f"HTTP timeout contacting {self._service_name}: {e}"
            except requests.HTTPError as e:
                error = e
                status = getattr(e.response, "status_code", None)
                reason = f"Unexpected HTTP {status} from {self._service_name}: {e}"

                if is_permanent_client_error(status):
                    msg = f"{reason}. Not retrying a client error."
                    raise PermanentHttpError(msg) from e
            except requests.RequestException as e:
                error = e
                reason = f"HTTP error contacting {self._service_name}: {e}"

            if max_attempts is not None and attempts >= max_attempts:
                msg = f"{reason}. Giving up after {attempts} attempts."
                raise requests.RequestException(msg) from error

            logger.warning("{}. Retrying in {} seconds...", reason, wait)
            self._backoff(wait, interruptible=interruptible)

            if not rate_limited:
                delay = min(MAX_DELAY, delay * 2)

    def get_cookies(self) -> dict[str, str]:
        """Return the session's cookies as a plain dictionary.

        Returns:
            dict[str, str]: The current session cookies.
        """
        return self.session.cookies.get_dict()

    def set_cookies(self, cookies: Mapping[str, str]) -> None:
        """Replace the session's cookies.

        Args:
            cookies (Mapping[str, str]): The new cookies to set for the session.
        """
        self.session.cookies.update(cookies)  # pyright: ignore[reportUnknownMemberType] (return type is Unknown, but irrelevant here)
