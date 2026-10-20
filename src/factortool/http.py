# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Shared HTTP client with retry and backoff behavior for remote factoring services."""

from __future__ import annotations

import datetime as dt
import threading
import time

from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

import requests

from loguru import logger

from factortool.__about__ import PROJECT_URL, __version__
from factortool.interrupt import Interrupted

# The first backoff between attempts for failures with no specific Retry-After header.
RETRY_DELAY = 1.0

# The longest backoff between attempts for failures with no specific Retry-After header.
MAX_DELAY = 3600.0

# How long to wait after a 429 that does not say when to retry.
DEFAULT_RATE_LIMIT_DELAY = 1800.0

# The shortest wait accepted from a Retry-After header. There is no maximum, as the service's request is honored.
MIN_RATE_LIMIT_DELAY = 1.0

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


def get_too_many_requests_delay(response: requests.Response, default_delay: float = DEFAULT_RATE_LIMIT_DELAY) -> float:
    """Get the delay time from a 429 Too Many Requests response.

    Returns:
        float: Delay time in seconds.
    """
    retry_after = response.headers.get("Retry-After")

    if not retry_after:
        return default_delay

    try:
        delay = float(int(retry_after))
    except ValueError:
        try:
            retry_date = dt.datetime.strptime(retry_after, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=dt.UTC)
        except ValueError:
            logger.warning("Failed to parse Retry-After header: {}", retry_after)
            return default_delay

        delay = (retry_date - dt.datetime.now(tz=dt.UTC)).total_seconds()

    return max(MIN_RATE_LIMIT_DELAY, delay)


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

    Requests, including each retry, are spaced at least request_interval apart across all threads. Client errors (4xx)
    other than 408 and 429 are treated as permanent and fail immediately regardless of max_attempts.
    """

    def __init__(
        self,
        service_name: str,
        request_interval: float,
        user_agent: str | None = None,
        rate_limited_until: float = 0.0,
        on_rate_limit: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            service_name (str): Name of the remote service. Only used for logging purposes.
            request_interval (float): The minimum time between the starts of consecutive requests, in seconds.
            user_agent (str | None): Custom User-Agent header value. If None, a default User-Agent will be used.
            rate_limited_until (float): When a rate limit already in effect ends, as a timestamp.
            on_rate_limit (Callable[[float], None] | None): Called with the timestamp at which each new rate limit ends.
        """
        self._service_name = service_name
        self._request_interval = request_interval
        self._next_request_at = 0.0
        self._request_lock = threading.Lock()
        self._rate_limited_until = rate_limited_until
        self._rate_limit_lock = threading.Lock()
        self._on_rate_limit = on_rate_limit
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent if user_agent is not None else build_user_agent()

    @property
    def rate_limited_until(self) -> float:
        """When the rate limit most recently reported by the service ends, as a timestamp."""
        return self._rate_limited_until

    @property
    def next_request_at(self) -> float:
        """When request_interval allows the next request, as a time.monotonic() value."""
        return self._next_request_at

    def _pause(self, delay: float, wait: Callable[[float], bool] | None) -> None:
        """Wait between attempts, abandoning the request if the wait function indicates so.

        Raises:
            Interrupted: If the wait was abandoned.
        """
        if wait is None:
            time.sleep(max(0.0, delay))
        elif wait(delay):
            msg = f"Abandoned a request to {self._service_name} while waiting to send it"
            raise Interrupted(msg)

    def _await_turn(self, wait: Callable[[float], bool] | None, *, reserve: bool, announce: bool) -> None:
        """Wait until any rate limit has ended and request_interval has passed since the last request.

        If reserve is set, the turn is claimed for a request about to be sent.
        """
        while True:
            with self._request_lock:
                now = time.monotonic()
                rate_limit = self._rate_limited_until - time.time()
                remaining = max(rate_limit, self._next_request_at - now)

                if remaining <= 0:
                    if reserve:
                        self._next_request_at = now + self._request_interval

                    return

            if announce and rate_limit > 0:
                logger.info("Waiting {:.0f} seconds for the {} rate limit to end", rate_limit, self._service_name)
                announce = False

            self._pause(remaining, wait)

    def await_ready(self, wait: Callable[[float], bool] | None = None) -> None:
        """Wait until a request could be sent without delay, for callers that prepare a request as late as possible.

        Raises:
            Interrupted: If the wait was abandoned.
        """
        self._await_turn(wait, reserve=False, announce=True)

    def _note_rate_limit(self, response: requests.Response) -> float:
        """Record the rate limit reported by a 429 response.

        Returns:
            float: How long the rate limit in effect lasts, in seconds.
        """
        with self._rate_limit_lock:
            now = time.time()
            rate_limited_until = now + get_too_many_requests_delay(response)

            if rate_limited_until > self._rate_limited_until:
                self._rate_limited_until = rate_limited_until

                if self._on_rate_limit is not None:
                    self._on_rate_limit(rate_limited_until)

            return self._rate_limited_until - now

    def request(  # ruff: ignore[too-many-arguments] (Matching the signature of requests.Session.request)
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, int | str] | None = None,
        data: Mapping[str, str] | None = None,
        files: Mapping[str, tuple[None, str]] | None = None,
        json: object = None,
        max_attempts: int | None = 5,
        timeout: float = 3.0,
        wait: Callable[[float], bool] | None = None,
    ) -> requests.Response:
        """Perform an HTTP request, retrying transient failures with exponential backoff.

        Args:
            method (str): HTTP method (e.g., "GET", "POST").
            url (str): URL of the request.
            params (Mapping[str, int | str] | None): Query parameters for the request.
            data (Mapping[str, str] | None): Form data for the request.
            files (Mapping[str, tuple[None, str]] | None): Files to upload with the request.
            json (object): JSON payload for the request, or None for no JSON body.
            max_attempts (int | None): Maximum number of attempts, or None for unlimited retries.
            timeout (float): Timeout for the request in seconds.
            wait (Callable[[float], bool] | None): Waits for the given number of seconds before an attempt, returning
                True to abandon the request instead.

        Returns:
            requests.Response: The HTTP response.

        Raises:
            Interrupted: If the wait abandoned the request.
            PermanentHttpError: Immediately on a permanent client error.
            requests.RequestException: If the request still fails after the maximum number of attempts.
        """
        delay = RETRY_DELAY
        attempts = 0

        while True:
            self._await_turn(wait, reserve=True, announce=attempts == 0)

            attempts += 1
            rate_limit_delay: float | None = None
            error: requests.RequestException | None = None

            start_time = time.monotonic()

            try:  # ruff: ignore[too-many-statements-in-try-clause]
                response = self.session.request(
                    method, url, params=params, data=data, files=files, json=json, timeout=timeout
                )
                logger.debug(
                    "{} {} returned HTTP {} after {:.2f}s",
                    method,
                    url,
                    response.status_code,
                    time.monotonic() - start_time,
                )

                if response.status_code == requests.codes.too_many_requests:
                    rate_limit_delay = self._note_rate_limit(response)
                    reason = f"Rate limited by {self._service_name} (429)"
                elif response.status_code in TRANSIENT_STATUS_CODES:
                    reason = f"Transient HTTP {response.status_code} from {self._service_name}"
                else:
                    response.raise_for_status()
                    return response
            except requests.Timeout as e:
                error = e
                reason = f"HTTP timeout contacting {self._service_name}: {e}"
                logger.debug("{} {} timed out after {:.2f}s", method, url, time.monotonic() - start_time)
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
                logger.debug(
                    "{} {} failed after {:.2f}s: {}", method, url, time.monotonic() - start_time, type(e).__name__
                )

            if max_attempts is not None and attempts >= max_attempts:
                msg = f"{reason}. Giving up after {attempts} attempts."
                raise requests.RequestException(msg) from error

            if rate_limit_delay is not None:
                logger.warning("{}. Retrying in {:.0f} seconds...", reason, rate_limit_delay)
                continue

            logger.warning("{}. Retrying in {} seconds...", reason, delay)
            self._pause(delay, wait)
            delay = min(MAX_DELAY, delay * 2)
