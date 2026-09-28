# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Shared HTTP client with retry and backoff behavior for remote factoring services."""

from __future__ import annotations

import datetime
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

import requests

from loguru import logger

MAX_DELAY = 3600.0
TRANSIENT_STATUS_CODES = frozenset({502, 503, 504})


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
                retry_date = datetime.datetime.strptime(retry_after, "%a, %d %b %Y %H:%M:%S GMT").astimezone(
                    datetime.UTC
                )
                delay = (retry_date - datetime.datetime.now(tz=datetime.UTC)).total_seconds()
                delay = max(1.0, delay)
            except ValueError:
                logger.warning("Failed to parse Retry-After header: {}", retry_after)

    return delay


class HttpClient:
    """HTTP client that retries transient failures with exponential backoff."""

    def __init__(self, service_name: str, cooldown_period: float) -> None:
        """Initialize the HTTP client.

        Args:
            service_name (str): Name of the remote service. Only used for logging purposes.
            cooldown_period (float): Cooldown period between requests in seconds.
        """
        self._service_name = service_name
        self._cooldown_period = cooldown_period
        self.session = requests.Session()

    def request(  # ruff: ignore[too-many-arguments] (Matching the signature of requests.Session.request)
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, int | str] | None = None,
        data: Mapping[str, str] | None = None,
        files: Mapping[str, tuple[None, str]] | None = None,
        json: Mapping[str, str] | None = None,
        max_retries: int | None = 5,
        timeout: float = 3.0,
    ) -> requests.Response:
        """Perform an HTTP request, retrying transient failures with exponential backoff.

        Args:
            method (str): HTTP method (e.g., "GET", "POST").
            url (str): URL of the request.
            params (Mapping[str, int | str] | None): Query parameters for the request.
            data (Mapping[str, str] | None): Form data for the request.
            files (Mapping[str, tuple[None, str]] | None): Files to upload with the request.
            json (Mapping[str, str] | None): JSON payload for the request.
            max_retries (int | None): Maximum number of retries for transient failures.
            timeout (float): Timeout for the request in seconds.

        Returns:
            requests.Response: The HTTP response.

        Raises:
            requests.RequestException: If the request fails after the maximum number of retries.
        """
        delay = max(0.1, self._cooldown_period)
        attempts = 0

        while True:
            attempts += 1

            try:  # ruff: ignore[too-many-statements-in-try-clause]
                response = self.session.request(
                    method, url, params=params, data=data, files=files, json=json, timeout=timeout
                )

                if response.status_code == requests.codes.too_many_requests:
                    rate_limit_delay = get_too_many_requests_delay(response)
                    logger.warning(
                        "Rate limited by {} (429). Retrying in {} seconds...", self._service_name, rate_limit_delay
                    )
                    time.sleep(rate_limit_delay)
                    continue

                if response.status_code in TRANSIENT_STATUS_CODES:
                    logger.warning(
                        "Transient HTTP {} from {}. Retrying in {} seconds...",
                        response.status_code,
                        self._service_name,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(MAX_DELAY, delay * 2)
                    continue

                response.raise_for_status()
            except requests.Timeout as e:
                logger.warning(
                    "HTTP timeout contacting {}: {}. Retrying in {} seconds...", self._service_name, e, delay
                )
                time.sleep(delay)
                delay = min(MAX_DELAY, delay * 2)
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                logger.warning(
                    "Unexpected HTTP {} from {}: {}. Retrying in {} seconds...", status, self._service_name, e, delay
                )
                time.sleep(delay)
                delay = min(MAX_DELAY, delay * 2)
            except requests.RequestException as e:
                logger.warning("HTTP error contacting {}: {}. Retrying in {} seconds...", self._service_name, e, delay)
                time.sleep(delay)
                delay = min(MAX_DELAY, delay * 2)
            else:
                return response

            if max_retries is not None and attempts >= max_retries:
                msg = f"Exceeded maximum retries ({max_retries}) for HTTP request to {self._service_name}"
                raise requests.RequestException(msg)

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
