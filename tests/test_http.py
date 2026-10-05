# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the shared HTTP client."""

from __future__ import annotations

import datetime
import sys
import time

from typing import TYPE_CHECKING

import pytest
import requests

from factortool.http import HttpClient, get_too_many_requests_delay

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def make_response(status_code: int, headers: dict[str, str] | None = None) -> requests.Response:
    """Build a bare response with the given status code and headers.

    Returns:
        requests.Response: The constructed HTTP response.
    """
    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers or {})
    return response


def make_client(
    monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[requests.Response | requests.RequestException]
) -> tuple[HttpClient, list[float]]:
    """Build a client whose requests yield the given outcomes in order, recording sleeps instead of sleeping.

    Returns:
        tuple[HttpClient, list[float]]: The constructed HTTP client and the recorded sleep durations.
    """
    client = HttpClient("Test", 1.0)
    remaining = iter(outcomes)
    sleeps: list[float] = []

    def fake_request(*_args: object, **_kwargs: object) -> requests.Response:
        outcome = next(remaining)

        if isinstance(outcome, requests.RequestException):
            raise outcome

        return outcome

    monkeypatch.setattr(client.session, "request", fake_request)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    return client, sleeps


@pytest.mark.parametrize(
    "failure",
    [
        lambda: make_response(429, {"Retry-After": "10"}),
        lambda: make_response(503),
        lambda: make_response(500),
        lambda: make_response(408),
        lambda: requests.Timeout("timed out"),
        lambda: requests.ConnectionError("refused"),
    ],
    ids=["429", "503", "500", "408", "timeout", "connection"],
)
def test_every_failure_counts_toward_max_attempts(
    monkeypatch: pytest.MonkeyPatch, failure: Callable[[], requests.Response | requests.RequestException]
) -> None:
    """Tests that each kind of failure gives up after max_attempts requests, without sleeping after the last."""
    max_attempts = 3
    client, sleeps = make_client(monkeypatch, [failure() for _ in range(max_attempts + 1)])

    with pytest.raises(requests.RequestException, match=f"Giving up after {max_attempts} attempts"):
        client.request("GET", "https://example.com/", max_attempts=max_attempts)

    assert len(sleeps) == max_attempts - 1


@pytest.mark.parametrize("status_code", [400, 403, 404, 422])
@pytest.mark.parametrize("max_attempts", [5, None], ids=["limited", "unlimited"])
def test_permanent_client_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, status_code: int, max_attempts: int | None
) -> None:
    """Tests that a permanent client error fails on the first attempt without sleeping."""
    client, sleeps = make_client(monkeypatch, [make_response(status_code), make_response(200)])

    with pytest.raises(requests.RequestException, match=f"Unexpected HTTP {status_code}.*Not retrying"):
        client.request("GET", "https://example.com/", max_attempts=max_attempts)

    assert sleeps == []


def test_unlimited_attempts_retry_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that max_attempts=None keeps retrying and returns the eventual success."""
    failures = 10
    ok = make_response(200)
    client, sleeps = make_client(monkeypatch, [make_response(503)] * failures + [ok])

    assert client.request("GET", "https://example.com/", max_attempts=None) is ok
    assert len(sleeps) == failures


def test_rate_limit_does_not_advance_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that the server-dictated 429 wait is honored and leaves the backoff timer unchanged."""
    client, sleeps = make_client(
        monkeypatch,
        [make_response(503), make_response(429, {"Retry-After": "7"}), make_response(503), make_response(200)],
    )

    client.request("GET", "https://example.com/")

    assert sleeps == [1.0, 7, 2.0]


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="Changing the local time zone requires time.tzset()")
def test_retry_after_date_is_gmt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that an HTTP-date Retry-After header is correctly interpreted as GMT."""
    # Work around mypy not understanding that this test only runs if time.tzset() is available.
    if sys.platform == "win32":
        return

    retry_date = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(minutes=2)
    response = make_response(429, {"Retry-After": retry_date.strftime("%a, %d %b %Y %H:%M:%S GMT")})

    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()

    try:
        delay = get_too_many_requests_delay(response)
    finally:
        monkeypatch.undo()
        time.tzset()

    assert delay == pytest.approx(120, abs=5)
