# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the shared HTTP client."""

from __future__ import annotations

import datetime
import sys
import time

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import requests

from factortool.__about__ import PROJECT_URL, __version__
from factortool.http import (
    DEFAULT_RATE_LIMIT_DELAY,
    MIN_RATE_LIMIT_DELAY,
    HttpClient,
    PermanentHttpError,
    build_user_agent,
    get_too_many_requests_delay,
)
from factortool.interrupt import Interrupted

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@pytest.mark.parametrize(
    ("identity", "encoded"),
    [
        ("FactorFinder", "FactorFinder"),
        ("用户", "%E7%94%A8%E6%88%B7"),
        ("Renée", "Ren%C3%A9e"),
        ("name(test)", "name%28test%29"),
        ("name\r\nvalue", "name%0D%0Avalue"),
    ],
)
def test_user_agent_encodes_account_names(identity: str, encoded: str) -> None:
    """Test that the User-Agent encodes account names correctly."""
    assert build_user_agent(identity) == (f"factortool/{__version__} ({encoded}; +{PROJECT_URL})")


def test_user_agent_identifies_the_tool_and_project() -> None:
    """Test that the User-Agent names the tool, its version and the project URL."""
    assert build_user_agent() == f"factortool/{__version__} (+{PROJECT_URL})"


def test_user_agent_includes_the_configured_account() -> None:
    """Test that the User-Agent includes the configured account name."""
    assert build_user_agent("FactorFinder") == f"factortool/{__version__} (FactorFinder; +{PROJECT_URL})"


def test_user_agent_override_wins() -> None:
    """Test that a configured override replaces the composed User-Agent."""
    assert build_user_agent("FactorFinder", "custom/1.0") == "custom/1.0"


def test_client_sends_the_user_agent() -> None:
    """Test that the client sends the given User-Agent."""
    client = HttpClient("test", 1.0, "factortool/test")

    assert client.session.headers["User-Agent"] == "factortool/test"


def test_client_defaults_to_the_composed_user_agent() -> None:
    """Test that the client defaults to the composed User-Agent."""
    assert HttpClient("test", 1.0).session.headers["User-Agent"] == build_user_agent()


def make_response(status_code: int, headers: dict[str, str] | None = None) -> requests.Response:
    """Build a bare response with the given status code and headers.

    Returns:
        requests.Response: The constructed HTTP response.
    """
    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers or {})
    return response


# The time at which the clock used by make_client starts.
START_TIME = 1_000_000.0


def make_client(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: Sequence[requests.Response | requests.RequestException],
    **options: float | Callable[[float], None],
) -> tuple[HttpClient, list[float]]:
    """Build a client whose requests yield the given outcomes in order, recording sleeps instead of sleeping.

    The clock starts at START_TIME and only advances by the recorded sleeps.

    Returns:
        tuple[HttpClient, list[float]]: The constructed HTTP client and the recorded sleep durations.
    """
    client = HttpClient("Test", 1.0, **options)  # type: ignore[arg-type]
    remaining = iter(outcomes)
    sleeps: list[float] = []

    def fake_request(*_args: object, **_kwargs: object) -> requests.Response:
        outcome = next(remaining)

        if isinstance(outcome, requests.RequestException):
            raise outcome

        return outcome

    monkeypatch.setattr(client.session, "request", fake_request)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(time, "time", lambda: START_TIME + sum(sleeps))

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

    with pytest.raises(PermanentHttpError, match=f"Unexpected HTTP {status_code}.*Not retrying"):
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

    assert sleeps == [1.0, 7.0, 2.0]


@pytest.mark.parametrize(
    ("retry_after", "delay"),
    [
        (None, DEFAULT_RATE_LIMIT_DELAY),
        ("", DEFAULT_RATE_LIMIT_DELAY),
        ("soon", DEFAULT_RATE_LIMIT_DELAY),
        ("90", 90.0),
        ("0", MIN_RATE_LIMIT_DELAY),
        ("-5", MIN_RATE_LIMIT_DELAY),
        ("Thu, 01 Jan 1970 00:00:00 GMT", MIN_RATE_LIMIT_DELAY),
    ],
    ids=["missing", "empty", "unparseable", "seconds", "zero", "negative", "past-date"],
)
def test_rate_limit_delay_is_bounded(retry_after: str | None, delay: float) -> None:
    """Tests that a missing Retry-After waits out the default."""
    response = make_response(429, {} if retry_after is None else {"Retry-After": retry_after})

    assert get_too_many_requests_delay(response) == delay


def test_rate_limit_is_reported_and_holds_back_later_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that a rate limit is reported with the time it ends, and that a later request waits for it too."""
    on_rate_limit = Mock()
    client, sleeps = make_client(
        monkeypatch,
        [make_response(429, {"Retry-After": "60"}), requests.ConnectionError("refused"), make_response(200)],
        on_rate_limit=on_rate_limit,
    )

    with pytest.raises(requests.RequestException, match="Giving up after 1 attempts"):
        client.request("GET", "https://example.com/", max_attempts=1)

    on_rate_limit.assert_called_once_with(START_TIME + 60.0)
    assert client.rate_limited_until == pytest.approx(START_TIME + 60.0)
    assert sleeps == []

    with pytest.raises(requests.RequestException, match="Giving up after 1 attempts"):
        client.request("GET", "https://example.com/", max_attempts=1)

    assert sleeps == [60.0]


def test_a_shorter_rate_limit_does_not_replace_a_longer_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that a limit reported to one request is not shortened by a briefer one reported to another."""
    on_rate_limit = Mock()
    client, sleeps = make_client(
        monkeypatch, [make_response(429), make_response(429, {"Retry-After": "1"})], on_rate_limit=on_rate_limit
    )

    # The second request stands in for one that was already in flight when the first was refused.
    assert client._note_rate_limit(make_response(429)) == pytest.approx(DEFAULT_RATE_LIMIT_DELAY)
    assert client._note_rate_limit(make_response(429, {"Retry-After": "1"})) == pytest.approx(DEFAULT_RATE_LIMIT_DELAY)

    assert client.rate_limited_until == pytest.approx(START_TIME + DEFAULT_RATE_LIMIT_DELAY)
    on_rate_limit.assert_called_once_with(START_TIME + DEFAULT_RATE_LIMIT_DELAY)
    assert sleeps == []


def test_a_rate_limit_from_a_previous_run_holds_back_the_first_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that a client told of a rate limit already in effect waits for it before sending anything."""
    client, sleeps = make_client(monkeypatch, [make_response(200)], rate_limited_until=START_TIME + 45.0)

    client.request("GET", "https://example.com/")

    assert sleeps == [45.0]


def test_an_abandoned_wait_abandons_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that a wait reporting that it was cut short stops the request, whichever wait it was."""
    wait = Mock(return_value=True)
    client, sleeps = make_client(monkeypatch, [make_response(503)] * 2, rate_limited_until=START_TIME + 45.0)

    with pytest.raises(Interrupted):
        client.request("GET", "https://example.com/", max_attempts=None, wait=wait)

    wait.assert_called_once_with(45.0)

    client, _ = make_client(monkeypatch, [make_response(503)] * 2)

    with pytest.raises(Interrupted):
        client.request("GET", "https://example.com/", max_attempts=None, wait=wait)

    wait.assert_called_with(1.0)
    assert sleeps == []


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
