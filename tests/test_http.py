# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the shared HTTP client."""

from __future__ import annotations

import datetime
import sys
import time

import pytest
import requests

from factortool.http import get_too_many_requests_delay


def make_response(status_code: int, headers: dict[str, str] | None = None) -> requests.Response:
    """Build a bare response with the given status code and headers.

    Returns:
        requests.Response: The constructed HTTP response.
    """
    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers or {})
    return response


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
