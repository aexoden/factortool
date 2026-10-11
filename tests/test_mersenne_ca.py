# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the mersenne.ca Aliquot backend."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

import pytest
import requests

from factortool.backend import FetchCriteria, SubmitOutcome
from factortool.http import HttpClient, PermanentHttpError
from factortool.mersenne_ca import MersenneCA, check_factorization
from factortool.number import format_factorization
from factortool.stats import FactoringStats
from factortool.submissions import Submission

from .helpers import make_config, make_number


@pytest.fixture
def mersenne_ca(tmp_path: Path) -> Iterator[MersenneCA]:
    """Create a mersenne.ca backend and always stop its submission worker.

    Yields:
        MersenneCA: The test backend.
    """
    config = make_config(
        gimps_login="tester",
        mersenne_ca_cooldown_period=0.0,
        state_path=tmp_path,
    )
    backend = MersenneCA(config, FactoringStats(tmp_path / "stats.json", read_only=True))

    try:
        yield backend
    finally:
        backend.close()


def test_check_factorization_accepts_a_consistent_factorization() -> None:
    """Test that factors that multiply back to the original number are accepted."""
    assert check_factorization(100, [2, 2, 5, 5])


def test_check_factorization_accepts_a_trailing_composite() -> None:
    """Test that a factorization ending with a composite is accepted."""
    assert check_factorization(100, [2, 2, 25])


def test_check_factorization_rejects_an_inconsistent_factorization() -> None:
    """Test that an inconsistent factorization is rejected, as has occasionally happened with YAFU."""
    assert not check_factorization(100, [2, 2, 5])
    assert not check_factorization(100, [3, 5, 7])
    assert not check_factorization(100, [])


def test_format_factorization_lists_every_factor() -> None:
    """Test that every factor is listed individually in the formatted factorization."""
    number = make_number(100)
    number.prime_factors = [2, 2, 5, 5]
    number.composite_factors = []

    assert format_factorization(number, "*") == "100=2*2*5*5"


def test_format_factorization_includes_trailing_composites() -> None:
    """Test that a trailing composite factor is included in the formatted factorization."""
    number = make_number(100)
    number.prime_factors = [2, 2]
    number.composite_factors = [25]

    assert format_factorization(number, "*") == "100=2*2*25"


@pytest.mark.parametrize(
    ("json_result", "json_error", "outcome"),
    [
        ({"status": "accepted"}, None, SubmitOutcome.ACCEPTED),
        ({"error": "submission rejected"}, None, SubmitOutcome.REJECTED),
        (["unexpected"], None, SubmitOutcome.FAILED),
        (None, ValueError("invalid JSON"), SubmitOutcome.FAILED),
    ],
    ids=["accepted", "rejected", "not-an-object", "malformed"],
)
def test_submit_reports_the_outcome_of_a_response(
    mersenne_ca: MersenneCA,
    monkeypatch: pytest.MonkeyPatch,
    json_result: object,
    json_error: ValueError | None,
    outcome: SubmitOutcome,
) -> None:
    """Test that only an error reported by the service is final, and a malformed response is worth another attempt."""
    submission = Submission(n=100, prime_factors=(2, 2, 5, 5), composite_factors=(), expires_at=0.0)
    response = Mock(spec=requests.Response)
    if json_error:
        response.json.side_effect = json_error
    else:
        response.json.return_value = json_result
    request = Mock(return_value=response)
    monkeypatch.setattr(HttpClient, "request", request)

    assert mersenne_ca._submit_number(submission) is outcome
    assert request.call_args.kwargs["files"]["compositefactorization"] == (None, "100=2*2*5*5")


def test_submit_leaves_a_request_error_for_another_attempt(
    mersenne_ca: MersenneCA, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a submission the service could not be asked about is worth another attempt."""
    submission = Submission(n=100, prime_factors=(2, 2, 5, 5), composite_factors=(), expires_at=0.0)
    monkeypatch.setattr(HttpClient, "request", Mock(side_effect=requests.RequestException("unreachable")))

    assert mersenne_ca._submit_number(submission) is SubmitOutcome.FAILED


def test_submit_does_not_keep_a_permanent_error(mersenne_ca: MersenneCA, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a submission the service will never accept is not left for another attempt."""
    submission = Submission(n=100, prime_factors=(2, 2, 5, 5), composite_factors=(), expires_at=0.0)
    monkeypatch.setattr(HttpClient, "request", Mock(side_effect=PermanentHttpError("HTTP 400")))

    assert mersenne_ca._submit_number(submission) is SubmitOutcome.REJECTED


def test_submit_refuses_an_inconsistent_factorization(mersenne_ca: MersenneCA, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a factorization that does not multiply back to its number is never sent."""
    submission = Submission(n=100, prime_factors=(2, 2, 5), composite_factors=(), expires_at=0.0)
    request = Mock()
    monkeypatch.setattr(HttpClient, "request", request)

    assert mersenne_ca._submit_number(submission) is SubmitOutcome.REJECTED
    request.assert_not_called()


def test_fetch_requires_max_digits(mersenne_ca: MersenneCA) -> None:
    """Test that fetching without specifying a maximum digit count raises a ValueError."""
    with pytest.raises(ValueError, match="maximum digit count"):
        mersenne_ca.fetch(FetchCriteria(count=3, min_digits=50))


def test_fetch_rejects_skip_count(mersenne_ca: MersenneCA) -> None:
    """Test that specifying a skip count raises a ValueError."""
    with pytest.raises(ValueError, match="skip count"):
        mersenne_ca.fetch(FetchCriteria(count=3, min_digits=50, max_digits=60, skip_count=5))


def test_fetch_maps_criteria(mersenne_ca: MersenneCA, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch criteria map to mersenne.ca query parameters, retrying transient failures indefinitely."""
    request = Mock(return_value=Mock(spec=requests.Response, text="999\n1001\n"))
    monkeypatch.setattr(HttpClient, "request", request)

    numbers = mersenne_ca.fetch(FetchCriteria(count=3, min_digits=2, max_digits=4))

    assert {number.n for number in numbers} == {999, 1001}
    request.assert_called_once_with(
        "GET",
        "https://www.mersenne.ca/aliquot/index.php",
        params={"composites_to_factor": 3, "min_digits": 2, "max_digits": 4, "gimps_login": "tester"},
        data=None,
        files=None,
        json=None,
        timeout=30.0,
        max_attempts=None,
        wait=mersenne_ca._interrupts.wait,
    )
