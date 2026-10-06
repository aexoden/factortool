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

from factortool.backend import FetchCriteria
from factortool.http import HttpClient
from factortool.mersenne_ca import MersenneCA, check_factorization
from factortool.number import format_factorization
from factortool.stats import FactoringStats

from .helpers import make_config, make_number


@pytest.fixture
def mersenne_ca(tmp_path: Path) -> Iterator[MersenneCA]:
    """Create a mersenne.ca backend and always stop its submission worker.

    Yields:
        MersenneCA: The test backend.
    """
    config = make_config(gimps_login="tester", mersenne_ca_cooldown_period=0.0)
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
    ("json_result", "json_error", "expected_count"),
    [
        ({"status": "accepted"}, None, 1),
        ({"error": "submission rejected"}, None, 0),
        (None, ValueError("invalid JSON"), 0),
    ],
    ids=["accepted", "rejected", "malformed"],
)
def test_submit_counts_only_accepted_responses(
    mersenne_ca: MersenneCA,
    monkeypatch: pytest.MonkeyPatch,
    json_result: dict[str, str] | None,
    json_error: ValueError | None,
    expected_count: int,
) -> None:
    """Only valid, accepted service responses count as successful submissions."""
    number = make_number(100)
    number.prime_factors = [2, 2, 5, 5]
    number.composite_factors = []
    response = Mock(spec=requests.Response)
    if json_error:
        response.json.side_effect = json_error
    else:
        response.json.return_value = json_result
    monkeypatch.setattr(HttpClient, "request", Mock(return_value=response))

    assert mersenne_ca._submit_number(number) == expected_count


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
        timeout=30.0,
        max_attempts=None,
        interruptible=True,
    )
