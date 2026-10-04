# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for generic backend submission and FactorDB fetching."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import requests

from factortool.backend import Backend, FetchCriteria
from factortool.config import Config
from factortool.factordb import FactorDB
from factortool.http import HttpClient
from factortool.number import Number
from factortool.stats import FactoringStats

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator
    from pathlib import Path


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Build an isolated configuration without FactorDB credentials.

    Returns:
        Config: The test configuration.
    """
    return Config.model_construct(
        batch_state_path=tmp_path / "batch_state.json",
        cado_nfs_path=tmp_path / "cado-nfs.py",
        factordb_cooldown_period=0.0,
        factordb_response_path=tmp_path / "factordb_response.html",
        factordb_session_path=tmp_path / "factordb_session.json",
        factordb_username="",
        factordb_password="",
        factoring_mode="standard",
        max_siqs_digits=100,
        max_threads=1,
        result_output_path=tmp_path / "results",
        stats_path=tmp_path / "stats.json",
        work_path=tmp_path / "work",
        yafu_path=tmp_path / "yafu",
        yafu_ini_path=None,
    )


@pytest.fixture
def http_request(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace HTTP requests with a recording mock returning a composite.

    Returns:
        Mock: The recording request mock.
    """
    response = Mock(spec=requests.Response, text="15\n")
    request = Mock(return_value=response)
    monkeypatch.setattr(HttpClient, "request", request)
    return request


@pytest.fixture
def factordb(config: Config) -> Iterator[FactorDB]:
    """Create a FactorDB backend and always stop its submission worker.

    Yields:
        FactorDB: The test backend.
    """
    backend = FactorDB(config, FactoringStats(config.stats_path, read_only=True))
    try:
        yield backend
    finally:
        backend.close()


def test_trial_factoring_submits_to_generic_backend_once(config: Config) -> None:
    """Test that a completed number is submitted once to a generic backend."""
    backend = Mock(spec=Backend)
    submissions: list[tuple[Number, list[int], list[int]]] = []

    def record_submission(numbers: Collection[Number]) -> None:
        submissions.extend((number, number.prime_factors.copy(), number.composite_factors.copy()) for number in numbers)

    backend.submit.side_effect = record_submission
    number = Number(15, config, FactoringStats(config.stats_path, read_only=True), backend)

    number.factor_tf()

    assert number.factored
    assert submissions == [(number, [3, 5], [])]
    backend.submit.assert_called_once_with([number])

    number.factor_tf()

    assert submissions == [(number, [3, 5], [])]
    backend.submit.assert_called_once_with([number])


def test_fetch_maps_criteria(factordb: FactorDB, http_request: Mock) -> None:
    """Test that fetch criteria map to FactorDB query parameters."""
    numbers = factordb.fetch(FetchCriteria(count=7, min_digits=2, skip_count=11))

    assert {number.n for number in numbers} == {15}
    http_request.assert_called_once_with(
        "GET",
        "https://factordb.com/listtype.php",
        params={"t": 3, "mindig": 2, "perpage": 7, "start": 11, "download": 1},
        timeout=3.0,
        max_attempts=None,
    )


def test_fetch_caps_count_at_fifty(factordb: FactorDB, http_request: Mock) -> None:
    """Test that requests for 100 numbers are capped at 50 per page."""
    expected_perpage = 50
    numbers = factordb.fetch(FetchCriteria(count=100, min_digits=2))

    assert {number.n for number in numbers} == {15}
    http_request.assert_called_once()
    assert http_request.call_args.kwargs["params"]["perpage"] == expected_perpage


def test_fetch_zero_skips_http_request(factordb: FactorDB, http_request: Mock) -> None:
    """Test that requesting zero numbers returns an empty set without HTTP."""
    assert factordb.fetch(FetchCriteria(count=0, min_digits=2)) == set()
    http_request.assert_not_called()


@pytest.mark.parametrize(
    ("response_text", "expected"),
    [
        ("15\n999\n1001\n", {15, 999}),
        ("1001\n", set[int]()),
        ("999\n", {999}),
    ],
    ids=["mixed", "all-excluded", "inclusive-bound"],
)
def test_fetch_filters_max_digits(
    factordb: FactorDB, http_request: Mock, response_text: str, expected: set[int]
) -> None:
    """Test digit filtering terminates after one successfully parsed response."""
    # A finite side effect makes an unexpected retry fail instead of hanging.
    http_request.side_effect = [Mock(spec=requests.Response, text=response_text)]

    numbers = factordb.fetch(FetchCriteria(count=3, min_digits=2, max_digits=3))

    assert {number.n for number in numbers} == expected
    http_request.assert_called_once()


@pytest.mark.parametrize(
    ("count", "min_digits", "max_digits", "skip_count", "message"),
    [
        (-1, 1, None, 0, "count must be nonnegative"),
        (1, 1, None, -1, "skip_count must be nonnegative"),
        (1, 0, None, 0, "min_digits must be at least 1"),
        (1, -1, None, 0, "min_digits must be at least 1"),
        (1, 3, 2, 0, "max_digits must be at least min_digits"),
    ],
)
def test_fetch_criteria_rejects_invalid_constraints(
    count: int, min_digits: int, max_digits: int | None, skip_count: int, message: str
) -> None:
    """Test invalid constraints are rejected at construction."""
    with pytest.raises(ValueError, match=message):
        FetchCriteria(count=count, min_digits=min_digits, max_digits=max_digits, skip_count=skip_count)


def test_fetch_criteria_accepts_boundary_values() -> None:
    """Test zero counts and equal positive digit bounds are valid."""
    criteria = FetchCriteria(count=0, min_digits=1, max_digits=1, skip_count=0)

    assert criteria.count == 0
    assert criteria.min_digits == criteria.max_digits == 1
    assert criteria.skip_count == 0
