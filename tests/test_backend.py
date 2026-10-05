# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for generic backend behavior and FactorDB."""

from __future__ import annotations

from typing import TYPE_CHECKING, override
from unittest.mock import Mock, call

import pytest
import requests

from factortool.backend import SUBMIT_SPACING, Backend, BaseBackend, FetchCriteria
from factortool.config import Config
from factortool.factordb import FactorDB
from factortool.http import HttpClient
from factortool.number import Number
from factortool.stats import FactoringStats

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator
    from pathlib import Path


class FakeBackend(BaseBackend):
    """Minimal backend serving canned fetch responses and recording submissions."""

    name = "Fake"
    assigns_work = False
    submission_unit = "items"

    def __init__(
        self, config: Config, stats: FactoringStats, responses: Iterable[str | Exception], successes_per_number: int = 1
    ) -> None:
        """Initialize the fake backend with the fetch responses to return in order."""
        self.responses = iter(responses)
        self.successes_per_number = successes_per_number
        self.submitted: list[int] = []
        super().__init__(config, stats, 1.0)

    @override
    def _request_composites(self, criteria: FetchCriteria) -> str:
        response = next(self.responses)

        if isinstance(response, Exception):
            raise response

        return response

    @override
    def _submit_number(self, number: Number) -> int:
        self.submitted.append(number.n)
        return self.successes_per_number


@pytest.fixture
def sleep(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Record sleeps instead of sleeping.

    Returns:
        Mock: The recording sleep mock.
    """
    sleep = Mock()
    monkeypatch.setattr("factortool.backend.time.sleep", sleep)
    return sleep


def make_fake_backend(
    config: Config, responses: Iterable[str | Exception] = (), successes_per_number: int = 1
) -> FakeBackend:
    """Create a fake backend. Callers are responsible for closing it.

    Returns:
        FakeBackend: The fake backend.
    """
    return FakeBackend(config, FactoringStats(config.stats_path, read_only=True), responses, successes_per_number)


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


def test_base_fetch_retries_failures_with_backoff(config: Config, sleep: Mock) -> None:
    """Test that malformed responses and request errors are retried with exponential backoff."""
    backend = make_fake_backend(config, ["invalid", requests.RequestException("rejected"), "15\n"])

    try:
        numbers = backend.fetch(FetchCriteria(count=3, min_digits=2))
    finally:
        backend.close()

    assert {number.n for number in numbers} == {15}
    assert sleep.call_args_list[:2] == [call(1.0), call(2.0)]


def test_base_fetch_returns_empty_set_for_blank_response(config: Config) -> None:
    """Test that a blank response is a successful fetch of nothing rather than a parse error."""
    backend = make_fake_backend(config, ["\n  \n"])

    try:
        assert backend.fetch(FetchCriteria(count=3, min_digits=2)) == set()
    finally:
        backend.close()


def test_base_close_flushes_submissions(config: Config, sleep: Mock) -> None:
    """Test that closing submits every queued number with factors and totals the reported successes."""
    successes_per_number = 2
    backend = make_fake_backend(config, successes_per_number=successes_per_number)
    stats = FactoringStats(config.stats_path, read_only=True)
    factored = [Number(n, config, stats, backend) for n in (15, 21)]
    unfactored = Number(35, config, stats, backend)

    for number, factors in zip(factored, ([3, 5], [3, 7]), strict=True):
        number.prime_factors = factors

    backend.submit([*factored, unfactored])
    backend.close()

    assert backend.submitted == [15, 21]
    assert backend.get_successful_submission_count() == successes_per_number * len(factored)
    assert sleep.call_args_list == [call(SUBMIT_SPACING)] * len(factored)


@pytest.mark.parametrize(
    ("prime_factors", "composite_factors", "expected"),
    [
        ([3, 3, 5, 7], [], [3, 5]),
        ([3, 5], [1001], [3, 5]),
    ],
    ids=["complete", "partial"],
)
def test_factordb_submits_each_distinct_factor(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] (Fixtures and parameters)
    factordb: FactorDB,
    config: Config,
    http_request: Mock,
    sleep: Mock,
    prime_factors: list[int],
    composite_factors: list[int],
    expected: list[int],
) -> None:
    """Test that FactorDB receives each distinct prime factor, skipping the largest of a complete factorization."""
    number = Number(315, config, FactoringStats(config.stats_path, read_only=True), factordb)
    number.prime_factors = prime_factors
    number.composite_factors = composite_factors

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == len(expected)
    assert [c.kwargs["data"]["factor"] for c in http_request.call_args_list] == [str(f) for f in expected]
    # One spacing between consecutive factors, plus one after the number.
    assert sleep.call_args_list == [call(SUBMIT_SPACING)] * len(expected)


def test_fetch_maps_criteria(factordb: FactorDB, http_request: Mock) -> None:
    """Test that fetch criteria map to FactorDB query parameters."""
    numbers = factordb.fetch(FetchCriteria(count=7, min_digits=2, skip_count=11))

    assert {number.n for number in numbers} == {15}
    http_request.assert_called_once_with(
        "GET",
        "https://factordb.com/listtype.php",
        params={"t": 3, "mindig": 2, "perpage": 7, "start": 11, "download": 1},
        data=None,
        files=None,
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
