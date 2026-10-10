# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for generic backend behavior and FactorDB."""

from __future__ import annotations

import math
import threading

from typing import TYPE_CHECKING, Any, override
from unittest.mock import Mock, call

import pytest
import requests

from loguru import logger

from factortool.backend import NO_WORK_DELAY, SUBMIT_SPACING, Backend, BaseBackend, FetchCriteria, parse_composites
from factortool.config import Config
from factortool.factordb import (
    API_URL,
    MAX_FETCH_COUNT,
    REJECTED_FACTORS_ERROR,
    SUBMIT_RPC_ATTEMPTS,
    FactorDB,
    RpcError,
    parse_rpc_responses,
)
from factortool.http import HttpClient, PermanentHttpError, build_user_agent
from factortool.interrupt import InterruptState
from factortool.number import Number
from factortool.stats import FactoringStats

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
    from pathlib import Path


class FakeBackend(BaseBackend):
    """Minimal backend serving canned fetch responses and recording submissions."""

    name = "Fake"
    assigns_work = False

    def __init__(self, config: Config, stats: FactoringStats, responses: Iterable[str | Exception]) -> None:
        """Initialize the fake backend with the fetch responses to return in order."""
        self.responses = iter(responses)
        self.submitted: list[int] = []
        super().__init__(config, stats, 1.0, "")

    @override
    def _request_composites(self, criteria: FetchCriteria) -> list[int]:
        response = next(self.responses)

        if isinstance(response, Exception):
            raise response

        return parse_composites(response)

    @override
    def _submit_number(self, number: Number) -> bool:
        self.submitted.append(number.n)
        return True


@pytest.fixture
def sleep(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Record sleeps instead of sleeping.

    Returns:
        Mock: The recording sleep mock.
    """
    sleep = Mock()
    monkeypatch.setattr("factortool.backend.time.sleep", sleep)
    return sleep


@pytest.fixture
def wait(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Record interruptible waits instead of waiting, reporting that no interrupt arrived.

    Returns:
        Mock: The recording wait mock.
    """
    wait = Mock(return_value=False)
    monkeypatch.setattr(InterruptState, "wait", wait)
    return wait


def make_fake_backend(config: Config, responses: Iterable[str | Exception] = ()) -> FakeBackend:
    """Create a fake backend. Callers are responsible for closing it.

    Returns:
        FakeBackend: The fake backend.
    """
    return FakeBackend(config, FactoringStats(config.stats_path, read_only=True), responses)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Build an isolated configuration without FactorDB credentials.

    Returns:
        Config: The test configuration.
    """
    return Config.model_construct(
        assignment_state_path=tmp_path / "assignment_state.json",
        backend="factordb",
        batch_state_path=tmp_path / "batch_state.json",
        cado_nfs_path=tmp_path / "cado-nfs.py",
        factordb_cooldown_period=0.0,
        max_threads=1,
        mersenne_ca_cooldown_period=0.0,
        result_output_path=tmp_path / "results",
        stats_path=tmp_path / "stats.json",
        work_path=tmp_path / "work",
        yafu_path=tmp_path / "yafu",
    )


def rpc_response(*results: object) -> Mock:
    """Build an HTTP response carrying a batch of successful JSON-RPC results.

    Returns:
        Mock: The response mock.
    """
    body = [{"jsonrpc": "2.0", "id": i, "result": result} for i, result in enumerate(results)]
    return Mock(spec=requests.Response, json=Mock(return_value=body))


def rpc_error(code: int, message: str) -> Mock:
    """Build an HTTP response carrying a single JSON-RPC error.

    Returns:
        Mock: The response mock.
    """
    body = [{"jsonrpc": "2.0", "id": 0, "error": {"code": code, "message": message}}]
    return Mock(spec=requests.Response, json=Mock(return_value=body))


def row(n: int, fid: int = 1) -> dict[str, object]:
    """Build a list_by_type row, truncating the preview as FactorDB does.

    Returns:
        dict[str, object]: The row.
    """
    decimal = str(n)
    return {"fid": fid, "digits": len(decimal), "preview": decimal[:100], "tail": decimal[100:][-100:], "term": ""}


def rpc_calls(http_request: Mock) -> list[tuple[str, dict[str, Any]]]:
    """Collect the JSON-RPC calls made through a recording request mock.

    Returns:
        list[tuple[str, dict[str, Any]]]: The method and parameters of each call, in order.
    """
    return [(c["method"], c["params"]) for request in http_request.call_args_list for c in request.kwargs["json"]]


@pytest.fixture
def http_request(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace HTTP requests with a recording mock returning a composite.

    Returns:
        Mock: The recording request mock.
    """
    request = Mock(return_value=rpc_response({"rows": [row(15)], "has_more": False}))
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


def test_parse_composites_reads_one_number_per_line() -> None:
    """Test that the fetch response is a plain list of decimal integers."""
    assert parse_composites("101\n103\n107\n") == [101, 103, 107]


def test_parse_composites_treats_empty_body_as_no_work() -> None:
    """Test that an empty fetch response is interpreted as no work."""
    assert parse_composites("") == []
    assert parse_composites("\n\n  \n") == []


def test_parse_composites_rejects_non_numeric_content() -> None:
    """Test that non-numeric content raises a ValueError."""
    with pytest.raises(ValueError, match="invalid literal"):
        parse_composites("<html>service unavailable</html>")


def test_base_fetch_retries_failures_with_backoff(config: Config, wait: Mock) -> None:
    """Test that malformed responses and request errors are retried with exponential backoff."""
    backend = make_fake_backend(config, ["invalid", requests.RequestException("rejected"), "15\n"])

    try:
        numbers = backend.fetch(FetchCriteria(count=3, min_digits=2))
    finally:
        backend.close()

    assert {number.n for number in numbers} == {15}
    assert wait.call_args_list[:2] == [call(1.0), call(2.0)]


@pytest.mark.parametrize("blank_response", ["", "\n  \n"], ids=["empty", "whitespace"])
def test_base_fetch_retries_when_no_work_is_available(config: Config, wait: Mock, blank_response: str) -> None:
    """Test that a blank response is retried after a delay until composites are returned."""
    backend = make_fake_backend(config, [blank_response] * 5 + ["15\n"])

    try:
        numbers = backend.fetch(FetchCriteria(count=3, min_digits=2))
    finally:
        backend.close()

    assert {number.n for number in numbers} == {15}
    assert wait.call_args_list == [call(NO_WORK_DELAY)] * 5


def test_base_fetch_treats_all_excluded_by_max_digits_as_no_work(config: Config, wait: Mock) -> None:
    """Test that a response entirely above max_digits is retried like an empty one."""
    backend = make_fake_backend(config, ["1001\n"] * 5 + ["15\n"])

    try:
        assert backend.fetch(FetchCriteria(count=3, min_digits=2, max_digits=3)) == {
            Number(15, config, FactoringStats(config.stats_path, read_only=True), backend)
        }
    finally:
        backend.close()

    assert wait.call_args_list == [call(NO_WORK_DELAY)] * 5


@pytest.mark.parametrize("response", ["", requests.RequestException("rejected")], ids=["no-work", "request-error"])
def test_base_fetch_abandons_the_wait_on_an_interrupt(config: Config, wait: Mock, response: str | Exception) -> None:
    """Test that an interrupt during a retry wait abandons the fetch instead of retrying."""
    wait.return_value = True
    backend = make_fake_backend(config, [response, "15\n"])

    try:
        assert backend.fetch(FetchCriteria(count=3, min_digits=2)) == set()
    finally:
        backend.close()

    assert wait.call_count == 1


def test_base_close_flushes_submissions(config: Config, sleep: Mock) -> None:
    """Test that closing submits every queued number with factors and counts the accepted factorizations."""
    backend = make_fake_backend(config)
    stats = FactoringStats(config.stats_path, read_only=True)
    factored = [Number(n, config, stats, backend) for n in (15, 21)]
    unfactored = Number(35, config, stats, backend)

    for number, factors in zip(factored, ([3, 5], [3, 7]), strict=True):
        number.prime_factors = factors

    backend.submit([*factored, unfactored])
    backend.close()

    assert backend.submitted == [15, 21]
    assert backend.get_successful_submission_count() == len(factored)
    assert sleep.call_args_list == [call(SUBMIT_SPACING)] * len(factored)


@pytest.mark.parametrize(
    ("n", "prime_factors", "composite_factors", "expected"),
    [
        (315, [3, 3, 5, 7], [], [3, 5]),
        (15015, [3, 5], [1001], [3, 5]),
        (9, [3, 3], [], [3]),
        (75, [3, 5, 5], [], [3, 5]),
    ],
    ids=["complete", "partial", "prime-power", "repeated-largest"],
)
def test_factordb_submits_distinct_factors_in_one_call(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] (Fixtures and parameters)
    factordb: FactorDB,
    config: Config,
    http_request: Mock,
    sleep: Mock,
    n: int,
    prime_factors: list[int],
    composite_factors: list[int],
    expected: list[int],
) -> None:
    """Test that FactorDB receives each distinct prime factor, skipping the largest of a complete factorization."""
    http_request.return_value = rpc_response({"id": {"fid": 1, "kind": "stored"}, "status": "CF"})
    number = Number(n, config, FactoringStats(config.stats_path, read_only=True), factordb)
    number.prime_factors = prime_factors
    number.composite_factors = composite_factors

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == 1
    http_request.assert_called_once()
    assert rpc_calls(http_request) == [
        (
            "report_factors",
            {"target": {"expr": str(n)}, "factors": [str(f) for f in expected], "credit": False},
        )
    ]
    assert sleep.call_args_list == [call(SUBMIT_SPACING)]


def test_factordb_skips_submission_of_a_prime(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock
) -> None:
    """Test that a number that turned out to be prime is not submitted."""
    number = Number(7, config, FactoringStats(config.stats_path, read_only=True), factordb)

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == 0
    http_request.assert_not_called()
    assert sleep.call_args_list == [call(SUBMIT_SPACING)]


def test_factordb_counts_rejected_submission_as_failure(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock
) -> None:
    """Test that a factor rejected by FactorDB is logged and not counted or retried."""
    http_request.return_value = rpc_error(*REJECTED_FACTORS_ERROR)
    number = Number(15, config, FactoringStats(config.stats_path, read_only=True), factordb)
    number.prime_factors = [3, 5]
    number.composite_factors = []

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == 0
    http_request.assert_called_once()
    assert sleep.call_args_list == [call(SUBMIT_SPACING)]


def test_factordb_gives_up_on_repeated_submission_errors(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock
) -> None:
    """Test that a submission still failing after SUBMIT_RPC_ATTEMPTS attempts is abandoned."""
    http_request.return_value = rpc_error(-32000, "database busy")
    number = Number(15, config, FactoringStats(config.stats_path, read_only=True), factordb)
    number.prime_factors = [3, 5]
    number.composite_factors = []

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == 0
    assert http_request.call_count == SUBMIT_RPC_ATTEMPTS
    assert sleep.call_args_list == [call(0.1), call(0.2), call(0.4), call(0.8), call(SUBMIT_SPACING)]


def test_factordb_retries_submission_errors(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock
) -> None:
    """Test that an error reported in a successful HTTP response is retried rather than discarding the submission."""
    http_request.side_effect = [
        rpc_error(-32603, "internal error"),
        Mock(spec=requests.Response, json=Mock(return_value=[])),
        rpc_response({"status": "FF"}),
    ]
    number = Number(15, config, FactoringStats(config.stats_path, read_only=True), factordb)
    number.prime_factors = [3, 5]
    number.composite_factors = []

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == 1
    assert http_request.call_count == 3  # ruff: ignore[magic-value-comparison]
    assert sleep.call_args_list == [call(0.1), call(0.2), call(SUBMIT_SPACING)]


def test_factordb_does_not_retry_malformed_submissions(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock
) -> None:
    """Test that an error indicating a malformed request is not retried."""
    http_request.return_value = rpc_error(-32602, "invalid params")
    number = Number(15, config, FactoringStats(config.stats_path, read_only=True), factordb)
    number.prime_factors = [3, 5]
    number.composite_factors = []

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == 0
    http_request.assert_called_once()
    assert sleep.call_args_list == [call(SUBMIT_SPACING)]


def make_factor_submission_responder(results_by_target: Mapping[str, object]) -> Callable[..., Mock]:
    """Create a callback for http_request.side_effect that simulates FactorDB submission responses.

    results_by_target maps each submitted number's expression (such as "15") to its report_factors result dictionary.
    The callback looks up each result by that expression, so submission order and batching do not affect the response.
    Other RPC calls receive a successful whoami result for the test user "x" to simulate being signed in.

    Returns:
        Callable[..., Mock]: A request callback returning a mock HTTP response containing the JSON-RPC results.
    """

    def respond(*_args: object, json: list[dict[str, Any]], **_kwargs: object) -> Mock:
        results = [
            results_by_target[c["params"]["target"]["expr"]]
            if c["method"] == "report_factors"
            else {"found": True, "login": "x"}
            for c in json
        ]
        return rpc_response(*results)

    return respond


def submit_signed_in(config: Config, http_request: Mock, reports: Mapping[str, object]) -> list[str]:
    """Submit 15, 21 and 35 as a signed-in user, then close, with FactorDB answering each with the given report.

    Returns:
        list[str]: The messages logged while submitting and closing.
    """
    http_request.side_effect = make_factor_submission_responder(reports)
    backend = FactorDB(config.model_copy(update={"factordb_api_token": "secret"}), FactoringStats(config.stats_path))
    stats = FactoringStats(config.stats_path, read_only=True)
    numbers = [Number(n, config, stats, backend) for n in (15, 21, 35)]

    for number, factors in zip(numbers, ([3, 5], [3, 7], [5, 7]), strict=True):
        number.prime_factors = factors
        number.composite_factors = []

    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")

    try:
        backend.submit(numbers)
        backend.close()
    finally:
        logger.remove(sink)

    return messages


def test_factordb_reports_credited_factors(config: Config, http_request: Mock, sleep: Mock) -> None:
    """Test that credit is claimed for each submission and the credit FactorDB reports is totalled as a lower bound."""
    # The response for 21 claims no credit, as when a submission is repeated after its first response was lost.
    reports = {
        "15": {"status": "FF", "credited": 1},
        "21": {"status": "CF", "credited": 0},
        "35": {"status": "FF", "credited": 1},
    }

    messages = submit_signed_in(config, http_request, reports)

    assert [params["credit"] for method, params in rpc_calls(http_request) if method == "report_factors"] == [True] * 3
    assert "FactorDB credited at least 2 factors to your account\n" in messages
    assert all(c == call(SUBMIT_SPACING) for c in sleep.call_args_list)


def make_factored(config: Config, backend: BaseBackend, *factorizations: list[int]) -> list[Number]:
    """Build completely factored numbers from their prime factors.

    Returns:
        list[Number]: The factored numbers.
    """
    stats = FactoringStats(config.stats_path, read_only=True)
    numbers: list[Number] = []

    for factors in factorizations:
        number = Number(math.prod(factors), config, stats, backend)
        number.prime_factors = factors
        number.composite_factors = []
        numbers.append(number)

    return numbers


def rpc_batch(*entries: object) -> Mock:
    """Build an HTTP response to a batch of JSON-RPC calls, where an RpcError entry stands for that call failing.

    Returns:
        Mock: The response mock.
    """
    body = [
        {"jsonrpc": "2.0", "id": i, "error": {"code": entry.code, "message": entry.message}}
        if isinstance(entry, RpcError)
        else {"jsonrpc": "2.0", "id": i, "result": entry}
        for i, entry in enumerate(entries)
    ]
    return Mock(spec=requests.Response, json=Mock(return_value=body))


def test_factordb_submits_a_batch_in_one_request(factordb: FactorDB, config: Config, http_request: Mock) -> None:
    """Test that several numbers are reported in a single batch request, one report_factors call each."""
    http_request.return_value = rpc_batch({"status": "FF"}, {"status": "FF"}, {"status": "FF"})
    numbers = make_factored(config, factordb, [3, 5], [3, 7], [5, 7])

    assert factordb._submit_numbers(numbers) == len(numbers)

    http_request.assert_called_once()
    assert [params["target"]["expr"] for _, params in rpc_calls(http_request)] == ["15", "21", "35"]


def test_factordb_retries_only_failed_calls_in_a_batch(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock
) -> None:
    """Test that a failed call is retried alone, while successes and rejections in the same batch are kept."""
    http_request.side_effect = [
        rpc_batch({"status": "FF"}, RpcError(*REJECTED_FACTORS_ERROR), RpcError(-32000, "database busy")),
        rpc_batch({"status": "FF"}),
    ]
    numbers = make_factored(config, factordb, [3, 5], [3, 7], [5, 7])

    assert factordb._submit_numbers(numbers) == 2  # ruff: ignore[magic-value-comparison]

    calls = [[c["params"]["target"]["expr"] for c in request.kwargs["json"]] for request in http_request.call_args_list]
    assert calls == [["15", "21", "35"], ["35"]]
    assert sleep.call_args_list == [call(0.1)]


def test_factordb_retries_a_whole_batch_after_a_malformed_response(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock
) -> None:
    """Test that every call is retried when the batch response as a whole is malformed."""
    http_request.side_effect = [
        Mock(spec=requests.Response, json=Mock(return_value=[])),
        rpc_batch({"status": "FF"}, {"status": "FF"}),
    ]
    numbers = make_factored(config, factordb, [3, 5], [3, 7])

    assert factordb._submit_numbers(numbers) == len(numbers)
    assert http_request.call_count == 2  # ruff: ignore[magic-value-comparison]
    assert sleep.call_args_list == [call(0.1)]


def test_base_worker_submits_a_backlog_in_batches(config: Config, sleep: Mock) -> None:
    """Test that numbers queued while a submission is in progress are submitted together, up to the batch size."""
    started = threading.Event()
    release = threading.Event()
    batches: list[list[int]] = []

    class BatchingBackend(FakeBackend):
        submit_batch_size = 2

        @override
        def _submit_numbers(self, numbers: Sequence[Number]) -> int:
            started.set()
            release.wait(timeout=5.0)
            batches.append([number.n for number in numbers])
            return len(numbers)

    backend = BatchingBackend(config, FactoringStats(config.stats_path, read_only=True), ())
    numbers = make_factored(config, backend, [3, 5], [3, 7], [5, 7], [7, 11])

    try:
        backend.submit(numbers[:1])
        assert started.wait(timeout=5.0)
        backend.submit(numbers[1:])
        release.set()
    finally:
        backend.close()

    assert batches == [[15], [21, 35], [77]]
    assert backend.get_successful_submission_count() == len(numbers)
    assert sleep.call_args_list == [call(SUBMIT_SPACING)] * 3


def test_factordb_counts_unaccepted_submission_as_failure(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock
) -> None:
    """Test that a number left with no known factors after reporting is not counted, as FactorDB returns no error."""
    http_request.return_value = rpc_response(
        {"created_ids": [], "id": {"kind": "literal", "text": "15"}, "status": "C"}
    )
    number = Number(15, config, FactoringStats(config.stats_path, read_only=True), factordb)
    number.prime_factors = [3, 5]
    number.composite_factors = []

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == 0
    http_request.assert_called_once()
    assert sleep.call_args_list == [call(SUBMIT_SPACING)]


def test_fetch_maps_criteria(factordb: FactorDB, http_request: Mock) -> None:
    """Test that fetch criteria map to list_by_type parameters."""
    numbers = factordb.fetch(FetchCriteria(count=7, min_digits=2, skip_count=11))

    assert {number.n for number in numbers} == {15}
    http_request.assert_called_once_with(
        "POST",
        API_URL,
        params=None,
        data=None,
        files=None,
        json=[
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "list_by_type",
                "params": {"table": "C", "min_digits": 2, "offset": 11, "limit": 7},
            }
        ],
        timeout=10.0,
        max_attempts=None,
        interruptible=True,
    )


def test_fetch_caps_count(factordb: FactorDB, http_request: Mock) -> None:
    """Test that requests for more numbers than list_by_type allows are capped."""
    numbers = factordb.fetch(FetchCriteria(count=MAX_FETCH_COUNT + 1, min_digits=2))

    assert {number.n for number in numbers} == {15}
    assert rpc_calls(http_request)[0][1]["limit"] == MAX_FETCH_COUNT


def test_fetch_zero_skips_http_request(factordb: FactorDB, http_request: Mock) -> None:
    """Test that requesting zero numbers returns an empty set without HTTP."""
    assert factordb.fetch(FetchCriteria(count=0, min_digits=2)) == set()
    http_request.assert_not_called()


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([row(15), row(999), row(1001)], {15, 999}),
        ([row(999)], {999}),
    ],
    ids=["mixed", "inclusive-bound"],
)
def test_fetch_filters_max_digits(
    factordb: FactorDB, http_request: Mock, rows: list[dict[str, object]], expected: set[int]
) -> None:
    """Test that max_digits is sent to FactorDB and also enforced locally."""
    # A finite side effect makes an unexpected retry fail instead of hanging.
    http_request.side_effect = [rpc_response({"rows": rows, "has_more": False})]

    numbers = factordb.fetch(FetchCriteria(count=3, min_digits=2, max_digits=3))

    assert {number.n for number in numbers} == expected
    http_request.assert_called_once()
    assert rpc_calls(http_request)[0][1]["max_digits"] == 3  # ruff: ignore[magic-value-comparison]


def test_fetch_requests_decimals_for_truncated_previews(factordb: FactorDB, http_request: Mock) -> None:
    """Test that numbers whose preview is truncated are expanded with a batch of get_number calls."""
    large = [10**150 + 7, 10**200 + 9]
    http_request.side_effect = [
        rpc_response({"rows": [row(15, 1), row(large[0], 2), row(large[1], 3)], "has_more": True}),
        rpc_response({"decimal": str(large[0])}, {"decimal": str(large[1])}),
    ]

    numbers = factordb._request_composites(FetchCriteria(count=3, min_digits=2))

    assert numbers == [15, *large]
    assert rpc_calls(http_request)[1:] == [
        ("get_number", {"target": {"id": 2}, "decimal": True}),
        ("get_number", {"target": {"id": 3}, "decimal": True}),
    ]


def test_fetch_skips_numbers_without_decimals(factordb: FactorDB, http_request: Mock) -> None:
    """Test that a number whose decimal expansion FactorDB omits is skipped."""
    http_request.side_effect = [
        rpc_response({"rows": [row(15, 1), row(10**150 + 7, 2)], "has_more": False}),
        rpc_response({"decimal_omitted": True}),
    ]

    assert factordb._request_composites(FetchCriteria(count=2, min_digits=2)) == [15]


def test_fetch_skips_decimals_of_the_wrong_length(factordb: FactorDB, http_request: Mock) -> None:
    """Test that a number whose decimal expansion disagrees with its reported digit count is skipped."""
    http_request.side_effect = [
        rpc_response({"rows": [row(15, 1), row(10**150 + 7, 2)], "has_more": False}),
        rpc_response({"decimal": "123"}),
    ]

    assert factordb._request_composites(FetchCriteria(count=2, min_digits=2)) == [15]


def test_parse_rpc_responses_orders_results_by_id() -> None:
    """Test that batch results are matched to their calls by id rather than position."""
    body = [{"jsonrpc": "2.0", "id": 1, "result": "b"}, {"jsonrpc": "2.0", "id": 0, "result": "a"}]

    assert parse_rpc_responses(body, 2) == ["a", "b"]


@pytest.mark.parametrize(
    "body",
    [
        {"jsonrpc": "2.0", "id": 0, "result": 1},
        [],
        ["result"],
        [{"jsonrpc": "2.0", "id": 1, "result": 1}],
        [{"jsonrpc": "2.0", "id": 0}],
        [{"jsonrpc": "2.0", "id": None, "result": 1}],
    ],
    ids=["not-a-batch", "wrong-length", "not-an-object", "missing-id", "no-result", "null-id"],
)
def test_parse_rpc_responses_rejects_malformed_responses(body: object) -> None:
    """Test that malformed batch responses raise a ValueError."""
    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad] (Pydantic and our own messages differ)
        parse_rpc_responses(body, 1)


@pytest.mark.parametrize("code", [-32700, -32600, -32601, -32602])
def test_parse_rpc_responses_treats_invalid_requests_as_permanent(code: int) -> None:
    """Test that errors indicating an invalid request are not retried."""
    with pytest.raises(PermanentHttpError, match="unsupported"):
        parse_rpc_responses(rpc_error(code, "unsupported").json(), 1)


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_parse_rpc_responses_raises_errors_without_an_id(*, batch: bool) -> None:
    """Test that an error not tied to any call, as for an unparseable batch, is raised with its own message."""
    response: dict[str, object] = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}

    with pytest.raises(PermanentHttpError, match="Parse error"):
        parse_rpc_responses([response] if batch else response, 2)


def test_parse_rpc_responses_raises_other_errors() -> None:
    """Test that other errors are raised as ordinary request exceptions."""
    with pytest.raises(RpcError, match="Division by zero") as exc_info:
        parse_rpc_responses(rpc_error(-32000, "eval: Division by zero").json(), 1)

    assert not isinstance(exc_info.value, PermanentHttpError)


def test_factordb_verifies_api_token(config: Config, http_request: Mock) -> None:
    """Test that a configured token is verified, sent with requests and used to claim credit."""
    http_request.return_value = rpc_response({"found": True, "login": "FactorFinder"})
    backend = FactorDB(config.model_copy(update={"factordb_api_token": "secret"}), FactoringStats(config.stats_path))

    try:
        headers = backend._http_client.session.headers
        assert rpc_calls(http_request) == [("whoami", {"session": "secret"})]
        assert headers["X-Fdb-User-Token"] == "secret"
        assert headers["User-Agent"] == build_user_agent()
        assert backend._signed_in
    finally:
        backend.close()


def test_factordb_rejects_unrecognized_api_token(config: Config, http_request: Mock) -> None:
    """Test that an unrecognized token is a permanent error rather than a silent fallback to anonymous submission."""
    http_request.return_value = rpc_response({"found": False})

    with pytest.raises(PermanentHttpError, match="does not recognize"):
        FactorDB(config.model_copy(update={"factordb_api_token": "bogus"}), FactoringStats(config.stats_path))


@pytest.mark.parametrize(
    "response",
    [requests.RequestException("unreachable"), rpc_response({"login": "FactorFinder"})],
    ids=["request-error", "malformed"],
)
def test_factordb_stops_if_token_cannot_be_verified(config: Config, http_request: Mock, response: object) -> None:
    """Test that a token that cannot be verified stops startup rather than claiming credit it may not get."""
    http_request.side_effect = [response]

    with pytest.raises(requests.RequestException, match="Unable to verify"):
        FactorDB(config.model_copy(update={"factordb_api_token": "secret"}), FactoringStats(config.stats_path))


def test_factordb_without_token_is_anonymous(factordb: FactorDB, http_request: Mock) -> None:
    """Test that no token means no verification request and no token header."""
    http_request.assert_not_called()
    assert "X-Fdb-User-Token" not in factordb._http_client.session.headers
    assert not factordb._signed_in


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


@pytest.mark.parametrize("status_code", [400, 403, 404])
def test_base_fetch_does_not_retry_permanent_errors(config: Config, sleep: Mock, status_code: int) -> None:
    """Test that a permanent client error escapes the fetch loop instead of being retried."""
    backend = make_fake_backend(config, [PermanentHttpError(f"HTTP {status_code}"), "15\n"])

    try:
        with pytest.raises(PermanentHttpError):
            backend.fetch(FetchCriteria(count=3, min_digits=2))
    finally:
        backend.close()

    sleep.assert_not_called()


def test_factordb_stops_if_token_verification_is_invalid(config: Config, http_request: Mock) -> None:
    """Test that a permanent error while verifying the token is raised rather than ignored."""
    http_request.return_value = rpc_error(-32602, "invalid params")

    with pytest.raises(PermanentHttpError, match="invalid params"):
        FactorDB(config.model_copy(update={"factordb_api_token": "secret"}), FactoringStats(config.stats_path))
