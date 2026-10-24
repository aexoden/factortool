# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for generic backend behavior and FactorDB."""

from __future__ import annotations

import math
import threading
import time

from typing import TYPE_CHECKING, Any, override
from unittest.mock import Mock, call

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
    from pathlib import Path

import pytest
import requests

from loguru import logger

from factortool.assignments import Assignment, AssignmentState, AssignmentStore
from factortool.backend import (
    NO_WORK_DELAY,
    SUBMIT_ATTEMPTS,
    Backend,
    BaseBackend,
    FetchCriteria,
    SubmitOutcome,
    parse_composites,
)
from factortool.factordb import (
    API_URL,
    MAX_FETCH_COUNT,
    REJECTED_FACTORS_ERROR,
    FactorDB,
    RpcError,
    parse_rpc_responses,
)
from factortool.http import HttpClient, PermanentHttpError, build_user_agent
from factortool.interrupt import ABORT, Interrupted, InterruptState
from factortool.number import Number
from factortool.stats import FactoringStats
from factortool.submissions import Submission, SubmissionJournal

if TYPE_CHECKING:
    from factortool.config import Config

from .helpers import make_config


class FakeBackend(BaseBackend):
    """Minimal backend serving canned fetch responses and recording submissions."""

    name = "Fake"
    assigns_work = False

    def __init__(self, config: Config, stats: FactoringStats, responses: Iterable[str | Exception]) -> None:
        """Initialize the fake backend with the fetch responses to return in order."""
        self.responses = iter(responses)
        self.requested: list[int] = []
        self.submitted: list[int] = []
        self.outcomes: dict[int, list[SubmitOutcome]] = {}
        super().__init__(config, stats, 1.0, "")
        self._start_submitting()

    @override
    def _request_composites(self, criteria: FetchCriteria) -> list[int]:
        self.requested.append(criteria.count)
        response = next(self.responses)

        if isinstance(response, Exception):
            raise response

        return parse_composites(response)

    @override
    def _submit_number(self, submission: Submission) -> SubmitOutcome:
        self.submitted.append(submission.n)

        outcomes = self.outcomes.get(submission.n)
        return outcomes.pop(0) if outcomes else SubmitOutcome.ACCEPTED


class RefusingBackend(FakeBackend):
    """A fake backend whose service refuses every submission request outright."""

    @override
    def _submit_number(self, submission: Submission) -> SubmitOutcome:
        self.submitted.append(submission.n)
        return SubmitOutcome.DEFERRED


class UnreachableBackend(FakeBackend):
    """A fake backend whose service never answers a submission."""

    @override
    def _submit_number(self, submission: Submission) -> SubmitOutcome:
        self.submitted.append(submission.n)

        while not self._submission_wait(0.01):
            pass

        raise Interrupted


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
def retry_wait(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Record the waits before another attempt at a submission instead of waiting, never abandoning it.

    Returns:
        Mock: The recording wait mock.
    """
    retry_wait = Mock(return_value=False)
    monkeypatch.setattr(BaseBackend, "_submission_wait", retry_wait)
    return retry_wait


@pytest.fixture
def wait(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Record interruptible waits, including those before another attempt at a fetch, instead of waiting.

    No wait reports that an interrupt arrived or a submission request was refused.

    Returns:
        Mock: The recording wait mock.
    """
    wait = Mock(return_value=False)
    monkeypatch.setattr(InterruptState, "wait", wait)
    monkeypatch.setattr(BaseBackend, "_fetch_wait", wait)
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
    return make_config(
        factordb_request_interval=0.0,
        mersenne_ca_request_interval=0.0,
        result_output_path=tmp_path / "results",
        state_path=tmp_path,
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


@pytest.mark.parametrize(
    ("max_fetch_count", "count", "requested", "expected"),
    [
        (None, 1, 1, "Fetching 1 composite number from Fake\n"),
        (None, 3, 3, "Fetching 3 composite numbers from Fake\n"),
        (3, 3, 3, "Fetching 3 composite numbers from Fake\n"),
        (
            2,
            3,
            2,
            "Fetching 2 composite numbers from Fake (reduced from the 3 requested, due to the service maximum)\n",
        ),
    ],
    ids=["single", "uncapped", "at-cap", "capped"],
)
def test_base_fetch_caps_the_count_and_logs_it_once(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] (Fixtures and parameters)
    config: Config,
    wait: Mock,
    monkeypatch: pytest.MonkeyPatch,
    max_fetch_count: int | None,
    count: int,
    requested: int,
    expected: str,
) -> None:
    """Test that a request is reduced to the most the service returns at once, logged once despite retries."""
    monkeypatch.setattr(FakeBackend, "max_fetch_count", max_fetch_count)
    backend = make_fake_backend(config, [requests.RequestException("rejected"), "", "15\n"])
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}", filter=lambda record: record["level"].name == "INFO")

    try:
        numbers = backend.fetch(FetchCriteria(count=count, min_digits=2))
    finally:
        logger.remove(sink)
        backend.close()

    assert {number.n for number in numbers} == {15}
    assert backend.requested == [requested] * 3
    assert [x for x in messages if x.startswith("Fetching")] == [expected]
    assert wait.call_args_list == [call(1.0), call(NO_WORK_DELAY)]


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


def test_base_fetch_discards_anything_that_is_not_composite(config: Config, wait: Mock) -> None:
    """Test that numbers with nothing to factor are dropped, and a response of only those is retried."""
    backend = make_fake_backend(config, ["1\n0\n-15\n7\n", "1\n15\n7\n21\n"])

    try:
        numbers = backend.fetch(FetchCriteria(count=4, min_digits=1))
    finally:
        backend.close()

    assert {number.n for number in numbers} == {15, 21}
    assert wait.call_args_list == [call(NO_WORK_DELAY)]


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
    sleep.assert_not_called()


def test_base_summarizes_each_batch_that_resolves_a_submission(config: Config, retry_wait: Mock) -> None:
    """Test that each batch with an accepted or rejected submission is summarized, but one that only failed is not."""
    backend = make_fake_backend(config)
    backend.outcomes[15] = [SubmitOutcome.FAILED, SubmitOutcome.ACCEPTED]
    backend.outcomes[21] = [SubmitOutcome.REJECTED]
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}", filter=lambda record: record["level"].name == "INFO")

    try:
        backend.submit(make_factored(config, backend, [3, 5], [3, 7]))
        backend.close()
    finally:
        logger.remove(sink)

    # The worker may send 21 before or after the first attempt at 15.
    assert sorted(backend.submitted) == [15, 15, 21]
    assert sorted(messages) == [
        "Fake accepted 0 of 1 factorization\n",
        "Fake accepted 1 factorization\n",
        "Successfully submitted 1 factorization to Fake\n",
    ]
    assert retry_wait.called


def test_base_submit_records_the_factorization_as_it_stood(config: Config) -> None:
    """Test that a queued number is recorded with its factors and its assignment's expiry before it is sent."""
    recorded: list[list[Submission]] = []

    class RecordingBackend(FakeBackend):
        @override
        def _submit_number(self, submission: Submission) -> SubmitOutcome:
            recorded.append(SubmissionJournal(config.pending_submissions_path, config.backend).load())
            return super()._submit_number(submission)

    backend = RecordingBackend(config, FactoringStats(config.stats_path, read_only=True), ())
    number = Number(15015, config, FactoringStats(config.stats_path, read_only=True), backend)
    number.prime_factors = [3, 5]
    number.composite_factors = [1001]
    expires_at = time.time() + 600.0
    number.expires_at = expires_at

    backend.submit([number])
    backend.close()

    assert recorded == [[Submission(n=15015, prime_factors=(3, 5), composite_factors=(1001,), expires_at=expires_at)]]
    assert not config.pending_submissions_path.exists()


def test_base_submit_gives_unassigned_work_the_backends_lifetime(config: Config, retry_wait: Mock) -> None:
    """Test that a number without an assignment expires after the backend's submission lifetime."""
    backend = make_fake_backend(config)
    backend.outcomes[15] = [SubmitOutcome.FAILED] * SUBMIT_ATTEMPTS
    start = time.time()

    backend.submit(make_factored(config, backend, [3, 5]))
    backend.close()

    (submission,) = SubmissionJournal(config.pending_submissions_path, config.backend).load()
    assert submission.expires_at == pytest.approx(start + FakeBackend.submission_lifetime, abs=5.0)
    assert retry_wait.called


def test_base_retries_failed_submissions_then_leaves_them_for_the_next_run(
    config: Config, sleep: Mock, retry_wait: Mock
) -> None:
    """Test that a failed submission is retried with backoff, and kept once it runs out of attempts."""
    backend = make_fake_backend(config)
    backend.outcomes[15] = [SubmitOutcome.FAILED, SubmitOutcome.ACCEPTED]
    backend.outcomes[21] = [SubmitOutcome.FAILED] * (SUBMIT_ATTEMPTS + 1)
    backend.outcomes[35] = [SubmitOutcome.REJECTED]

    backend.submit(make_factored(config, backend, [3, 5], [3, 7], [5, 7]))
    backend.close()

    assert backend.submitted.count(15) == 2  # ruff: ignore[magic-value-comparison]
    assert backend.submitted.count(21) == SUBMIT_ATTEMPTS
    assert backend.submitted.count(35) == 1
    assert backend.get_successful_submission_count() == 1
    assert retry_wait.call_args_list[0] == call(1.0)
    sleep.assert_not_called()
    assert journaled(config) == [21]


def test_base_resubmits_what_a_previous_run_left_unsent(config: Config, sleep: Mock) -> None:
    """Test that submissions left in the journal are sent when the next run starts, unless they have expired."""
    journal = SubmissionJournal(config.pending_submissions_path, config.backend)
    journal.add([make_submission(3, 5), make_submission(3, 7, lifetime=-1.0), make_submission(5, 7)])

    backend = make_fake_backend(config)
    backend.close()

    assert backend.submitted == [15, 35]
    assert backend.get_successful_submission_count() == 2  # ruff: ignore[magic-value-comparison]
    assert not config.pending_submissions_path.exists()
    sleep.assert_not_called()


def test_base_discards_a_new_result_whose_assignment_has_expired(config: Config, retry_wait: Mock) -> None:
    """Test that a result from this run is never sent once its assignment has expired."""
    backend = make_fake_backend(config)
    (number,) = make_factored(config, backend, [3, 5])
    number.expires_at = time.time() - 1.0

    backend.submit([number])
    backend.close()

    assert backend.submitted == []
    assert backend.get_successful_submission_count() == 0
    retry_wait.assert_not_called()
    assert not config.pending_submissions_path.exists()


def test_base_stops_submitting_once_a_request_is_refused(config: Config, sleep: Mock, retry_wait: Mock) -> None:
    """Test that a refused request leaves it and everything after it unsent for the next run, without any retries."""
    backend = make_fake_backend(config)
    backend.outcomes[15] = [SubmitOutcome.DEFERRED]

    backend.submit(make_factored(config, backend, [3, 5], [3, 7]))
    backend._submit_thread.join(timeout=5.0)
    assert not backend._submit_thread.is_alive()
    backend.submit(make_factored(config, backend, [5, 7]))
    backend.close()

    assert backend.submitted == [15]
    assert backend.submissions_refused
    assert backend.get_successful_submission_count() == 0
    retry_wait.assert_not_called()
    assert journaled(config) == [15, 21, 35]
    sleep.assert_not_called()


def test_base_refuses_to_fetch_once_a_submission_request_is_refused(config: Config, sleep: Mock) -> None:
    """Test that no work is fetched once the service has refused to accept the results of earlier work."""
    SubmissionJournal(config.pending_submissions_path, config.backend).add([make_submission(3, 5)])

    backend = RefusingBackend(config, FactoringStats(config.stats_path, read_only=True), ["21\n"])
    backend._submit_thread.join(timeout=5.0)
    assert backend.submissions_refused

    try:
        with pytest.raises(PermanentHttpError, match="refused a submission request"):
            backend.fetch(FetchCriteria(count=1, min_digits=1))
    finally:
        backend.close()

    assert backend.requested == []
    assert journaled(config) == [15]
    sleep.assert_not_called()


def test_base_fetch_stops_retrying_once_a_submission_request_is_refused(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a refusal arriving while a fetch waits to try again abandons the fetch rather than retrying it."""
    monkeypatch.setattr("factortool.backend.RETRY_DELAY", 60.0)

    class OverlappingBackend(RefusingBackend):
        @override
        def _request_composites(self, criteria: FetchCriteria) -> list[int]:
            # The refusal arrives from the submission worker while the fetch waits before its next attempt.
            if not self.requested:
                self.submit(make_factored(config, self, [3, 5]))

            return super()._request_composites(criteria)

    backend = OverlappingBackend(
        config, FactoringStats(config.stats_path, read_only=True), [requests.RequestException("503"), "21\n"]
    )
    start = time.monotonic()

    try:
        with pytest.raises(PermanentHttpError, match="refused a submission request"):
            backend.fetch(FetchCriteria(count=1, min_digits=1))
    finally:
        backend.close()

    assert time.monotonic() - start < 10.0  # ruff: ignore[magic-value-comparison]
    assert backend.requested == [1]
    assert journaled(config) == [15]


def test_base_keeps_the_assignments_fetched_as_a_submission_request_is_refused(config: Config) -> None:
    """Test that work returned by a fetch already sent when a submission request is refused is kept for the next run."""

    class InFlightBackend(AssigningBackend):
        @override
        def _request_composites(self, criteria: FetchCriteria) -> list[int]:
            self.submit(make_factored(config, self, [3, 5]))
            self._submit_thread.join(timeout=5.0)
            return super()._request_composites(criteria)

        @override
        def _submit_number(self, submission: Submission) -> SubmitOutcome:
            self.submitted.append(submission.n)
            return SubmitOutcome.DEFERRED

    backend = InFlightBackend(config, FactoringStats(config.stats_path, read_only=True), ["21\n"])

    try:
        with pytest.raises(PermanentHttpError, match="refused a submission request"):
            backend.fetch(FetchCriteria(count=1, min_digits=1))
    finally:
        backend.close()

    assert [x.n for x in retained(config)] == [21]
    assert journaled(config) == [15]


def test_base_defers_the_rest_of_a_batch_after_a_refused_request(config: Config) -> None:
    """Test that the default batch submission sends nothing further once a request has been refused."""
    backend = make_fake_backend(config)
    backend.outcomes[21] = [SubmitOutcome.DEFERRED]

    try:
        outcomes = backend._submit_numbers([make_submission(3, 5), make_submission(3, 7), make_submission(5, 7)])
    finally:
        backend.close()

    assert outcomes == [SubmitOutcome.ACCEPTED, SubmitOutcome.DEFERRED, SubmitOutcome.DEFERRED]
    assert backend.submitted == [15, 21]


def test_base_honors_a_rate_limit_from_a_previous_run(config: Config) -> None:
    """Test that a rate limit recorded by a previous run holds back the next run's requests, however long it is."""
    rate_limited_until = time.time() + 6 * 3600.0
    SubmissionJournal(config.pending_submissions_path, config.backend).note_rate_limit(rate_limited_until)

    backend = make_fake_backend(config)

    try:
        assert backend._http_client.rate_limited_until == rate_limited_until
    finally:
        backend.close()


def test_base_close_gives_up_on_an_unresponsive_service(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that closing stops waiting once nothing has been resolved for FLUSH_TIMEOUT, keeping what is unsent."""
    monkeypatch.setattr("factortool.backend.FLUSH_TIMEOUT", 0.2)
    backend = UnreachableBackend(config, FactoringStats(config.stats_path, read_only=True), ())
    backend.submit(make_factored(config, backend, [3, 5], [3, 7]))
    start = time.monotonic()

    backend.close()

    assert time.monotonic() - start < 5.0  # ruff: ignore[magic-value-comparison]
    assert backend.submitted == [15]
    assert backend.get_successful_submission_count() == 0
    assert journaled(config) == [15, 21]


def test_base_close_gives_up_at_once_on_an_interrupt(config: Config) -> None:
    """Test that an interrupt received while closing abandons the remaining submissions without losing them."""
    backend = UnreachableBackend(config, FactoringStats(config.stats_path, read_only=True), ())
    backend.submit(make_factored(config, backend, [3, 5]))
    timer = threading.Timer(0.2, setattr, (backend._interrupts, "_level", 1))
    timer.start()
    start = time.monotonic()

    try:
        backend.close()
    finally:
        timer.cancel()

    assert time.monotonic() - start < 5.0  # ruff: ignore[magic-value-comparison]
    assert journaled(config) == [15]


def test_base_close_waits_out_a_rate_limit(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that time spent waiting for a rate limit to end does not count toward giving up."""
    monkeypatch.setattr("factortool.backend.FLUSH_TIMEOUT", 0.2)
    rate_limit = 1.0

    class RateLimitedBackend(FakeBackend):
        @override
        def _submit_number(self, submission: Submission) -> SubmitOutcome:
            if self._submission_wait(self._http_client.rate_limited_until - time.time()):
                raise Interrupted

            return super()._submit_number(submission)

    backend = RateLimitedBackend(config, FactoringStats(config.stats_path, read_only=True), ())
    backend._http_client._rate_limited_until = time.time() + rate_limit
    backend.submit(make_factored(config, backend, [3, 5]))
    backend._closing_since = time.monotonic()
    backend._closing.set()
    start = time.monotonic()

    backend.close()

    assert time.monotonic() - start >= rate_limit * 0.9
    assert backend.submitted == [15]
    assert not config.pending_submissions_path.exists()


def test_base_close_does_not_wait_out_a_rate_limit_once_told_to_abandon(config: Config) -> None:
    """Test that a run ended by a termination signal or a third interrupt leaves rather than wait for a rate limit."""
    backend = UnreachableBackend(config, FactoringStats(config.stats_path, read_only=True), ())
    backend._http_client._rate_limited_until = time.time() + 600.0
    backend._interrupts._level = ABORT
    backend.submit(make_factored(config, backend, [3, 5]))
    start = time.monotonic()

    backend.close()

    assert time.monotonic() - start < 5.0  # ruff: ignore[magic-value-comparison]
    assert journaled(config) == [15]


def test_base_wait_is_cut_short_when_closing_begins(config: Config) -> None:
    """Test that a backoff already under way ends when the backend starts closing, so the attempt is made at once."""
    backend = make_fake_backend(config)
    timer = threading.Timer(0.2, backend._closing.set)
    timer.start()
    start = time.monotonic()

    try:
        assert not backend._submission_wait(30.0)
    finally:
        timer.cancel()
        backend.close()

    assert time.monotonic() - start < 5.0  # ruff: ignore[magic-value-comparison]


def test_base_close_survives_a_failed_worker(config: Config) -> None:
    """Test that a bug in a submission leaves the results in the journal instead of hanging or losing them."""

    class BrokenBackend(FakeBackend):
        @override
        def _submit_number(self, submission: Submission) -> SubmitOutcome:
            msg = "bug"
            raise RuntimeError(msg)

    backend = BrokenBackend(config, FactoringStats(config.stats_path, read_only=True), ())
    backend.submit(make_factored(config, backend, [3, 5]))
    backend.close()

    assert journaled(config) == [15]


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
    sleep.assert_not_called()


def test_factordb_skips_submission_of_a_prime(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock
) -> None:
    """Test that a number that turned out to be prime is not submitted."""
    number = Number(7, config, FactoringStats(config.stats_path, read_only=True), factordb)

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == 0
    http_request.assert_not_called()
    sleep.assert_not_called()


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
    sleep.assert_not_called()


def test_factordb_leaves_repeated_submission_errors_for_the_next_run(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock, retry_wait: Mock
) -> None:
    """Test that a submission still failing after SUBMIT_ATTEMPTS attempts is kept, but not attempted again."""
    http_request.return_value = rpc_error(-32000, "database busy")
    number = Number(15, config, FactoringStats(config.stats_path, read_only=True), factordb)
    number.prime_factors = [3, 5]
    number.composite_factors = []

    factordb.submit([number])
    factordb.close()

    assert factordb.get_successful_submission_count() == 0
    assert http_request.call_count == SUBMIT_ATTEMPTS
    assert retry_wait.call_args_list == [call(1.0), call(2.0), call(4.0), call(8.0)]
    sleep.assert_not_called()
    assert journaled(config) == [15]


def test_factordb_retries_submission_errors(
    factordb: FactorDB, config: Config, http_request: Mock, sleep: Mock, retry_wait: Mock
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
    assert retry_wait.call_args_list == [call(1.0), call(2.0)]
    sleep.assert_not_called()
    assert not config.pending_submissions_path.exists()


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
    sleep.assert_not_called()


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
    assert "FactorDB accepted 3 factorizations, crediting 2 factors\n" in messages
    assert "FactorDB credited at least 2 factors to your account\n" in messages
    sleep.assert_not_called()


def make_submission(*prime_factors: int, lifetime: float = 3600.0) -> Submission:
    """Build the submission of a completely factored number from its prime factors.

    Returns:
        Submission: The submission, which expires after the given lifetime.
    """
    return Submission(
        n=math.prod(prime_factors),
        prime_factors=prime_factors,
        composite_factors=(),
        expires_at=time.time() + lifetime,
    )


def journaled(config: Config) -> list[int]:
    """Read the composites whose submissions are still pending on disk.

    Returns:
        list[int]: The composites, in the order they were queued.
    """
    return [x.n for x in SubmissionJournal(config.pending_submissions_path, config.backend).load()]


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


def test_factordb_submits_a_batch_in_one_request(factordb: FactorDB, http_request: Mock) -> None:
    """Test that several numbers are reported in a single batch request, one report_factors call each."""
    http_request.return_value = rpc_batch({"status": "FF"}, {"status": "FF"}, {"status": "FF"})
    submissions = [make_submission(3, 5), make_submission(3, 7), make_submission(5, 7)]

    assert factordb._submit_numbers(submissions) == [SubmitOutcome.ACCEPTED] * len(submissions)

    http_request.assert_called_once()
    assert [params["target"]["expr"] for _, params in rpc_calls(http_request)] == ["15", "21", "35"]


def test_factordb_reports_the_outcome_of_each_call_in_a_batch(factordb: FactorDB, http_request: Mock) -> None:
    """Test that only a failed call is left for another attempt, and a number with nothing to report is not sent."""
    http_request.return_value = rpc_batch(
        {"status": "FF"},
        RpcError(*REJECTED_FACTORS_ERROR),
        RpcError(-32000, "database busy"),
        RpcError(-32602, "invalid params"),
    )
    submissions = [
        make_submission(3, 5),
        make_submission(7),
        make_submission(3, 7),
        make_submission(5, 7),
        make_submission(7, 11),
    ]

    assert factordb._submit_numbers(submissions) == [
        SubmitOutcome.ACCEPTED,
        SubmitOutcome.REJECTED,
        SubmitOutcome.REJECTED,
        SubmitOutcome.FAILED,
        SubmitOutcome.REJECTED,
    ]

    http_request.assert_called_once()
    assert [params["target"]["expr"] for _, params in rpc_calls(http_request)] == ["15", "21", "35", "77"]


@pytest.mark.parametrize(
    "error",
    [PermanentHttpError("HTTP 400"), rpc_error(-32600, "invalid request").json()[0] | {"id": None}],
    ids=["http", "rpc"],
)
def test_factordb_keeps_a_batch_that_is_refused_for_the_next_run(
    factordb: FactorDB, http_request: Mock, error: object
) -> None:
    """Test that a batch refused as a whole is not taken as a verdict on its factors, and is not sent again this run."""
    http_request.side_effect = [
        error if isinstance(error, Exception) else Mock(spec=requests.Response, json=Mock(return_value=error))
    ]
    submissions = [make_submission(3, 5), make_submission(7), make_submission(3, 7)]

    # The prime has nothing to report, so it was never part of the request.
    assert factordb._submit_numbers(submissions) == [
        SubmitOutcome.DEFERRED,
        SubmitOutcome.REJECTED,
        SubmitOutcome.DEFERRED,
    ]


def test_base_keeps_a_newer_result_when_an_older_one_is_resolved(config: Config, sleep: Mock) -> None:
    """Test that a carried-over result being accepted leaves a newer result for the same composite pending."""
    started = threading.Event()
    release = threading.Event()

    class SlowBackend(FakeBackend):
        @override
        def _submit_number(self, submission: Submission) -> SubmitOutcome:
            started.set()
            release.wait(timeout=5.0)

            # Only the carried-over result gets an answer before the run ends.
            if len(self.submitted) > 0:
                raise Interrupted

            return super()._submit_number(submission)

    journal = SubmissionJournal(config.pending_submissions_path, config.backend)
    journal.add([Submission(n=15015, prime_factors=(3, 5), composite_factors=(1001,), expires_at=time.time() + 600.0)])

    backend = SlowBackend(config, FactoringStats(config.stats_path, read_only=True), ())
    assert started.wait(timeout=5.0)
    backend.submit(make_factored(config, backend, [3, 5, 7, 11, 13]))
    release.set()
    backend.close()

    assert backend.get_successful_submission_count() == 1
    assert [x.prime_factors for x in journal.load()] == [(3, 5, 7, 11, 13)]
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [Mock(spec=requests.Response, json=Mock(return_value=[])), requests.RequestException("unreachable")],
    ids=["malformed", "request-error"],
)
def test_factordb_leaves_a_whole_batch_for_another_attempt(
    factordb: FactorDB, http_request: Mock, response: object
) -> None:
    """Test that every call is left for another attempt when the batch as a whole fails."""
    http_request.side_effect = [response]
    submissions = [make_submission(3, 5), make_submission(3, 7)]

    assert factordb._submit_numbers(submissions) == [SubmitOutcome.FAILED] * len(submissions)


def test_base_worker_submits_a_backlog_in_batches(config: Config, sleep: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that numbers queued while a submission is in progress are submitted together, up to the batch size."""
    monkeypatch.setattr("factortool.backend.SUBMIT_MAX_HOLD", 0.0)
    started = threading.Event()
    release = threading.Event()
    batches: list[list[int]] = []

    class BatchingBackend(FakeBackend):
        submit_batch_size = 2

        @override
        def _submit_numbers(self, submissions: Sequence[Submission]) -> list[SubmitOutcome]:
            started.set()
            release.wait(timeout=5.0)
            batches.append([submission.n for submission in submissions])
            return [SubmitOutcome.ACCEPTED] * len(submissions)

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
    sleep.assert_not_called()


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
    sleep.assert_not_called()


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
        wait=factordb._fetch_wait,
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
    [
        requests.RequestException("unreachable"),
        rpc_response({"login": "FactorFinder"}),
        rpc_error(-32000, "database busy"),
    ],
    ids=["request-error", "malformed", "rpc-error"],
)
def test_factordb_retries_until_the_token_is_verified(
    config: Config, http_request: Mock, wait: Mock, response: object
) -> None:
    """Test that a token FactorDB can't verify for now is tried again with backoff, as a fetch would be."""
    http_request.side_effect = [response, response, rpc_response({"found": True, "login": "FactorFinder"})]
    backend = FactorDB(config.model_copy(update={"factordb_api_token": "secret"}), FactoringStats(config.stats_path))

    try:
        assert backend._signed_in
    finally:
        backend.close()

    assert wait.call_args_list == [call(1.0), call(2.0)]


def test_factordb_stops_verifying_the_token_on_an_interrupt(config: Config, http_request: Mock, wait: Mock) -> None:
    """Test that an interrupt while waiting to verify the token ends startup, rather than submitting without credit."""
    wait.return_value = True
    http_request.side_effect = [requests.RequestException("unreachable")]

    with pytest.raises(Interrupted):
        FactorDB(config.model_copy(update={"factordb_api_token": "secret"}), FactoringStats(config.stats_path))

    http_request.assert_called_once()


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


class RecordingBatchesBackend(FakeBackend):
    """A fake backend that records each batch it is asked to submit."""

    submit_batch_size = 3

    def __init__(self, config: Config) -> None:
        """Initialize the backend with no batches submitted."""
        self.batches: list[list[int]] = []
        self.batch_sent = threading.Event()
        super().__init__(config, FactoringStats(config.stats_path, read_only=True), ())

    @override
    def _submit_numbers(self, submissions: Sequence[Submission]) -> list[SubmitOutcome]:
        self.batches.append([submission.n for submission in submissions])
        self.batch_sent.set()
        return [SubmitOutcome.ACCEPTED] * len(submissions)


def test_base_holds_a_partial_batch_until_it_is_full(config: Config) -> None:
    """Test that results arriving close together are held until they fill a batch, which is then sent at once."""
    backend = RecordingBatchesBackend(config)
    numbers = make_factored(config, backend, [3, 5], [3, 7], [5, 7], [7, 11])

    try:
        backend.submit(numbers[:1])
        assert not backend.batch_sent.wait(timeout=0.5)
        backend.submit(numbers[1:])
        assert backend.batch_sent.wait(timeout=5.0)
        assert backend.batches == [[15, 21, 35]]
    finally:
        backend.close()

    assert backend.batches == [[15, 21, 35], [77]]


def test_base_sends_a_partial_batch_once_its_hold_ends(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a partial batch is sent without waiting to close once its oldest result has been held long enough."""
    monkeypatch.setattr("factortool.backend.SUBMIT_MAX_HOLD", 0.5)
    backend = RecordingBatchesBackend(config)

    try:
        start = time.monotonic()
        backend.submit(make_factored(config, backend, [3, 5]))
        assert backend.batch_sent.wait(timeout=5.0)
        assert time.monotonic() - start >= 0.5  # ruff: ignore[magic-value-comparison]
    finally:
        backend.close()

    assert backend.batches == [[15]]


def test_base_sends_an_isolated_result_at_once(config: Config) -> None:
    """Test that a result arriving long after the previous one is sent without waiting for others to join it."""
    backend = RecordingBatchesBackend(config)
    backend._last_queued = time.monotonic() - 60.0

    try:
        backend.submit(make_factored(config, backend, [3, 5]))
        assert backend.batch_sent.wait(timeout=5.0)
    finally:
        backend.close()

    assert backend.batches == [[15]]


def test_base_does_not_hold_a_result_close_to_expiry(config: Config) -> None:
    """Test that a result whose assignment expires within the hold is sent at once rather than risk expiring."""
    backend = RecordingBatchesBackend(config)
    (number,) = make_factored(config, backend, [3, 5])
    number.expires_at = time.time() + 20.0

    try:
        backend.submit([number])
        assert backend.batch_sent.wait(timeout=5.0)
    finally:
        backend.close()

    assert backend.batches == [[15]]


def test_base_close_sends_a_held_batch_at_once(config: Config) -> None:
    """Test that closing sends a held partial batch without waiting for its hold to end."""
    backend = RecordingBatchesBackend(config)
    backend.submit(make_factored(config, backend, [3, 5], [3, 7]))

    start = time.monotonic()
    backend.close()

    assert time.monotonic() - start < 5.0  # ruff: ignore[magic-value-comparison]
    assert backend.batches == [[15, 21]]


def test_base_sends_what_a_previous_run_left_unsent_without_holding_it(config: Config) -> None:
    """Test that results carried over from a previous run are sent at once, even as a partial batch."""
    SubmissionJournal(config.pending_submissions_path, config.backend).add([make_submission(3, 5)])
    backend = RecordingBatchesBackend(config)

    try:
        assert backend.batch_sent.wait(timeout=5.0)
    finally:
        backend.close()

    assert backend.batches == [[15]]


def test_base_close_does_not_count_request_spacing_as_a_stall(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that waiting out the request interval while closing doesn't give up on a service that is answering."""
    monkeypatch.setattr("factortool.backend.FLUSH_TIMEOUT", 0.3)
    response = requests.Response()
    response.status_code = 200

    class SpacedBackend(FakeBackend):
        @override
        def _submit_number(self, submission: Submission) -> SubmitOutcome:
            self._service_request("POST", "https://example.invalid/", timeout=1.0, wait=self._submission_wait)
            return super()._submit_number(submission)

    backend = SpacedBackend(config, FactoringStats(config.stats_path, read_only=True), ())
    monkeypatch.setattr(backend._http_client.session, "request", Mock(return_value=response))

    # FakeBackend spaces its requests a second apart, well beyond the shortened flush timeout.
    backend.submit(make_factored(config, backend, [3, 5], [3, 7]))
    backend.close()

    assert backend.submitted == [15, 21]
    assert not config.pending_submissions_path.exists()


class AssigningBackend(FakeBackend):
    """A fake backend whose service reserves the composites it hands out."""

    assigns_work = True
    assignment_lifetime = 3600.0


# Products of two primes too large for trial factoring to find.
HARD_COMPOSITES = (1000003 * 1000033, 1000037 * 1000039)


def make_assigning_backend(config: Config, responses: Iterable[str | Exception] = ()) -> AssigningBackend:
    """Create a fake backend that assigns work. Callers are responsible for closing it.

    Returns:
        AssigningBackend: The fake backend.
    """
    return AssigningBackend(config, FactoringStats(config.stats_path, read_only=True), responses)


def retained(config: Config) -> list[Assignment]:
    """Read the assignments in the state file as they stand, without discarding any that have expired.

    Returns:
        list[Assignment]: The assignments, smallest composite first.
    """
    return AssignmentState.model_validate_json(config.assignment_state_path.read_text(encoding="utf-8")).assignments


def retain(config: Config, *assignments: Assignment) -> None:
    """Leave assignments in the state file, as a previous run would have."""
    store = AssignmentStore(config.assignment_state_path, config.backend)
    store.record(assignments)


def test_assigned_work_is_recorded_as_soon_as_it_is_fetched(config: Config) -> None:
    """Test that fetched composites are recorded with their expiry before any work is done on them."""
    backend = make_assigning_backend(config, ["15\n21\n"])
    start = time.time()

    try:
        numbers = backend.fetch(FetchCriteria(count=2, min_digits=1))

        assert {x.n for x in numbers} == {15, 21}
        assert [(x.n, x.factors) for x in retained(config)] == [(15, (15,)), (21, (21,))]
        assert all(x.expires_at == pytest.approx(start + backend.assignment_lifetime, abs=5.0) for x in numbers)
        assert all(x.expires_at == y.expires_at for x, y in zip(sorted(numbers), retained(config), strict=True))
    finally:
        backend.close()


def test_work_that_is_not_assigned_is_not_recorded(config: Config) -> None:
    """Test that a backend that does not assign work leaves no assignment state, and resumes nothing."""
    backend = make_fake_backend(config, ["15\n"])

    try:
        (number,) = backend.fetch(FetchCriteria(count=1, min_digits=1))
        number.factor_tf()

        assert number.expires_at is None
        assert backend.resume() == set()
    finally:
        backend.close()

    assert not config.assignment_state_path.exists()


def test_progress_on_an_assigned_number_is_recorded_as_it_is_made(config: Config) -> None:
    """Test that factors found for an assigned number are recorded at once, before the run ends."""
    backend = make_assigning_backend(config, [f"{2 * HARD_COMPOSITES[0]}\n"])

    try:
        (number,) = backend.fetch(FetchCriteria(count=1, min_digits=1))
        number.factor_tf()

        assert [x.factors for x in retained(config)] == [(2, HARD_COMPOSITES[0])]
    finally:
        backend.close()


def test_a_finished_number_is_released_once_its_result_is_journaled(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    """Test that a factored number leaves the assignment state only after its result has entered the journal."""
    backend = make_assigning_backend(config, ["15\n21\n"])
    journaled_at_release: list[list[int]] = []
    release = AssignmentStore.release

    def recording_release(store: AssignmentStore, composites: Iterable[int]) -> None:
        journaled_at_release.append(journaled(config))
        release(store, composites)

    monkeypatch.setattr(AssignmentStore, "release", recording_release)

    try:
        numbers = sorted(backend.fetch(FetchCriteria(count=2, min_digits=1)))
        numbers[0].factor_tf()

        assert [x.n for x in retained(config)] == [21]
    finally:
        backend.close()

    assert journaled_at_release == [[15]]
    assert backend.submitted == [15]


def test_a_composite_assigned_again_keeps_its_progress_and_takes_the_new_expiry(config: Config) -> None:
    """Test that a held composite handed out again is returned as the same number, with the later expiry."""
    n = 2 * HARD_COMPOSITES[0]
    retain(config, Assignment(n=n, expires_at=time.time() + 60.0, factors=(2, HARD_COMPOSITES[0])))
    backend = make_assigning_backend(config, [f"{n}\n"])
    start = time.time()

    try:
        (resumed,) = backend.resume()
        (fetched,) = backend.fetch(FetchCriteria(count=1, min_digits=1))

        assert fetched is resumed
        assert fetched.prime_factors == [2]
        assert fetched.expires_at == pytest.approx(start + backend.assignment_lifetime, abs=5.0)
        assert [(x.factors, x.expires_at) for x in retained(config)] == [((2, HARD_COMPOSITES[0]), fetched.expires_at)]
    finally:
        backend.close()


def test_resumed_work_continues_from_the_factors_already_found(config: Config) -> None:
    """Test that a retained number is rebuilt with its factors and its original expiry."""
    expires_at = time.time() + 1800.0
    retain(config, Assignment(n=15015, expires_at=expires_at, factors=(3, 5, 1001)))
    backend = make_assigning_backend(config)

    try:
        (number,) = backend.resume()

        assert (number.prime_factors, number.composite_factors) == ([3, 5], [1001])
        assert number.expires_at == expires_at
    finally:
        backend.close()


def test_resumed_work_with_inconsistent_factors_is_recorded_as_rebuilt(config: Config) -> None:
    """Test that factors rejected when a retained number is rebuilt are dropped from the state as well."""
    retain(config, Assignment(n=15015, expires_at=time.time() + 1800.0, factors=(3, 5, 7)))
    backend = make_assigning_backend(config)

    try:
        backend.resume()

        assert [x.factors for x in retained(config)] == [(15015,)]
    finally:
        backend.close()


def test_work_whose_result_is_awaiting_resubmission_is_not_resumed(config: Config) -> None:
    """Test that a run killed between journaling a result and releasing its assignment doesn't factor it again."""
    expires_at = time.time() + 1800.0
    retain(config, Assignment(n=15, expires_at=expires_at), Assignment(n=21, expires_at=expires_at))
    SubmissionJournal(config.pending_submissions_path, config.backend).add([make_submission(3, 5)])
    backend = make_assigning_backend(config)

    try:
        assert [x.n for x in backend.resume()] == [21]
        assert [x.n for x in retained(config)] == [21]
    finally:
        backend.close()

    assert backend.submitted == [15]


def test_closing_retains_unfinished_work_and_drops_what_has_expired(config: Config) -> None:
    """Test that the state left for the next run holds only the unfinished assignments that are still held."""
    backend = make_assigning_backend(config, ["15\n21\n"])

    try:
        held, expiring = sorted(backend.fetch(FetchCriteria(count=2, min_digits=1)))
        expiring.expires_at = time.time() - 1.0
        backend.checkpoint(expiring)
    finally:
        backend.close()

    assert [x.n for x in retained(config)] == [held.n]


def test_progress_is_not_recorded_for_a_number_the_backend_does_not_hold(config: Config) -> None:
    """Test that a number that was never assigned, or has been released, is not added back to the state."""
    backend = make_assigning_backend(config)
    number = Number(15, config, FactoringStats(config.stats_path, read_only=True), backend)
    number.expires_at = time.time() + 600.0

    try:
        backend.checkpoint(number)
    finally:
        backend.close()

    assert retained(config) == []


def test_closing_reports_a_failure_to_retain_unfinished_work(monkeypatch: pytest.MonkeyPatch, config: Config) -> None:
    """Test that losing track of unfinished assignments is raised by close, after the journal is compacted."""
    backend = make_assigning_backend(config)
    compact = Mock()
    monkeypatch.setattr(backend._journal, "compact", compact)
    monkeypatch.setattr(AssignmentStore, "compact", Mock(side_effect=OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        backend.close()

    compact.assert_called_once_with()


def test_factors_found_before_a_method_is_abandoned_are_recorded(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    """Test that an abort partway through a method still records the factors it had already found."""
    n = math.prod(HARD_COMPOSITES)
    retain(config, Assignment(n=n, expires_at=time.time() + 1800.0, factors=HARD_COMPOSITES))
    backend = make_assigning_backend(config)
    factor_yafu = Mock(side_effect=[[1000003, 1000033], Interrupted()])
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)

    try:
        (number,) = backend.resume()

        with pytest.raises(Interrupted):
            number.factor_rho()

        assert [sorted(x.factors) for x in retained(config)] == [[1000003, 1000033, HARD_COMPOSITES[1]]]
    finally:
        backend.close()


class UnreachableAssigningBackend(UnreachableBackend):
    """A fake backend that assigns work and whose service never answers a submission."""

    assigns_work = True
    assignment_lifetime = 3600.0


def close_abandoning_submissions(backend: BaseBackend) -> None:
    """Close a backend, interrupting it shortly after it starts flushing its submissions."""
    timer = threading.Timer(0.2, setattr, (backend._interrupts, "_level", 1))
    timer.start()

    try:
        backend.close()
    finally:
        timer.cancel()


def test_a_result_the_journal_fails_to_record_keeps_its_assignment(config: Config) -> None:
    """Test that a result that never reaches the journal is retained with its assignment, and submitted when resumed."""
    config.pending_submissions_path.mkdir()
    backend = UnreachableAssigningBackend(config, FactoringStats(config.stats_path, read_only=True), ["15\n"])

    try:
        (number,) = backend.fetch(FetchCriteria(count=1, min_digits=1))
        number.factor_tf()

        assert [(x.n, x.factors) for x in retained(config)] == [(15, (3, 5))]
    finally:
        with pytest.raises((IsADirectoryError, PermissionError)):
            close_abandoning_submissions(backend)

    assert [(x.n, x.factors) for x in retained(config)] == [(15, (3, 5))]

    config.pending_submissions_path.rmdir()
    resuming = make_assigning_backend(config)

    try:
        assert resuming.resume() == set()
    finally:
        resuming.close()

    assert resuming.submitted == [15]
    assert retained(config) == []


def test_a_result_the_journal_fails_to_record_is_released_once_submitted(config: Config) -> None:
    """Test that the assignment retained for an unjournaled result is released once the result is resolved."""
    config.pending_submissions_path.mkdir()
    backend = make_assigning_backend(config, ["15\n"])

    try:
        (number,) = backend.fetch(FetchCriteria(count=1, min_digits=1))
        number.factor_tf()
    finally:
        with pytest.raises((IsADirectoryError, PermissionError)):
            backend.close()

    assert backend.submitted == [15]
    assert retained(config) == []


def test_a_result_the_journal_fails_to_record_is_released_once_the_journal_is_compacted(config: Config) -> None:
    """Test that the assignment retained for an unjournaled result is released once compaction writes the journal."""
    config.pending_submissions_path.mkdir()
    backend = UnreachableAssigningBackend(config, FactoringStats(config.stats_path, read_only=True), ["15\n"])

    try:
        (number,) = backend.fetch(FetchCriteria(count=1, min_digits=1))
        number.factor_tf()
        config.pending_submissions_path.rmdir()
    finally:
        close_abandoning_submissions(backend)

    assert journaled(config) == [15]
    assert retained(config) == []


def test_closing_before_resuming_keeps_the_saved_assignments(config: Config) -> None:
    """Test that a run that ends before resuming its saved work leaves that work for the next run."""
    expires_at = time.time() + 1800.0
    retain(config, Assignment(n=15015, expires_at=expires_at, factors=(3, 5, 1001)))

    backend = make_assigning_backend(config)
    backend.close()

    assert [(x.n, x.factors) for x in retained(config)] == [(15015, (3, 5, 1001))]


def test_progress_is_recorded_before_the_next_cofactor_is_factored(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    """Test that a split is recorded before the method moves on to the next composite factor."""
    n = math.prod(HARD_COMPOSITES)
    retain(config, Assignment(n=n, expires_at=time.time() + 1800.0, factors=HARD_COMPOSITES))
    backend = make_assigning_backend(config)
    seen: list[list[int]] = []

    def factor(composite: int, *_args: object) -> list[int]:
        seen.append(sorted(x for assignment in retained(config) for x in assignment.factors))
        return [1000003, 1000033] if composite == HARD_COMPOSITES[0] else [composite]

    monkeypatch.setattr("factortool.number.factor_yafu", factor, raising=True)

    try:
        (number,) = backend.resume()
        number.factor_rho()
    finally:
        backend.close()

    assert seen == [sorted(HARD_COMPOSITES), [1000003, 1000033, HARD_COMPOSITES[1]]]


def test_progress_is_recorded_before_a_composite_factor_is_factored_further(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    """Test that a final method's split is recorded before its composite factors are factored in turn."""
    n = math.prod(HARD_COMPOSITES)
    retain(config, Assignment(n=n, expires_at=time.time() + 1800.0))
    backend = make_assigning_backend(config)
    seen: list[list[int]] = []

    splits = {n: list(HARD_COMPOSITES), HARD_COMPOSITES[0]: [1000003, 1000033], HARD_COMPOSITES[1]: [1000037, 1000039]}

    def run_final(_number: Number, composite: int, _method: str) -> list[int]:
        seen.append(sorted(x for assignment in retained(config) for x in assignment.factors))
        return splits[composite]

    monkeypatch.setattr(Number, "_run_final", run_final)

    try:
        (number,) = backend.resume()
        number.factor_final()
    finally:
        backend.close()

    assert seen == [[n], sorted(HARD_COMPOSITES), [1000003, 1000033, HARD_COMPOSITES[1]]]
    assert backend.submitted == [n]
