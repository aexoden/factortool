# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the CLI of factortool."""

from __future__ import annotations

import json
import signal
import sys
import time

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from collections.abc import Collection

import pytest
import requests

from factortool.assignments import AssignmentStore
from factortool.backend import FetchCriteria
from factortool.cli.main import (
    CLEANUP_FAILED_EXIT_STATUS,
    Arguments,
    Cleanup,
    acquire_numbers,
    get_time_limit,
    main,
    preserve_unfinished,
    start_backend,
    validate_arguments,
    warn_if_assignments_may_expire,
)
from factortool.engine import ExitStatus, FactorEngine
from factortool.http import PermanentHttpError
from factortool.interrupt import EXIT_STATUS, Interrupted, InterruptState
from factortool.number import Number
from factortool.stats import FactoringStats
from factortool.tools import CadoNfsError, ToolError, YafuError

from .helpers import make_config

# The exit status of a run ended by a permanent HTTP error in the backend.
PERMANENT_HTTP_ERROR_EXIT_STATUS = 6

CRITERIA = FetchCriteria(count=3, min_digits=1, max_digits=100)

# Composite, so Number does not treat them as already factored.
CARRIED_OVER = [100, 102]
FETCHED = [104, 106, 108]


def write_config(tmp_path: Path, **overrides: object) -> None:
    """Write a configuration file whose tools are present, with any executable standing in for YAFU."""
    config = {"backend": "factordb", "max_threads": 1, "yafu_path": sys.executable, **overrides}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


class FakeBackend:
    """A backend that records requests, assigning work unless told otherwise."""

    assignment_lifetime = 3600.0

    def __init__(self, *, assigns_work: bool = True) -> None:
        """Initialize the fake backend."""
        self.assigns_work = assigns_work
        self.requested: list[int] = []
        self.submitted: list[int] = []

    def fetch(self, criteria: FetchCriteria) -> set[Number]:
        """Record the request and hand back the canned composites.

        Returns:
            set[Number]: The composites this backend always returns.
        """
        self.requested.append(criteria.count)
        config = make_config()
        stats = FactoringStats(Path("stats.json"), read_only=True)

        return {Number(n, config, stats, None) for n in FETCHED[: criteria.count]}

    def submit(self, numbers: Collection[Number]) -> None:
        """Record a factorization instead of reporting it anywhere."""
        self.submitted.extend(x.n for x in numbers)

    def get_successful_submission_count(self) -> int:
        """Report how many factorizations were handed over.

        Returns:
            int: The number of submissions recorded.
        """
        return len(self.submitted)

    def close(self) -> None:
        """Close the fake backend, doing nothing."""


def test_a_normal_run_tops_the_batch_back_up(tmp_path: Path) -> None:
    """Test that retained work counts toward the batch and the rest is fetched."""
    backend = FakeBackend()
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", backend.assignment_lifetime)
    store.note_assigned(CARRIED_OVER)
    store.save(CARRIED_OVER)

    numbers = acquire_numbers(backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA)

    assert backend.requested == [1]
    assert sorted(x.n for x in numbers) == sorted([*CARRIED_OVER, FETCHED[0]])


def test_acquired_numbers_carry_their_assignment_expiry(tmp_path: Path) -> None:
    """Test that both retained and fetched numbers know when their assignment expires."""
    backend = FakeBackend()
    state_path = tmp_path / "assignments.json"
    first = AssignmentStore(state_path, "mersenne_ca", backend.assignment_lifetime)
    first.note_assigned(CARRIED_OVER)
    first.save(CARRIED_OVER)

    store = AssignmentStore(state_path, "mersenne_ca", backend.assignment_lifetime)
    numbers = acquire_numbers(backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA)

    assert all(x.expires_at is not None and x.expires_at == store.expires_at(x.n) for x in numbers)


def test_no_new_work_works_only_what_is_already_assigned(tmp_path: Path) -> None:
    """Test that the no new work option only uses already assigned work."""
    backend = FakeBackend()
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", backend.assignment_lifetime)
    store.note_assigned(CARRIED_OVER)
    store.save(CARRIED_OVER)

    numbers = acquire_numbers(
        backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA, fetch=False
    )

    assert backend.requested == []
    assert sorted(x.n for x in numbers) == CARRIED_OVER


def test_no_new_work_with_nothing_carried_over_finds_no_work(tmp_path: Path) -> None:
    """Test that the no new work option finds no work when nothing is retained."""
    backend = FakeBackend()
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", backend.assignment_lifetime)

    numbers = acquire_numbers(
        backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA, fetch=False
    )

    assert backend.requested == []
    assert numbers == set()


def test_an_early_exit_reports_partial_progress_without_an_assignment(tmp_path: Path) -> None:
    """Test that an early exit reports partial progress for a backend that does not assign work."""
    backend = FakeBackend(assigns_work=False)
    store = AssignmentStore(tmp_path / "assignments.json", "factordb", backend.assignment_lifetime)
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)

    partially_factored = Number(200, make_config(), stats, backend)
    partially_factored.prime_factors = [2, 2]
    partially_factored.composite_factors = [50]
    untouched = Number(300, make_config(), stats, backend)

    preserve_unfinished(backend, store, [partially_factored, untouched])

    assert backend.submitted == [200]
    assert not (tmp_path / "assignments.json").exists()


def test_no_new_work_is_rejected_by_a_backend_that_assigns_nothing() -> None:
    """Test that the no new work option is refused on FactorDB, where it could only ever find no work."""
    args = Arguments().parse_args(["--no_new_work"])

    with pytest.raises(SystemExit) as error:
        validate_arguments(args, "factordb")

    assert error.value.code == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["--min_digits", "0"],
        ["--min_digits", "-3"],
        ["--batch_size", "-1"],
        ["--skip_count", "-1"],
        ["--target_duration", "0"],
        ["--target_duration", "-600"],
        ["--target_duration", "nan"],
        ["--target_duration", "inf"],
        ["--max_digits", "-1"],
        ["--min_digits", "60", "--max_digits", "50"],
    ],
)
def test_out_of_range_arguments_are_rejected(arguments: list[str]) -> None:
    """Test that arguments no run could honor are refused up front."""
    args = Arguments().parse_args(arguments)

    with pytest.raises(SystemExit) as error:
        validate_arguments(args, "factordb")

    assert error.value.code == 1


@pytest.mark.parametrize(
    "arguments",
    [[], ["--min_digits", "1", "--max_digits", "1"], ["--batch_size", "5", "--skip_count", "0"]],
)
def test_arguments_at_the_edge_of_their_range_are_accepted(arguments: list[str]) -> None:
    """Test that the smallest usable values are not refused."""
    validate_arguments(Arguments().parse_args(arguments), "factordb")


@pytest.mark.parametrize("arguments", [["--min_digits", "many"], ["--no_such_option"]])
def test_unparseable_arguments_exit_as_a_usage_error(arguments: list[str]) -> None:
    """Test that arguments the parser rejects don't exit with the status that means interrupted."""
    with pytest.raises(SystemExit) as error:
        Arguments().parse_args(arguments)

    assert error.value.code == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"yafu_path": "missing-yafu"},
        {"yafu_ini_path": "missing.ini"},
        {"cado_nfs_path": "missing-cado-nfs.py", "use_nfs_cado": True},
    ],
)
def test_missing_tools_end_the_run_before_the_backend_is_started(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overrides: dict[str, object]
) -> None:
    """Test that a tool that isn't there is reported before work is fetched for it."""
    write_config(tmp_path, **overrides)
    create_backend = Mock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool"])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState.install", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", create_backend)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 1
    create_backend.assert_not_called()


@pytest.mark.parametrize(("error", "exit_status"), [(YafuError("YAFU failed"), 5), (CadoNfsError("CADO failed"), 4)])
def test_a_tool_failure_exits_with_the_tool_status_after_cleaning_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: ToolError, exit_status: int
) -> None:
    """Test that a tool failure during the run still saves state before exiting with that tool's status."""
    write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", "--batch_size", "3"])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState.install", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(return_value=FakeBackend(assigns_work=False)))
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", Mock(side_effect=error), raising=True)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == exit_status
    assert (tmp_path / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1


def test_a_termination_signal_still_saves_state_before_exiting_as_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a run ended by SIGTERM saves its state and results, and exits with the interrupted status."""
    write_config(tmp_path)
    backend = FakeBackend(assigns_work=False)
    backend.close = Mock()  # type: ignore[method-assign]
    interrupts = InterruptState()

    def run(_self: FactorEngine, _numbers: Collection[Number], _time_limit: float | None) -> ExitStatus:
        signal.raise_signal(signal.SIGTERM)
        time.sleep(0.5)

        return ExitStatus.SUCCESS

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", "--batch_size", "3"])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(return_value=backend))
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", run, raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState", lambda: interrupts, raising=True)

    try:
        with pytest.raises(SystemExit) as raised:
            main()
    finally:
        interrupts.uninstall()

    assert raised.value.code == EXIT_STATUS
    assert (tmp_path / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1
    backend.close.assert_called_once()


def test_an_automatic_batch_is_limited_to_twice_its_target_duration() -> None:
    """Test that a run with an automatic batch size may take up to twice the target duration."""
    assert get_time_limit(Arguments().parse_args(["--target_duration", "300"])) == pytest.approx(600.0)


def test_an_explicit_batch_size_has_no_time_limit() -> None:
    """Test that the target duration imposes no time limit when the batch size was chosen by the user."""
    assert get_time_limit(Arguments().parse_args(["--batch_size", "3", "--target_duration", "300"])) is None


@pytest.mark.parametrize(("time_limit", "warned"), [(3600.0, True), (1200.0, False), (None, False)])
def test_a_time_limit_that_may_outlast_assignments_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, time_limit: float | None, *, warned: bool
) -> None:
    """Test that the expiry warning depends on the time limit, and is skipped when there is none."""
    warning = Mock()
    monkeypatch.setattr("factortool.cli.main.logger.warning", warning, raising=True)

    warn_if_assignments_may_expire(FakeBackend(), time_limit)

    assert warning.called == warned


@pytest.mark.parametrize(("arguments", "time_limit"), [([], 1200.0), (["--batch_size", "3"], None)])
def test_the_run_is_given_the_time_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arguments: list[str], time_limit: float | None
) -> None:
    """Test that the time limit reaches the engine with the run rather than when the engine is created."""
    write_config(tmp_path)
    run = Mock(return_value=ExitStatus.SUCCESS)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", *arguments])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState.install", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(return_value=FakeBackend(assigns_work=False)))
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", run, raising=True)

    main()

    assert run.call_args.args[1] == time_limit


def test_a_failed_cleanup_step_does_not_skip_the_rest() -> None:
    """Test that every cleanup step runs, whether an earlier one failed as expected or unexpectedly."""
    cleanup = Cleanup()
    last = Mock()

    cleanup.run("write a file", Mock(side_effect=OSError("disk full")))
    cleanup.run("do something else", Mock(side_effect=RuntimeError("bug")))
    cleanup.run("finish", last, 1, key="value")

    last.assert_called_once_with(1, key="value")
    assert cleanup.failed


def test_a_cleanup_without_failures_is_not_failed() -> None:
    """Test that steps that succeed leave the cleanup unfailed."""
    cleanup = Cleanup()

    cleanup.run("finish", Mock())

    assert not cleanup.failed


def run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run: Mock,
    *,
    backend: FakeBackend | None = None,
    close: Mock | None = None,
) -> tuple[int, Mock]:
    """Run main against a fake backend and a stand-in for the engine.

    Returns:
        tuple[int, Mock]: The exit status and the mock standing in for closing the backend.
    """
    write_config(tmp_path)
    backend = FakeBackend(assigns_work=False) if backend is None else backend
    close = Mock() if close is None else close
    backend.close = close  # type: ignore[method-assign]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool"])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState.install", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(return_value=backend))
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", run, raising=True)

    try:
        main()
    except SystemExit as e:
        status = e.code
    else:
        status = 0

    assert isinstance(status, int)

    return status, close


@pytest.mark.parametrize("failing", ["BatchController.record_batch", "write_results", "report_summary"])
def test_a_failed_cleanup_step_still_saves_the_rest_and_closes_the_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failing: str
) -> None:
    """Test that a cleanup failure leaves the later steps to run, and is reported by the exit status."""
    monkeypatch.setattr(f"factortool.cli.main.{failing}", Mock(side_effect=OSError("disk full")), raising=True)

    status, close = run_main(monkeypatch, tmp_path, Mock(return_value=ExitStatus.SUCCESS))

    assert status == CLEANUP_FAILED_EXIT_STATUS
    assert (tmp_path / "stats.json").exists()
    close.assert_called_once_with()


def test_a_failure_to_save_assignments_is_reported_by_the_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that losing track of unfinished assignments counts as a cleanup failure, without skipping later steps."""
    monkeypatch.setattr("factortool.cli.main.AssignmentStore.save", Mock(side_effect=OSError("disk full")))

    status, close = run_main(monkeypatch, tmp_path, Mock(return_value=ExitStatus.SUCCESS), backend=FakeBackend())

    assert status == CLEANUP_FAILED_EXIT_STATUS
    assert len(list((tmp_path / "results").iterdir())) == 1
    close.assert_called_once_with()


@pytest.mark.parametrize(
    ("run", "exit_status"),
    [
        (Mock(side_effect=YafuError("YAFU failed")), 5),
        (Mock(return_value=ExitStatus.INTERRUPTED), CLEANUP_FAILED_EXIT_STATUS),
        (Mock(return_value=ExitStatus.TIME_LIMIT_EXCEEDED), CLEANUP_FAILED_EXIT_STATUS),
    ],
)
def test_a_cleanup_failure_only_gives_way_to_another_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: Mock, exit_status: int
) -> None:
    """Test that a tool failure keeps its own exit status, while an interrupt or time limit yields to the failure."""
    monkeypatch.setattr("factortool.cli.main.write_results", Mock(side_effect=OSError("disk full")), raising=True)

    status, close = run_main(monkeypatch, tmp_path, run)

    assert status == exit_status
    close.assert_called_once_with()


def test_a_failure_to_close_the_backend_is_reported_by_the_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a backend that cannot shut down cleanly is reported once everything else has been saved."""
    close = Mock(side_effect=RuntimeError("bug"))

    status, _ = run_main(monkeypatch, tmp_path, Mock(return_value=ExitStatus.SUCCESS), close=close)

    assert status == CLEANUP_FAILED_EXIT_STATUS
    assert (tmp_path / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1


def test_an_unexpected_engine_failure_still_cleans_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a failure other than a tool error is raised only after the state is saved and the backend closed."""
    close = Mock()

    with pytest.raises(RuntimeError, match="bug"):
        run_main(monkeypatch, tmp_path, Mock(side_effect=RuntimeError("bug")), close=close)

    assert (tmp_path / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1
    close.assert_called_once_with()


def test_a_permanent_fetch_error_closes_the_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a fetch the backend permanently refuses ends the run with its status after closing the backend."""
    backend = FakeBackend(assigns_work=False)
    backend.fetch = Mock(side_effect=PermanentHttpError("forbidden"))  # type: ignore[method-assign]
    run = Mock()

    status, close = run_main(monkeypatch, tmp_path, run, backend=backend)

    assert status == PERMANENT_HTTP_ERROR_EXIT_STATUS
    run.assert_not_called()
    close.assert_called_once_with()


@pytest.mark.parametrize(
    ("run", "close_error", "exit_status"),
    [
        (Mock(return_value=ExitStatus.SUCCESS), None, EXIT_STATUS),
        (Mock(return_value=ExitStatus.TIME_LIMIT_EXCEEDED), None, EXIT_STATUS),
        (Mock(side_effect=YafuError("YAFU failed")), None, 5),
        (Mock(return_value=ExitStatus.SUCCESS), RuntimeError("bug"), CLEANUP_FAILED_EXIT_STATUS),
    ],
)
def test_an_interrupt_while_closing_the_backend_is_reported_by_the_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: Mock, close_error: Exception | None, exit_status: int
) -> None:
    """Test that an interrupt during the final flush still ends the run as interrupted, unless an error outranks it."""

    def interrupt_during_close() -> None:
        monkeypatch.setattr(InterruptState, "interrupted", True)

        if close_error is not None:
            raise close_error

    status, _ = run_main(monkeypatch, tmp_path, run, close=Mock(side_effect=interrupt_during_close))

    assert status == exit_status


@pytest.mark.parametrize(
    ("error", "exit_status"),
    [(Interrupted("interrupted"), EXIT_STATUS), (requests.RequestException("unreachable"), 6)],
    ids=["interrupted", "request-error"],
)
def test_a_backend_that_cannot_start_ends_the_run(
    monkeypatch: pytest.MonkeyPatch, error: Exception, exit_status: int
) -> None:
    """Test that an interrupt while a backend waits to start is reported as one, rather than as a failure."""
    config = make_config()
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(side_effect=error))

    with pytest.raises(SystemExit) as raised:
        start_backend(config, FactoringStats(config.stats_path, read_only=True), InterruptState())

    assert raised.value.code == exit_status
