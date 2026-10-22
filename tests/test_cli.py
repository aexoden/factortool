# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the CLI of factortool."""

from __future__ import annotations

import contextlib
import json
import signal
import sys
import time

from typing import TYPE_CHECKING
from unittest.mock import Mock, call

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

import pytest

from loguru import logger

from factortool.assignments import Assignment, AssignmentStore
from factortool.backend import FetchCriteria
from factortool.cli.main import (
    ALREADY_RUNNING_EXIT_STATUS,
    CLEANUP_FAILED_EXIT_STATUS,
    Arguments,
    Cleanup,
    acquire_numbers,
    get_time_limits,
    main,
    preserve_unfinished,
    report_summary,
    start_backend,
    validate_arguments,
    warn_if_assignments_may_expire,
    write_results,
)
from factortool.constants import MAX_CONSECUTIVE_TOOL_FAILURES
from factortool.engine import ExitStatus, FactorEngine, TimeLimits
from factortool.http import PermanentHttpError
from factortool.interrupt import EXIT_STATUS, Interrupted, InterruptState
from factortool.lock import lock_directories
from factortool.number import Number, Outcome
from factortool.stats import FactoringStats
from factortool.tools import CadoNfsError, ToolError, ToolFailure, YafuError, tool_failures

from .helpers import NO_STATS_PATH, make_config, make_number

# The exit status of a run ended by a permanent HTTP error in the backend.
PERMANENT_HTTP_ERROR_EXIT_STATUS = 6

CRITERIA = FetchCriteria(count=3, min_digits=1, max_digits=100)

# Composite, so Number does not treat them as already factored.
CARRIED_OVER = [100, 102]
FETCHED = [104, 106, 108]


def carry_over(state_path: Path, lifetime: float = 1800.0) -> AssignmentStore:
    """Retain CARRIED_OVER as unfinished assignments from a previous run.

    Returns:
        AssignmentStore: A store that will resume them.
    """
    expires_at = time.time() + lifetime
    store = AssignmentStore(state_path, "mersenne_ca")
    store.save(Assignment(n=n, expires_at=expires_at) for n in CARRIED_OVER)

    return store


def write_config(tmp_path: Path, **overrides: object) -> None:
    """Write a configuration file whose tools are present, with any executable standing in for YAFU."""
    config = {"backend": "factordb", "max_threads": 1, "yafu_path": sys.executable, **overrides}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


class FakeBackend:
    """A backend that records requests, assigning work unless told otherwise."""

    assignment_lifetime = 3600.0

    def __init__(self, *, assigns_work: bool = True, accepts_partial_results: bool = True) -> None:
        """Initialize the fake backend."""
        self.assigns_work = assigns_work
        self.accepts_partial_results = accepts_partial_results
        self.requested: list[int] = []
        self.submitted: list[int] = []

    def fetch(self, criteria: FetchCriteria) -> set[Number]:
        """Record the request and hand back the canned composites.

        Returns:
            set[Number]: The composites this backend always returns.
        """
        self.requested.append(criteria.count)
        config = make_config()
        stats = FactoringStats(NO_STATS_PATH, read_only=True)

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


def factoring(outcome: ExitStatus | Exception = ExitStatus.SUCCESS) -> Mock:
    """Stand in for an engine run that trial factors every number before ending with the given outcome.

    Returns:
        Mock: A replacement for FactorEngine.run.
    """

    def run(numbers: Collection[Number], _limits: TimeLimits) -> ExitStatus:
        for number in numbers:
            number.factor_tf()

        if isinstance(outcome, Exception):
            raise outcome

        return outcome

    return Mock(side_effect=run)


def test_a_normal_run_tops_the_batch_back_up(tmp_path: Path) -> None:
    """Test that retained work counts toward the batch and the rest is fetched."""
    backend = FakeBackend()
    store = carry_over(tmp_path / "assignments.json")

    numbers = acquire_numbers(backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA)

    assert backend.requested == [1]
    assert sorted(x.n for x in numbers) == sorted([*CARRIED_OVER, FETCHED[0]])


def test_acquired_numbers_carry_their_assignment_expiry(tmp_path: Path) -> None:
    """Test that retained numbers keep their original expiry, and fetched ones are given the backend's lifetime."""
    backend = FakeBackend()
    store = carry_over(tmp_path / "assignments.json", lifetime=1800.0)
    start = time.time()

    numbers = acquire_numbers(backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA)

    remaining = {x.n: x.expires_at - start for x in numbers if x.expires_at is not None}

    assert remaining == {
        **dict.fromkeys(CARRIED_OVER, pytest.approx(1800.0, abs=5.0)),
        FETCHED[0]: pytest.approx(backend.assignment_lifetime, abs=5.0),
    }


def test_a_retained_number_resumes_from_the_factors_already_found(tmp_path: Path) -> None:
    """Test that a number retained with factors is rebuilt with them, rather than started from scratch."""
    backend = FakeBackend()
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    store.save([Assignment(n=15015, expires_at=time.time() + 1800.0, factors=[3, 5, 1001])])

    (number,) = acquire_numbers(
        backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA, fetch=False
    )

    assert (number.prime_factors, number.composite_factors) == ([3, 5], [1001])


def test_a_retained_number_assigned_again_keeps_its_factors_and_takes_the_new_expiry(tmp_path: Path) -> None:
    """Test that a retained composite the backend hands out again is neither restarted nor left to expire early."""
    backend = FakeBackend()
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    store.save([Assignment(n=FETCHED[0], expires_at=time.time() + 60.0, factors=[2, FETCHED[0] // 2])])
    start = time.time()

    numbers = acquire_numbers(backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA)
    again = next(x for x in numbers if x.n == FETCHED[0])

    assert again.prime_factors == [2]
    assert again.expires_at == pytest.approx(start + backend.assignment_lifetime, abs=5.0)


def test_no_new_work_works_only_what_is_already_assigned(tmp_path: Path) -> None:
    """Test that the no new work option only uses already assigned work."""
    backend = FakeBackend()
    store = carry_over(tmp_path / "assignments.json")

    numbers = acquire_numbers(
        backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA, fetch=False
    )

    assert backend.requested == []
    assert sorted(x.n for x in numbers) == CARRIED_OVER


def test_no_new_work_with_nothing_carried_over_finds_no_work(tmp_path: Path) -> None:
    """Test that the no new work option finds no work when nothing is retained."""
    backend = FakeBackend()
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")

    numbers = acquire_numbers(
        backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA, fetch=False
    )

    assert backend.requested == []
    assert numbers == set()


def test_an_early_exit_reports_partial_progress_without_an_assignment(tmp_path: Path) -> None:
    """Test that an early exit reports partial progress for a backend that does not assign work."""
    backend = FakeBackend(assigns_work=False)
    store = AssignmentStore(tmp_path / "assignments.json", "factordb")
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
        ["--time_limit", "-1"],
        ["--time_limit", "nan"],
        ["--hard_time_limit", "-1"],
        ["--hard_time_limit", "inf"],
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
    [
        [],
        ["--min_digits", "1", "--max_digits", "1"],
        ["--batch_size", "5", "--skip_count", "0"],
        ["--time_limit", "0", "--hard_time_limit", "0"],
    ],
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


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ("factordb", "standard", 1)),
        (
            ["--backend", "mersenne_ca", "--factoring_mode", "yafu", "--max_threads", "8", "--max_digits", "80"],
            ("mersenne_ca", "yafu", 8),
        ),
    ],
    ids=["configured", "overridden"],
)
def test_the_options_override_the_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], expected: tuple[str, str, int]
) -> None:
    """Test that the backend, factoring mode and thread count options override the configured settings."""
    write_config(tmp_path, gimps_login="user")
    validate_tools = Mock(side_effect=SystemExit(1))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", *argv])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.validate_tools", validate_tools, raising=True)

    with pytest.raises(SystemExit):
        main()

    (config,) = validate_tools.call_args.args

    assert (config.backend, config.factoring_mode, config.max_threads) == expected


@pytest.mark.parametrize("argv", [["--max_threads", "0"], ["--backend", "mersenne_ca", "--max_digits", "80"]])
def test_an_invalid_override_ends_the_run_as_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]
) -> None:
    """Test that an override the configuration rejects ends the run before the backend is started."""
    write_config(tmp_path)
    create_backend = Mock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", *argv])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", create_backend)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 1
    create_backend.assert_not_called()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], [call(None), call("WARNING")]), (["--log_level", "debug"], [call("DEBUG"), call("DEBUG")])],
    ids=["configured", "overridden"],
)
def test_the_log_level_option_overrides_the_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], expected: list[object]
) -> None:
    """Test that logging is set up before the configuration is read, then at the configured level unless overridden."""
    write_config(tmp_path, log_level="warning", yafu_path="missing-yafu")
    setup_logger = Mock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", *argv])
    monkeypatch.setattr("factortool.cli.main.setup_logger", setup_logger, raising=True)

    # The missing tool ends the run once the configuration has been read.
    with pytest.raises(SystemExit):
        main()

    assert setup_logger.call_args_list == expected


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
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", factoring(error), raising=True)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == exit_status
    assert (tmp_path / "state" / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1


def test_a_termination_signal_still_saves_state_before_exiting_as_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a run ended by SIGTERM saves its state and results, and exits with the interrupted status."""
    write_config(tmp_path)
    backend = FakeBackend(assigns_work=False)
    backend.close = Mock()  # type: ignore[method-assign]
    interrupts = InterruptState()

    def run(_self: FactorEngine, numbers: Collection[Number], _limits: TimeLimits) -> ExitStatus:
        for number in numbers:
            number.factor_tf()

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
    assert (tmp_path / "state" / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1
    backend.close.assert_called_once()


@pytest.mark.parametrize(
    ("arguments", "assigns_work", "limits"),
    [
        (["--target_duration", "300"], False, TimeLimits(soft=600.0)),
        (["--target_duration", "300"], True, TimeLimits()),
        (["--batch_size", "3", "--target_duration", "300"], False, TimeLimits()),
        (["--time_limit", "900"], True, TimeLimits(soft=900.0)),
        (["--batch_size", "3", "--time_limit", "900"], False, TimeLimits(soft=900.0)),
        (["--time_limit", "0"], False, TimeLimits()),
        (["--hard_time_limit", "7200"], False, TimeLimits(soft=1200.0, hard=7200.0)),
        (["--hard_time_limit", "7200"], True, TimeLimits(hard=7200.0)),
        (["--time_limit", "0", "--hard_time_limit", "7200"], False, TimeLimits(hard=7200.0)),
    ],
    ids=[
        "automatic-batch",
        "assigned-work",
        "explicit-batch",
        "explicit-limit-on-assigned-work",
        "explicit-limit-on-explicit-batch",
        "disabled",
        "hard-limit",
        "hard-limit-on-assigned-work",
        "hard-limit-only",
    ],
)
def test_the_time_limits_follow_the_arguments_and_the_backend(
    arguments: list[str], limits: TimeLimits, *, assigns_work: bool
) -> None:
    """Test that only an automatic batch of unassigned work is limited by default, and that the options override it."""
    backend = FakeBackend(assigns_work=assigns_work)

    assert get_time_limits(Arguments().parse_args(arguments), backend) == limits


@pytest.mark.parametrize(
    ("arguments", "warned"),
    [
        (["--target_duration", "3601"], True),
        (["--target_duration", "3600"], False),
        (["--target_duration", "3601", "--batch_size", "3"], False),
    ],
)
def test_a_target_duration_that_outlasts_assignments_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], *, warned: bool
) -> None:
    """Test that the expiry warning depends on the target duration, and is skipped for an explicit batch size."""
    warning = Mock()
    monkeypatch.setattr("factortool.cli.main.logger.warning", warning, raising=True)

    warn_if_assignments_may_expire(FakeBackend(), Arguments().parse_args(arguments))

    assert warning.called == warned


@pytest.mark.parametrize(
    ("arguments", "limits"),
    [([], TimeLimits(soft=1200.0)), (["--batch_size", "3", "--hard_time_limit", "60"], TimeLimits(hard=60.0))],
)
def test_the_run_is_given_the_time_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arguments: list[str], limits: TimeLimits
) -> None:
    """Test that the time limits reach the engine with the run rather than when the engine is created."""
    write_config(tmp_path)
    run = Mock(return_value=ExitStatus.SUCCESS)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", *arguments])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState.install", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(return_value=FakeBackend(assigns_work=False)))
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", run, raising=True)

    main()

    assert run.call_args.args[1] == limits


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
    assert (tmp_path / "state" / "stats.json").exists()
    close.assert_called_once_with()


def test_a_failure_to_save_assignments_is_reported_by_the_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that losing track of unfinished assignments counts as a cleanup failure, without skipping later steps."""
    monkeypatch.setattr("factortool.cli.main.AssignmentStore.save", Mock(side_effect=OSError("disk full")))

    status, close = run_main(monkeypatch, tmp_path, factoring(), backend=FakeBackend())

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


def test_tool_failures_are_saved_for_the_next_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a tool failing once in each run is reported as broken once the runs add up."""

    def run(_numbers: Collection[Number], _limits: TimeLimits) -> ExitStatus:
        with contextlib.suppress(ToolFailure):
            tool_failures.failed(YafuError, "ECM", "it broke")

        return ExitStatus.SUCCESS

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        status, _ = run_main(monkeypatch, tmp_path, Mock(side_effect=run))
        tool_failures.reset()

        assert status == 0

    status, _ = run_main(monkeypatch, tmp_path, Mock(side_effect=run))

    assert status == YafuError.exit_status


def test_a_failure_to_close_the_backend_is_reported_by_the_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a backend that cannot shut down cleanly is reported once everything else has been saved."""
    close = Mock(side_effect=RuntimeError("bug"))

    status, _ = run_main(monkeypatch, tmp_path, factoring(), close=close)

    assert status == CLEANUP_FAILED_EXIT_STATUS
    assert (tmp_path / "state" / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1


@pytest.mark.parametrize("directory", ["state", "work"])
def test_a_directory_in_use_by_another_instance_ends_the_run_before_any_state_is_touched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, directory: str
) -> None:
    """Test that a second instance sharing the state or working directory refuses to run."""
    run = Mock(return_value=ExitStatus.SUCCESS)

    with lock_directories([tmp_path / directory]):
        status, close = run_main(monkeypatch, tmp_path, run)

    assert status == ALREADY_RUNNING_EXIT_STATUS
    assert not (tmp_path / "state" / "stats.json").exists()
    run.assert_not_called()
    close.assert_not_called()


def test_the_directories_are_released_once_the_run_ends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a finished run does not keep the next one out."""
    for _ in range(2):
        status, _ = run_main(monkeypatch, tmp_path, Mock(return_value=ExitStatus.SUCCESS))

        assert status == 0


def summarize(numbers: list[Number]) -> list[str]:
    """Report the summary of a run, capturing the log messages.

    Returns:
        list[str]: The log messages, without their trailing newlines.
    """
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")

    try:
        report_summary(numbers)
    finally:
        logger.remove(sink)

    return [message.rstrip("\n") for message in messages]


def test_the_summary_only_counts_tool_failures_as_failures() -> None:
    """Test that numbers the run never finished are reported as such, rather than as having failed to factor."""
    factored, failed, expired, unfinished, *_ = numbers = [make_number(n) for n in (100, 102, 104, 106, 108, 110)]
    factored.prime_factors, factored.composite_factors, factored.methods = [2, 2, 5, 5], [], ["TF", "TF"]
    failed.tool_failed = True
    expired.expires_at = 0.0
    unfinished.started = True

    assert summarize(numbers) == [
        "1 number failed to factor: 102 <3>",
        "Factored 1 of 6 numbers (1 failed, 1 expired, 1 unfinished, 2 not started)",
        "The following methods were used: TF (1)",
    ]


def test_the_summary_of_a_complete_run_has_nothing_else_to_report() -> None:
    """Test that a run that factored everything reports no other outcomes."""
    number = make_number(100)
    number.prime_factors, number.composite_factors, number.methods = [2, 2, 5, 5], [], ["TF"]

    assert summarize([number]) == ["Factored 1 of 1 number", "The following methods were used: TF (1)"]


def test_the_summary_of_a_run_that_started_nothing_lists_no_methods() -> None:
    """Test that a run stopped before any method found a factor does not report an empty list of methods."""
    assert summarize([make_number(100), make_number(102)]) == ["Factored 0 of 2 numbers (2 not started)"]


# A product of two primes too large for trial factoring to find.
HARD_COMPOSITE = 1000003 * 1000033


def read_results(output_path: Path) -> list[str]:
    """Read the contents of every results file in a directory.

    Returns:
        list[str]: The contents of each file, in order of file name.
    """
    return [path.read_text(encoding="utf-8") for path in sorted(output_path.iterdir())]


def test_results_written_in_the_same_second_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """Test that a result file whose name is already taken is left alone."""
    numbers = [make_number(n) for n in (4, 6)]

    for number in numbers:
        number.factor_tf()
        write_results([number], tmp_path)

    assert sorted(read_results(tmp_path)) == ["4=2 2\n", "6=2 3\n"]


def test_results_only_list_numbers_the_run_found_a_factor_of(tmp_path: Path) -> None:
    """Test that numbers the run learned nothing about are left out, whatever state they ended in."""
    not_started, failed, expired, no_factor, partial, factored = numbers = [
        make_number(n) for n in (100, 102, 104, HARD_COMPOSITE, 2 * HARD_COMPOSITE, 106)
    ]
    failed.tool_failed = True
    expired.expires_at = 0.0

    for number in (no_factor, partial, factored):
        number.factor_tf()

    assert not_started.outcome is Outcome.NOT_STARTED
    assert no_factor.outcome is Outcome.UNFINISHED

    write_results(set(numbers), tmp_path)

    assert read_results(tmp_path) == [f"106=2 53\n{2 * HARD_COMPOSITE}=2 {HARD_COMPOSITE}\n"]


def test_results_leave_out_a_resumed_number_until_it_gains_a_factor(tmp_path: Path) -> None:
    """Test that factors carried over from an earlier run are not recorded again until the number gets further."""
    config, stats = make_config(), FactoringStats(NO_STATS_PATH, read_only=True)
    number = Number(4 * HARD_COMPOSITE, config, stats, None, [4, HARD_COMPOSITE])

    write_results([number], tmp_path / "results")

    assert not (tmp_path / "results").exists()

    number.factor_tf()
    write_results([number], tmp_path / "results")

    assert read_results(tmp_path / "results") == [f"{4 * HARD_COMPOSITE}=2 2 {HARD_COMPOSITE}\n"]


def test_an_unexpected_engine_failure_still_cleans_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a failure other than a tool error is raised only after the state is saved and the backend closed."""
    close = Mock()

    with pytest.raises(RuntimeError, match="bug"):
        run_main(monkeypatch, tmp_path, factoring(RuntimeError("bug")), close=close)

    assert (tmp_path / "state" / "stats.json").exists()
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
    [(Interrupted("interrupted"), EXIT_STATUS), (PermanentHttpError("forbidden"), PERMANENT_HTTP_ERROR_EXIT_STATUS)],
    ids=["interrupted", "permanent-error"],
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
