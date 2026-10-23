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

from factortool.backend import FetchCriteria
from factortool.cli.main import (
    Arguments,
    Cleanup,
    acquire_numbers,
    get_time_limits,
    main,
    report_summary,
    report_unfinished,
    start_backend,
    validate_arguments,
    warn_if_assignments_may_expire,
    write_results,
)
from factortool.constants import MAX_CONSECUTIVE_TOOL_FAILURES
from factortool.engine import EngineStatus, FactorEngine, TimeLimits
from factortool.exit_status import ExitStatus
from factortool.http import PermanentHttpError
from factortool.interrupt import Interrupted, InterruptState
from factortool.lock import lock_directories
from factortool.number import Number, Outcome
from factortool.stats import FactoringStats
from factortool.tools import CadoNfsError, ToolError, ToolFailure, YafuError, tool_failures
from factortool.util import WORK_DIR_MARKER

from .helpers import NO_STATS_PATH, make_config, make_number

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

    def __init__(
        self, *, assigns_work: bool = True, accepts_partial_results: bool = True, resumed: Collection[int] = ()
    ) -> None:
        """Initialize the fake backend, which resumes the given composites."""
        self.assigns_work = assigns_work
        self.accepts_partial_results = accepts_partial_results
        self.resumed = resumed
        self.requested: list[int] = []
        self.submitted: list[int] = []
        self.submissions_refused = False

    def resume(self) -> set[Number]:
        """Hand back the composites this backend resumes.

        Returns:
            set[Number]: The resumed composites.
        """
        return {make_number(n) for n in self.resumed}

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

    def checkpoint(self, number: Number) -> None:
        """Ignore the progress made on a number."""

    def get_successful_submission_count(self) -> int:
        """Report how many factorizations were handed over.

        Returns:
            int: The number of submissions recorded.
        """
        return len(self.submitted)

    def close(self) -> None:
        """Close the fake backend, doing nothing."""


def factoring(outcome: EngineStatus | Exception = EngineStatus.SUCCESS) -> Mock:
    """Stand in for an engine run that trial factors every number before ending with the given outcome.

    Returns:
        Mock: A replacement for FactorEngine.run.
    """

    def run(numbers: Collection[Number], _limits: TimeLimits) -> EngineStatus:
        for number in numbers:
            number.factor_tf()

        if isinstance(outcome, Exception):
            raise outcome

        return outcome

    return Mock(side_effect=run)


def test_a_normal_run_tops_the_batch_back_up() -> None:
    """Test that retained work counts toward the batch and the rest is fetched."""
    backend = FakeBackend(resumed=CARRIED_OVER)

    numbers = acquire_numbers(backend, CRITERIA)

    assert backend.requested == [1]
    assert sorted(x.n for x in numbers) == sorted([*CARRIED_OVER, FETCHED[0]])


def test_no_new_work_works_only_what_is_already_assigned() -> None:
    """Test that the no new work option only uses already assigned work."""
    backend = FakeBackend(resumed=CARRIED_OVER)

    numbers = acquire_numbers(backend, CRITERIA, fetch=False)

    assert backend.requested == []
    assert sorted(x.n for x in numbers) == CARRIED_OVER


def test_no_new_work_with_nothing_carried_over_finds_no_work() -> None:
    """Test that the no new work option finds no work when nothing is retained."""
    backend = FakeBackend()

    numbers = acquire_numbers(backend, CRITERIA, fetch=False)

    assert backend.requested == []
    assert numbers == set()


def test_an_early_exit_reports_partial_progress_without_an_assignment(tmp_path: Path) -> None:
    """Test that an early exit reports partial progress for a backend that does not assign work."""
    backend = FakeBackend(assigns_work=False)
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)

    partially_factored = Number(200, make_config(), stats, backend)
    partially_factored.prime_factors = [2, 2]
    partially_factored.composite_factors = [50]
    untouched = Number(300, make_config(), stats, backend)

    report_unfinished(backend, [partially_factored, untouched])

    assert backend.submitted == [200]


def make_unfinished(expires_at: float | None) -> tuple[list[Number], Number]:
    """Build a finished, a partially factored and an untouched number sharing an assignment expiry.

    Returns:
        tuple[list[Number], Number]: All three numbers, and the partially factored one by itself.
    """
    done = make_number(100)
    done.prime_factors = [2, 2, 5, 5]
    done.composite_factors = []

    # A method can return a composite cofactor without the engine reporting anything other than success.
    partially_factored = make_number(200)
    partially_factored.prime_factors = [2]
    partially_factored.composite_factors = [100]
    partially_factored.report_partial = Mock()  # type: ignore[method-assign]

    numbers = [done, partially_factored, make_number(300)]

    for number in numbers:
        number.expires_at = expires_at

    return numbers, partially_factored


@pytest.mark.parametrize("accepts_partial_results", [True, False])
def test_unfinished_work_whose_assignment_is_held_is_left_to_the_backend(
    monkeypatch: pytest.MonkeyPatch, *, accepts_partial_results: bool
) -> None:
    """Test that a partial factorization is neither reported nor discarded while its assignment is held."""
    warning = Mock()
    monkeypatch.setattr("factortool.cli.main.logger.warning", warning, raising=True)
    backend = FakeBackend(accepts_partial_results=accepts_partial_results)
    numbers, partially_factored = make_unfinished(time.time() + 3600.0)

    report_unfinished(backend, numbers)

    partially_factored.report_partial.assert_not_called()  # type: ignore[attr-defined]
    warning.assert_not_called()


def test_a_partial_result_the_backend_does_not_want_is_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a partial factorization is not reported once its assignment has expired, if the backend refuses it."""
    warning = Mock()
    monkeypatch.setattr("factortool.cli.main.logger.warning", warning, raising=True)
    backend = FakeBackend(accepts_partial_results=False)
    numbers, partially_factored = make_unfinished(time.time() - 1.0)

    report_unfinished(backend, numbers)

    partially_factored.report_partial.assert_not_called()  # type: ignore[attr-defined]
    assert [call.args[1] for call in warning.call_args_list] == ["200 <3>"]


def test_a_partial_result_whose_assignment_expired_is_reported() -> None:
    """Test that a backend accepting partial factorizations is sent one that can no longer be completed."""
    backend = FakeBackend(accepts_partial_results=True)
    numbers, partially_factored = make_unfinished(time.time() - 1.0)

    report_unfinished(backend, numbers)

    partially_factored.report_partial.assert_called_once_with()  # type: ignore[attr-defined]


def test_no_new_work_is_rejected_by_a_backend_that_assigns_nothing() -> None:
    """Test that the no new work option is refused on FactorDB, where it could only ever find no work."""
    args = Arguments().parse_args(["--no_new_work"])

    with pytest.raises(SystemExit) as error:
        validate_arguments(args, "factordb")

    assert error.value.code == ExitStatus.INVALID_INPUT


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

    assert error.value.code == ExitStatus.INVALID_INPUT


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
    """Test that arguments the parser rejects exit with the same status as an invalid configuration."""
    with pytest.raises(SystemExit) as error:
        Arguments().parse_args(arguments)

    assert error.value.code == ExitStatus.INVALID_INPUT


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

    assert raised.value.code == ExitStatus.INVALID_INPUT
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

    assert raised.value.code == ExitStatus.INVALID_INPUT
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


@pytest.mark.parametrize("error", [YafuError("YAFU failed"), CadoNfsError("CADO failed")])
def test_a_tool_failure_exits_with_the_tool_failure_status_after_cleaning_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: ToolError
) -> None:
    """Test that a tool failure during the run still saves state before exiting with the tool failure status."""
    write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", "--batch_size", "3"])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState.install", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(return_value=FakeBackend(assigns_work=False)))
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", factoring(error), raising=True)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == ExitStatus.TOOL_FAILURE
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

    def run(_self: FactorEngine, numbers: Collection[Number], _limits: TimeLimits) -> EngineStatus:
        for number in numbers:
            number.factor_tf()

        signal.raise_signal(signal.SIGTERM)
        time.sleep(0.5)

        return EngineStatus.SUCCESS

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

    assert raised.value.code == ExitStatus.INTERRUPTED
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
    run = Mock(return_value=EngineStatus.SUCCESS)
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

    status, close = run_main(monkeypatch, tmp_path, Mock(return_value=EngineStatus.SUCCESS))

    assert status == ExitStatus.CLEANUP_FAILED
    assert (tmp_path / "state" / "stats.json").exists()
    close.assert_called_once_with()


@pytest.mark.parametrize(
    ("run", "exit_status"),
    [
        (Mock(side_effect=YafuError("YAFU failed")), ExitStatus.TOOL_FAILURE),
        (Mock(return_value=EngineStatus.INTERRUPTED), ExitStatus.CLEANUP_FAILED),
        (Mock(return_value=EngineStatus.TIME_LIMIT_EXCEEDED), ExitStatus.CLEANUP_FAILED),
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

    def run(_numbers: Collection[Number], _limits: TimeLimits) -> EngineStatus:
        with contextlib.suppress(ToolFailure):
            tool_failures.failed(YafuError, "ECM", "it broke")

        return EngineStatus.SUCCESS

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        status, _ = run_main(monkeypatch, tmp_path, Mock(side_effect=run))
        tool_failures.reset()

        assert status == 0

    status, _ = run_main(monkeypatch, tmp_path, Mock(side_effect=run))

    assert status == ExitStatus.TOOL_FAILURE


def test_a_failure_to_close_the_backend_is_reported_by_the_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a backend that cannot shut down cleanly is reported once everything else has been saved."""
    close = Mock(side_effect=RuntimeError("bug"))

    status, _ = run_main(monkeypatch, tmp_path, factoring(), close=close)

    assert status == ExitStatus.CLEANUP_FAILED
    assert (tmp_path / "state" / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1


@pytest.mark.parametrize("directory", ["state", "work"])
def test_a_directory_in_use_by_another_instance_ends_the_run_before_any_state_is_touched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, directory: str
) -> None:
    """Test that a second instance sharing the state or working directory refuses to run."""
    run = Mock(return_value=EngineStatus.SUCCESS)

    with lock_directories([tmp_path / directory]):
        status, close = run_main(monkeypatch, tmp_path, run)

    assert status == ExitStatus.ALREADY_RUNNING
    assert not (tmp_path / "state" / "stats.json").exists()
    run.assert_not_called()
    close.assert_not_called()


def test_the_directories_are_released_once_the_run_ends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a finished run does not keep the next one out."""
    for _ in range(2):
        status, _ = run_main(monkeypatch, tmp_path, Mock(return_value=EngineStatus.SUCCESS))

        assert status == ExitStatus.SUCCESS


def test_a_run_removes_the_working_directories_left_by_a_killed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that the tools' working directories left behind by a hard kill are removed once the lock is held."""
    stale = tmp_path / "work" / "cado-nfs-abc"
    stale.mkdir(parents=True)
    (stale / WORK_DIR_MARKER).touch()

    with lock_directories([tmp_path / "work"]):
        status, _ = run_main(monkeypatch, tmp_path, Mock(return_value=EngineStatus.SUCCESS))

    assert status == ExitStatus.ALREADY_RUNNING
    assert stale.is_dir()

    status, _ = run_main(monkeypatch, tmp_path, Mock(return_value=EngineStatus.SUCCESS))

    assert status == ExitStatus.SUCCESS
    assert not stale.exists()


def test_a_run_ended_by_its_time_limit_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that reaching the time limit is not reported as a failure, so a wrapper script starts the next run."""
    status, _ = run_main(monkeypatch, tmp_path, Mock(return_value=EngineStatus.TIME_LIMIT_EXCEEDED))

    assert status == ExitStatus.SUCCESS


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

    assert status == ExitStatus.PERMANENT_HTTP_ERROR
    run.assert_not_called()
    close.assert_called_once_with()


@pytest.mark.parametrize("interrupted", [False, True], ids=["finished", "interrupted"])
def test_a_refused_submission_request_ends_the_run_with_a_permanent_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, interrupted: bool
) -> None:
    """Test that a run whose results the backend refused to accept stops the loop, even if it was also interrupted."""
    backend = FakeBackend(assigns_work=False)
    backend.submissions_refused = True
    monkeypatch.setattr(InterruptState, "interrupted", interrupted)

    status, close = run_main(monkeypatch, tmp_path, factoring(), backend=backend)

    assert status == ExitStatus.PERMANENT_HTTP_ERROR
    close.assert_called_once_with()


@pytest.mark.parametrize(
    ("run", "close_error", "exit_status"),
    [
        (Mock(return_value=EngineStatus.SUCCESS), None, ExitStatus.INTERRUPTED),
        (Mock(return_value=EngineStatus.TIME_LIMIT_EXCEEDED), None, ExitStatus.INTERRUPTED),
        (Mock(side_effect=YafuError("YAFU failed")), None, ExitStatus.TOOL_FAILURE),
        (Mock(return_value=EngineStatus.SUCCESS), RuntimeError("bug"), ExitStatus.CLEANUP_FAILED),
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
    [
        (Interrupted("interrupted"), ExitStatus.INTERRUPTED),
        (PermanentHttpError("forbidden"), ExitStatus.PERMANENT_HTTP_ERROR),
    ],
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
