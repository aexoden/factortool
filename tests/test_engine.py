# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the factorization engine."""

from __future__ import annotations

import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import threading
import time

from typing import TYPE_CHECKING, Literal
from unittest.mock import Mock

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import pytest

from factortool.engine import ExitStatus, FactorEngine
from factortool.interrupt import ABORT, FINISH_BATCH, POLL_INTERVAL, STOP_SOON, Interrupted, InterruptState
from factortool.number import Number
from factortool.stats import ECMCutoffs, FactoringStats
from factortool.tools import YafuError, run_tool

from .helpers import (
    TOOL_STOP_TIMEOUT,
    heartbeat_stopped,
    installed_interrupts,
    make_config,
    make_number,
    make_tool_with_helper,
    wait_for_heartbeat,
)

# Composite, so Number does not treat them as already factored.
COMPOSITES = [100, 102, 104]

# Prime factors of a 41-digit composite, which is too small for CADO-NFS.
C41_FACTORS = [100000000000000000039, 300000000000000000053]


def run_interrupted_on(
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["standard", "yafu"],
    count: int,
    interrupt_after: int,
    level: int = STOP_SOON,
) -> tuple[ExitStatus, int]:
    """Run the engine, raising an interrupt after a given number of factorizations.

    Returns:
        tuple[ExitStatus, int]: The status the engine reported and the number of completed factorizations.
    """
    config = make_config().model_copy(update={"factoring_mode": mode})
    interrupts = InterruptState()
    engine = FactorEngine(config, interrupts)
    numbers = [make_number(n) for n in COMPOSITES[:count]]
    completed = 0

    def factor(self: Number) -> None:
        nonlocal completed
        completed += 1
        self.methods.append(mode)

        # The interrupt lands while this factorization is running; YAFU is in its own process group, so the run
        # always completes rather than being killed.
        if completed == interrupt_after:
            interrupts._level = level

    def factor_ecm(self: Number, level: int) -> None:
        factor(self)
        self._ecm_level = level

    for method in ("factor_yafu_direct", "factor_tf", "factor_rho", "factor_pm1", "factor_final"):
        monkeypatch.setattr(f"factortool.number.Number.{method}", factor, raising=True)

    monkeypatch.setattr("factortool.number.Number.factor_ecm", factor_ecm, raising=True)

    return engine.run(numbers), completed


@pytest.mark.parametrize("mode", ["yafu", "standard"])
@pytest.mark.parametrize(("count", "interrupt_after"), [(3, 1), (3, 3), (1, 1)])
def test_interrupt_is_reported_regardless_of_when_it_arrives(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"], count: int, interrupt_after: int
) -> None:
    """Test that an interrupt is reported even when it arrives during the final number."""
    status, _ = run_interrupted_on(monkeypatch, mode, count, interrupt_after)

    assert status == ExitStatus.INTERRUPTED


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_uninterrupted_run_reports_success(monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]) -> None:
    """Test that an uninterrupted run reports success."""
    status, _ = run_interrupted_on(monkeypatch, mode, 3, 0)

    assert status == ExitStatus.SUCCESS


def test_finished_ecm_number_does_not_resume_when_statistics_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that changing cutoffs between rounds stops ECM without later skipping ahead to resume it."""
    config = make_config()
    stats = Mock(spec=FactoringStats)
    stats.get_ecm_cutoffs.return_value = ECMCutoffs(2, 4)
    stats.get_final_method.return_value = "siqs"
    finished = Number(100, config, stats, None)
    continuing_stats = Mock(spec=FactoringStats)
    continuing_stats.get_ecm_cutoffs.return_value = ECMCutoffs(2, 4)
    continuing_stats.get_final_method.return_value = "siqs"
    continuing = Number(102, config, continuing_stats, None)
    first_level = 2
    ecm_calls: list[tuple[int, int]] = []
    final_calls: list[int] = []

    def factor_ecm(n: int, level: int, *_args: object) -> list[int]:
        ecm_calls.append((n, level))
        if n == continuing.n:
            # Lower the first number's cutoff after round 2, then raise it after round 3.
            # The second number keeps the engine running so round 4 can expose an erroneous resume.
            stats.get_ecm_cutoffs.return_value = ECMCutoffs(2, first_level if level == first_level else 4)
        return [n]

    def factor_yafu(n: int, method: str, *_args: object) -> list[int]:
        assert method == "siqs"
        final_calls.append(n)
        return [2, 2, 5, 5] if n == finished.n else [2, 3, 17]

    for method in ("factor_tf", "factor_rho", "factor_pm1"):
        monkeypatch.setattr(Number, method, Mock(), raising=True)
    monkeypatch.setattr("factortool.number.factor_ecm", factor_ecm, raising=True)
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)

    assert FactorEngine(config, InterruptState()).run([finished, continuing]) == ExitStatus.SUCCESS
    assert ecm_calls == [(100, 2), (102, 2), (102, 3), (102, 4)]
    assert final_calls == [100, 102]
    assert finished.factored
    assert continuing.factored


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_a_single_interrupt_finishes_the_batch(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]
) -> None:
    """Test that a single interrupt lets the batch in hand run to completion."""
    _, expected = run_interrupted_on(monkeypatch, mode, 3, 0)
    status, completed = run_interrupted_on(monkeypatch, mode, 3, 1, level=FINISH_BATCH)

    assert status == ExitStatus.SUCCESS
    assert completed == expected


def test_interrupt_before_the_run_starts_no_yafu_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a second interrupt arriving before the run, such as during the fetch, starts no YAFU factorization."""
    interrupts = InterruptState()
    engine = FactorEngine(make_config().model_copy(update={"factoring_mode": "yafu"}), interrupts)
    factor = Mock()
    monkeypatch.setattr("factortool.number.Number.factor_yafu_direct", factor, raising=True)
    interrupts._level = STOP_SOON

    assert engine.run([make_number(n) for n in COMPOSITES]) == ExitStatus.INTERRUPTED
    factor.assert_not_called()


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_abandoned_factorization_leaves_the_number_unfactored(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]
) -> None:
    """Test that a factorization abandoned by an abort keeps its composite, so shutdown carries it over."""
    engine = FactorEngine(make_config().model_copy(update={"factoring_mode": mode}), InterruptState())
    number = make_number(COMPOSITES[0])

    def abandon(*_args: object) -> list[int]:
        raise Interrupted

    for function in ("factor_tf", "factor_yafu_direct"):
        monkeypatch.setattr(f"factortool.number.{function}", abandon, raising=True)

    assert engine.run([number]) == ExitStatus.INTERRUPTED
    assert number.composite_factors == [COMPOSITES[0]]
    assert not number.factored


def abort_once_started(heartbeat: Path, signum: int) -> threading.Thread:
    """Ask for an abort as soon as a stand-in tool's helper is running, which takes three interrupts.

    Returns:
        threading.Thread: The started thread that raises the signal.
    """

    def abort() -> None:
        wait_for_heartbeat(heartbeat)

        for _ in range(ABORT if signum == signal.SIGINT else 1):
            signal.raise_signal(signum)
            time.sleep(0.1)

    thread = threading.Thread(target=abort)
    thread.start()

    return thread


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_an_abort_kills_the_running_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Literal["standard", "yafu"], signum: int
) -> None:
    """Test that an abort kills the tool, whether it runs in the main thread (YAFU) or a worker thread (rho)."""
    number = make_number(COMPOSITES[0])
    heartbeat = tmp_path / "heartbeat"
    finished = threading.Event()

    def factor(n: int, *_args: object) -> list[int]:
        try:
            run_tool(make_tool_with_helper(heartbeat), tmp_path, 1)
        finally:
            finished.set()

        return [n]

    monkeypatch.setattr("factortool.number.factor_tf", lambda n, _stats: [n], raising=True)
    monkeypatch.setattr("factortool.number.factor_yafu", factor, raising=True)
    monkeypatch.setattr("factortool.number.factor_yafu_direct", factor, raising=True)

    with installed_interrupts() as interrupts:
        engine = FactorEngine(make_config().model_copy(update={"factoring_mode": mode}), interrupts)
        thread = abort_once_started(heartbeat, signum)
        start = time.monotonic()
        status = engine.run([number])
        thread.join()

    assert status == ExitStatus.INTERRUPTED
    assert time.monotonic() - start < TOOL_STOP_TIMEOUT
    assert finished.is_set()
    assert heartbeat_stopped(heartbeat)
    assert number.composite_factors == [COMPOSITES[0]]


def test_an_abort_kills_a_tool_started_after_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a tool is still killed when the abort arrives before the tool has been started."""
    number = make_number(COMPOSITES[0])
    heartbeat = tmp_path / "heartbeat"

    def factor(n: int, *_args: object) -> list[int]:
        # The signal lands after the engine has decided to factor this number, but before the tool is running.
        signal.raise_signal(signal.SIGTERM)
        time.sleep(2 * POLL_INTERVAL)
        run_tool(make_tool_with_helper(heartbeat), tmp_path, 1)

        return [n]

    monkeypatch.setattr("factortool.number.factor_yafu_direct", factor, raising=True)

    with installed_interrupts() as interrupts:
        engine = FactorEngine(make_config().model_copy(update={"factoring_mode": "yafu"}), interrupts)
        start = time.monotonic()
        status = engine.run([number])

    assert status == ExitStatus.INTERRUPTED
    assert time.monotonic() - start < TOOL_STOP_TIMEOUT
    assert not heartbeat.exists() or heartbeat_stopped(heartbeat)
    assert number.composite_factors == [COMPOSITES[0]]


def run_yafu_mode_with_15_failing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, numbers: list[Number]) -> ExitStatus:
    """Run the engine in YAFU mode with a YAFU that exits with an error for 15 and factors 21.

    Returns:
        ExitStatus: The status the engine reported.
    """

    def yafu(cmd: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1] == "factor(15)":
            raise subprocess.CalledProcessError(1, cmd, "", "it broke")

        return subprocess.CompletedProcess(cmd, 0, "***factors found***\nP1 = 3\nP1 = 7\n", "")

    monkeypatch.setattr("factortool.tools.run_tool", yafu, raising=True)
    config = make_config(work_path=tmp_path, factoring_mode="yafu")

    for number in numbers:
        number._config = config

    return FactorEngine(config, InterruptState()).run(numbers)


def test_a_tool_failure_leaves_its_number_unfactored_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a number a tool fails on is left unfactored while the rest of the batch is still factored."""
    failing, succeeding = make_number(15), make_number(21)

    assert run_yafu_mode_with_15_failing(tmp_path, monkeypatch, [failing, succeeding]) == ExitStatus.SUCCESS
    assert failing.composite_factors == [15]
    assert sorted(succeeding.prime_factors) == [3, 7]


def test_direct_yafu_returning_a_composite_unsplit_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that direct YAFU mode treats an unchanged composite as a tool failure rather than a finished run."""
    config = make_config(work_path=tmp_path, factoring_mode="yafu")
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    number = Number(15, config, stats, None)
    run_tool = Mock(return_value=subprocess.CompletedProcess([], 0, "***factors found***\nC2 = 15\n", ""))
    monkeypatch.setattr("factortool.tools.run_tool", run_tool, raising=True)

    with pytest.raises(YafuError, match="failed every time it was run"):
        FactorEngine(config, InterruptState()).run([number])

    assert number.tool_failed
    assert stats.get_yafu_stats(2, 1) == (0, None)


def test_a_tool_that_failed_every_time_ends_the_run_with_its_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a batch too small for repeated failures still reports a tool that never worked."""
    with pytest.raises(YafuError, match="failed every time it was run") as raised:
        run_yafu_mode_with_15_failing(tmp_path, monkeypatch, [make_number(15)])

    assert raised.value.exit_status == YafuError.exit_status


def test_an_interrupted_run_is_not_judged_by_its_tool_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a run cut short by an interrupt is reported as interrupted, even if its only tool run failed."""
    interrupts = InterruptState()
    interrupts._level = FINISH_BATCH
    numbers = [make_number(15), make_number(21)]

    def yafu(cmd: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, "", "it broke")

    monkeypatch.setattr("factortool.tools.run_tool", yafu, raising=True)
    monkeypatch.setattr("factortool.engine.FactorEngine._stop_status", Mock(side_effect=[None, ExitStatus.INTERRUPTED]))
    config = make_config(work_path=tmp_path, factoring_mode="yafu")

    for number in numbers:
        number._config = config

    assert FactorEngine(config, interrupts).run(numbers) == ExitStatus.INTERRUPTED
    assert numbers[0].tool_failed


def test_a_number_set_aside_is_skipped_by_the_later_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a number a tool failed on during rho is not attempted again by P-1, ECM or the final methods."""
    engine = FactorEngine(make_config(), InterruptState())
    failing, other = (make_number(n) for n in COMPOSITES[:2])
    attempted: list[tuple[str, int]] = []

    def factor_rho(self: Number) -> None:
        self.tool_failed = self.n == failing.n

    def record(stage: str) -> Callable[..., None]:
        return lambda self, *_args: attempted.append((stage, self.n))

    monkeypatch.setattr("factortool.number.Number.factor_tf", Mock(), raising=True)
    monkeypatch.setattr("factortool.number.Number.factor_rho", factor_rho, raising=True)
    monkeypatch.setattr("factortool.number.Number.factor_pm1", record("pm1"), raising=True)
    monkeypatch.setattr("factortool.number.Number.factor_final", record("final"), raising=True)
    monkeypatch.setattr("factortool.number.Number.ecm_cutoffs", ECMCutoffs(0, 0), raising=True)

    assert engine.run([failing, other]) == ExitStatus.SUCCESS
    assert attempted == [("pm1", other.n), ("final", other.n)]


@pytest.mark.parametrize("error", [YafuError("YAFU failed"), RuntimeError("bug"), SystemExit(5)])
def test_worker_thread_failure_ends_the_run(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """Test that a failure in a rho worker is raised from the run and stops the numbers still queued from starting."""
    engine = FactorEngine(make_config(), InterruptState())
    attempted: list[int] = []

    def factor_yafu(n: int, *_args: object) -> list[int]:
        attempted.append(n)
        raise error

    monkeypatch.setattr("factortool.number.factor_tf", lambda n, _stats: [n], raising=True)
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)

    with pytest.raises(type(error)) as raised:
        engine.run([make_number(n) for n in COMPOSITES])

    assert raised.value is error
    assert attempted == [COMPOSITES[0]]


def test_worker_thread_failure_keeps_results_already_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a factorization running alongside a failed one finishes and keeps its result."""
    engine = FactorEngine(make_config(max_threads=2), InterruptState())
    failing, succeeding = (make_number(n) for n in COMPOSITES[:2])
    both_started = threading.Barrier(2, timeout=TOOL_STOP_TIMEOUT)
    failure_raised = threading.Event()

    def factor_yafu(n: int, *_args: object) -> list[int]:
        both_started.wait()

        if n == failing.n:
            failure_raised.set()
            message = "YAFU failed"
            raise YafuError(message)

        # Outlast the failure, so this result arrives while the stage is already ending.
        failure_raised.wait(TOOL_STOP_TIMEOUT)
        time.sleep(0.1)
        return [2, 3, 17]

    monkeypatch.setattr("factortool.number.factor_tf", lambda n, _stats: [n], raising=True)
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)

    with pytest.raises(YafuError):
        engine.run([failing, succeeding])

    assert not failing.factored
    assert sorted(succeeding.prime_factors) == [2, 3, 17]


def test_an_abort_kills_a_worker_outlasting_a_failed_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that an abort arriving while a failed stage waits for its other worker kills that worker's tool."""
    failing, outlasting = (make_number(n) for n in COMPOSITES[:2])
    heartbeat = tmp_path / "heartbeat"
    finished = threading.Event()
    threads: list[threading.Thread] = []

    def factor_yafu(n: int, *_args: object) -> list[int]:
        if n == failing.n:
            # Fail only once the other worker's tool is running, so the stage has something to wait for. The abort
            # follows the failure, arriving while the stage is waiting.
            wait_for_heartbeat(heartbeat)
            threads.append(abort_once_started(heartbeat, signal.SIGTERM))
            message = "YAFU failed"
            raise YafuError(message)

        try:
            run_tool(make_tool_with_helper(heartbeat), tmp_path, 1)
        finally:
            finished.set()

        return [n]

    monkeypatch.setattr("factortool.number.factor_tf", lambda n, _stats: [n], raising=True)
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)

    with installed_interrupts() as interrupts:
        engine = FactorEngine(make_config(max_threads=2), interrupts)
        start = time.monotonic()

        # The failure came first, so it is still what the run reports.
        with pytest.raises(YafuError):
            engine.run([failing, outlasting])

        for thread in threads:
            thread.join()

    assert time.monotonic() - start < TOOL_STOP_TIMEOUT
    assert finished.is_set()
    assert heartbeat_stopped(heartbeat)
    assert not failing.factored
    assert outlasting.composite_factors == [COMPOSITES[1]]


class FakeClock:
    """A stand-in for the engine's time module, advanced by hand."""

    def __init__(self) -> None:
        """Initialize the clock at an arbitrary time."""
        self.now = 1000.0

    def monotonic(self) -> float:
        """Report the current fake time.

        Returns:
            float: The current fake time in seconds.
        """
        return self.now


def run_with_clock(
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["standard", "yafu"],
    time_limit: float | None,
    *,
    wait_before_run: float = 0.0,
    seconds_per_factorization: float = 0.0,
) -> tuple[ExitStatus, list[Number]]:
    """Run the engine against a fake clock that advances before the run and during each factorization.

    Returns:
        tuple[ExitStatus, list[Number]]: The status the engine reported and the numbers it was given.
    """
    clock = FakeClock()
    monkeypatch.setattr("factortool.engine.time", clock, raising=True)
    engine = FactorEngine(make_config().model_copy(update={"factoring_mode": mode}), InterruptState())
    numbers = [make_number(n) for n in COMPOSITES]

    def factor(self: Number) -> None:
        clock.now += seconds_per_factorization
        self.prime_factors = [self.n]
        self.composite_factors = []

    for method in ("factor_yafu_direct", "factor_tf"):
        monkeypatch.setattr(f"factortool.number.Number.{method}", factor, raising=True)

    clock.now += wait_before_run

    return engine.run(numbers, time_limit), numbers


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_time_before_the_run_does_not_count_toward_the_time_limit(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]
) -> None:
    """Test that time passing between creating the engine and running it, such as a slow fetch, is not counted."""
    status, numbers = run_with_clock(monkeypatch, mode, 1200.0, wait_before_run=7200.0)

    assert status == ExitStatus.SUCCESS
    assert all(x.factored for x in numbers)


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_exceeding_the_time_limit_ends_the_run(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]
) -> None:
    """Test that the run stops starting factorizations once the time limit has passed."""
    status, numbers = run_with_clock(monkeypatch, mode, 1200.0, seconds_per_factorization=1000.0)

    assert status == ExitStatus.TIME_LIMIT_EXCEEDED
    assert [x.factored for x in numbers] == [True, True, False]


@pytest.mark.parametrize("stage", ["factor_rho", "factor_pm1"])
def test_exceeding_the_time_limit_stops_queued_concurrent_work(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    """Test that a concurrent stage starts no more of its queued numbers once the time limit has passed."""
    clock = FakeClock()
    monkeypatch.setattr("factortool.engine.time", clock, raising=True)
    engine = FactorEngine(make_config(), InterruptState())
    attempted: list[int] = []

    def factor(self: Number) -> None:
        attempted.append(self.n)
        clock.now += 1000.0

    for method in ("factor_tf", "factor_rho", "factor_pm1"):
        monkeypatch.setattr(f"factortool.number.Number.{method}", Mock(), raising=True)

    monkeypatch.setattr(f"factortool.number.Number.{stage}", factor, raising=True)

    assert engine.run([make_number(n) for n in COMPOSITES], 1200.0) == ExitStatus.TIME_LIMIT_EXCEEDED
    assert attempted == COMPOSITES[:2]


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_a_run_without_a_time_limit_is_never_cut_short(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]
) -> None:
    """Test that a run given no time limit finishes however long it takes."""
    status, numbers = run_with_clock(monkeypatch, mode, None, seconds_per_factorization=1e9)

    assert status == ExitStatus.SUCCESS
    assert all(x.factored for x in numbers)


def test_each_run_gets_a_fresh_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a second run on the same engine isn't charged for the first run's time."""
    clock = FakeClock()
    monkeypatch.setattr("factortool.engine.time", clock, raising=True)
    engine = FactorEngine(make_config().model_copy(update={"factoring_mode": "yafu"}), InterruptState())

    def factor(self: Number) -> None:
        clock.now += 1000.0
        self.prime_factors = [self.n]
        self.composite_factors = []

    monkeypatch.setattr("factortool.number.Number.factor_yafu_direct", factor, raising=True)

    assert engine.run([make_number(COMPOSITES[0])], 1200.0) == ExitStatus.SUCCESS
    assert engine.run([make_number(COMPOSITES[1])], 1200.0) == ExitStatus.SUCCESS


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_numbers_whose_assignment_has_expired_are_skipped(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]
) -> None:
    """Test that no work is started on a number whose assignment has expired, while other numbers are unaffected."""
    config = make_config().model_copy(update={"factoring_mode": mode})
    engine = FactorEngine(config, InterruptState())
    lapsed, assigned, unassigned = (make_number(n) for n in COMPOSITES)
    lapsed.expires_at = time.time()
    assigned.expires_at = time.time() + 3600.0
    attempted: set[int] = set()

    def factor(self: Number, *_: object) -> None:
        attempted.add(self.n)
        self.prime_factors = [self.n]
        self.composite_factors = []

    for method in ("factor_yafu_direct", "factor_tf"):
        monkeypatch.setattr(f"factortool.number.Number.{method}", factor, raising=True)

    assert engine.run([lapsed, assigned, unassigned]) == ExitStatus.SUCCESS
    assert attempted == {assigned.n, unassigned.n}
    assert not lapsed.factored


def test_composites_too_small_for_nfs_are_finished_with_siqs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a composite below the CADO-NFS minimum is finished with SIQS when only SIQS has statistics."""
    n = C41_FACTORS[0] * C41_FACTORS[1]
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("siqs", len(str(n)), 1, 0.001)
    config = make_config(work_path=tmp_path)
    number = Number(n, config, stats, None)
    methods: list[str] = []

    def yafu(cmd: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        method = cmd[1].split("(", maxsplit=1)[0]
        methods.append(method)
        output = "".join(f"P21 = {p}\n" for p in C41_FACTORS) if method == "siqs" else f"C41 = {n}\n"
        return subprocess.CompletedProcess(cmd, 0, output, "")

    monkeypatch.setattr("factortool.tools.run_tool", yafu, raising=True)

    assert FactorEngine(config, InterruptState()).run([number]) == ExitStatus.SUCCESS
    assert methods[-1] == "siqs"
    assert number.factored
    assert sorted(number.prime_factors) == C41_FACTORS
