# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Factoring tests for factortool."""

from __future__ import annotations

import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from factortool.constants import MAX_CONSECUTIVE_TOOL_FAILURES
from factortool.number import (
    FinalMethodNeeded,
    Number,
    Outcome,
    factor_ecm,
    factor_nfs_cado,
    factor_tf,
    factor_yafu,
    factor_yafu_direct,
)
from factortool.progress import CompositeProgress
from factortool.stats import ECMCutoffs, FactoringStats
from factortool.tools import CadoNfsError, ToolError, ToolFailure, YafuError, run_tool

from .helpers import NO_STATS_PATH, fake_time, make_config, make_number

# A composite below the CADO-NFS minimum.
C56 = 10**55 + 1

# Composites on either side of the YAFU NFS minimum, each with a small factor.
C84 = 10**83 + 1
C84_SPLIT = [11, C84 // 11]
C90 = 10**89 + 7 * 13
C90_SPLIT = [101, C90 // 101]


def is_prime_unless_c90(n: int) -> bool:
    """Treat every number but C90 as prime, so that only C90 needs factoring.

    Returns:
        bool: Whether n is to be treated as prime.
    """
    return n != C90


def never_prime(_n: int) -> bool:
    """Treat every number as composite, so that test factors need not be real primes.

    Returns:
        bool: Always False.
    """
    return False


def yafu_output(factors: list[int]) -> str:
    """Build the part of YAFU's output that lists the given factors.

    Returns:
        str: The list of factors, in YAFU's format.
    """
    return "\n\n***factors found***\n" + "".join(f"C{len(str(x))} = {x}\n" for x in factors) + "\nans = 1\n"


def test_tf() -> None:
    """Test trial factoring."""
    n = 15825810
    stats = FactoringStats(NO_STATS_PATH, read_only=True)
    assert factor_tf(n, stats) == [2, 3, 5, 7, 11, 13, 17, 31]


def test_tf_time_excludes_generating_the_primes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that generating the trial factoring primes on first use doesn't count towards the time taken."""
    clock = 0

    def small_primes() -> tuple[int, ...]:
        nonlocal clock
        clock += 1_000_000_000
        return (2, 3, 5)

    monkeypatch.setattr("factortool.number.small_primes", small_primes)
    monkeypatch.setattr("factortool.number.time", fake_time(perf_counter_ns=lambda: clock))
    update_probability = Mock()
    stats = Mock(update_probability=update_probability)

    assert factor_tf(2 * 3 * 5 * 7 * 11, stats) == [2, 3, 5, 77]

    update_probability.assert_called_once_with(4, "tf", 1, 0.0, success=True)


def resumed_number(n: int, factors: list[int]) -> Number:
    """Build a Number resuming from factors found by a previous run.

    Returns:
        Number: The resumed number.
    """
    return Number(n, make_config(), FactoringStats(NO_STATS_PATH, read_only=True), None, factors)


def test_a_number_resumes_from_saved_factors() -> None:
    """Test that saved factors are split into primes and the composites still to be factored."""
    number = resumed_number(15015, [1001, 3, 5])

    assert (number.prime_factors, number.composite_factors) == ([3, 5], [1001])
    assert number.partially_factored
    assert number.known_factors == [3, 5, 1001]


@pytest.mark.parametrize("factors", [[3, 5], [1, 15015], [-3, -5005], []], ids=["product", "one", "negative", "none"])
def test_a_number_ignores_saved_factors_that_are_not_a_factorization(factors: list[int]) -> None:
    """Test that a number starts from scratch if its saved factors are missing or unusable."""
    number = resumed_number(15015, factors)

    assert (number.prime_factors, number.composite_factors) == ([], [15015])
    assert not number.partially_factored


@pytest.mark.parametrize("n", [1, 0, -15], ids=["one", "zero", "negative"])
def test_a_number_below_two_is_rejected(n: int) -> None:
    """Test that a number with no prime factorization is refused rather than treated as a composite."""
    with pytest.raises(ValueError, match="cannot be factored"):
        _ = make_number(n)


def test_a_number_split_into_composites_is_partially_factored() -> None:
    """Test that progress is recognized even when no prime factor has been found yet."""
    assert resumed_number(1001 * 1003, [1001, 1003]).partially_factored


def test_small_cofactors_use_statistical_ecm_cutoffs() -> None:
    """Test that a composite too small for CADO-NFS stops ECM at the learned cutoff rather than the maximum level."""
    number = make_number(15 * 10**50 + 7)
    get_ecm_cutoffs = Mock(return_value=ECMCutoffs(2, 5))
    number._stats = Mock(spec_set=FactoringStats, get_ecm_cutoffs=get_ecm_cutoffs)
    number.composite_factors = [10**40 + 1]
    number._progress = {10**40 + 1: CompositeProgress(n=10**40 + 1, ecm_level=4)}

    assert number.ecm_needed()

    get_ecm_cutoffs.assert_called_once_with(41, number._config.max_threads, ("siqs",))

    number._progress = {10**40 + 1: CompositeProgress(n=10**40 + 1, ecm_level=5)}
    assert not number.ecm_needed()


def test_no_ecm_is_needed_when_the_cutoff_is_none() -> None:
    """Test that a number that has done no ECM does none when the cutoff says none is worthwhile."""
    number = make_number(10**59 + 1)
    number._stats = Mock(spec_set=FactoringStats, get_ecm_cutoffs=Mock(return_value=ECMCutoffs(0, 0)))

    assert not number.ecm_needed()


def test_ecm_cutoff_follows_the_latest_statistics() -> None:
    """Test that the ECM cutoff is reevaluated as statistics arrive, and that a finished number stays finished."""
    number = make_number(10**59 + 1)
    get_ecm_cutoffs = Mock()
    number._stats = Mock(spec_set=FactoringStats, get_ecm_cutoffs=get_ecm_cutoffs)
    number._progress = {number.n: CompositeProgress(n=number.n, ecm_level=10)}

    # Each reading is collected rather than asserted in turn, which type checkers would narrow as if it couldn't change.
    # In the last case, resuming would skip the levels the engine ran in the meantime.
    needed: list[bool] = []
    for cutoffs in (ECMCutoffs(None, 20), ECMCutoffs(8, 10), ECMCutoffs(8, 20)):
        get_ecm_cutoffs.return_value = cutoffs
        needed.append(number.ecm_needed())

    assert needed == [True, False, False]


def test_yafu_nfs_forces_nfs_and_records_its_own_statistics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that YAFU NFS is forced below YAFU's QS/NFS crossover and recorded separately from CADO-NFS."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    run_tool = Mock(return_value=subprocess.CompletedProcess([], 0, yafu_output(C90_SPLIT), ""))
    monkeypatch.setattr("factortool.tools.run_tool", run_tool, raising=True)

    factors = factor_yafu(C90, "nfs", 4, make_config(work_path=tmp_path).yafu_paths, stats)

    cmd = cast("list[str]", run_tool.call_args.args[0])
    assert factors == C90_SPLIT
    assert cmd[1] == f"nfs({C90})"
    assert cmd[cmd.index("-xover") + 1] == "1"
    assert stats.get_final_stats("nfs_yafu", 90, 4)[0] == 1
    assert stats.get_final_stats("nfs_cado", 90, 4)[0] == 0


@pytest.mark.parametrize("tool", ["nfs_yafu", "nfs_cado"])
def test_nfs_is_not_run_below_its_minimum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool: str) -> None:
    """Test that asking for NFS on a composite too small for it is a bug, rather than a run that found nothing."""
    run_tool = Mock()
    monkeypatch.setattr("factortool.tools.run_tool", run_tool, raising=True)

    stats = FactoringStats(tmp_path / "stats.json", read_only=True)

    calls: dict[str, Callable[[], object]] = {
        "nfs_yafu": lambda: factor_yafu(C84, "nfs", 4, make_config().yafu_paths, stats),
        "nfs_cado": lambda: factor_nfs_cado(C56, 4, Path("cado-nfs.py"), tmp_path, stats),
    }

    with pytest.raises(ValueError, match="needs at least"):
        _ = calls[tool]()

    run_tool.assert_not_called()


def test_final_factoring_chooses_a_method_for_each_composite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that each remaining composite is factored with the fastest final method eligible for its own size."""
    config = make_config(cado_nfs_path="cado-nfs.py", use_nfs_cado=True, use_nfs_yafu=True)
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)

    for method, execution_time in (("siqs", 100.0), ("nfs_yafu", 50.0), ("nfs_cado", 80.0)):
        stats.update_final(method, 84, 1, execution_time)
        stats.update_final(method, 90, 1, execution_time)

    number = Number(C84 * C90, config, stats, None)
    number.composite_factors = [C84, C90]
    calls: list[tuple[int, str]] = []

    splits = {C84: C84_SPLIT, C90: C90_SPLIT}

    def factor_yafu(n: int, method: str, *_args: object) -> list[int]:
        calls.append((n, method + "_yafu"))
        return splits[n]

    def factor_nfs_cado(n: int, *_args: object) -> list[int]:
        calls.append((n, "nfs_cado"))
        return splits[n]

    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)
    monkeypatch.setattr("factortool.number.factor_nfs_cado", factor_nfs_cado, raising=True)

    # Treat the cofactors as prime, so that only the two composites need a final method.
    def is_prime(n: int) -> bool:
        return n not in splits

    monkeypatch.setattr("factortool.number.is_prime", is_prime, raising=True)

    assert number.final_method == "nfs_yafu"

    number.factor_final()

    assert calls == [(C84, "nfs_cado"), (C90, "nfs_yafu")]


def test_ecm_runs_yafu_nfs_immediately_without_statistics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that ECM gives way to an immediate YAFU NFS run while YAFU NFS has no data for the composite's size."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("nfs_cado", 90, 1, 80.0)
    number = Number(C90, make_config(use_nfs_yafu=True, max_siqs_digits=80), stats, None)
    factor_yafu = Mock(return_value=C90_SPLIT)
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)
    monkeypatch.setattr("factortool.number.is_prime", is_prime_unless_c90, raising=True)

    number.factor_ecm(2)

    factor_yafu.assert_called_once_with(C90, "nfs", *number._yafu_args)
    assert number.factored


def test_tools_inherit_the_environment_with_their_thread_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a tool keeps the surrounding environment, with OpenMP limited to the threads it is allowed."""
    monkeypatch.setenv("FACTORTOOL_TEST", "inherited")
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    script = "import os; print(os.environ['FACTORTOOL_TEST'], os.environ['OMP_NUM_THREADS'], 'PATH' in os.environ)"

    result = run_tool([sys.executable, "-c", script], tmp_path, 3)

    assert result.stdout.split() == ["inherited", "3", "True"]


def test_tools_do_not_see_the_virtual_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a tool doesn't find factortool's virtual environment on PATH, as CADO-NFS would then use its Python."""
    venv_path = tmp_path / "venv"
    venv_bin = venv_path / ("Scripts" if sys.platform == "win32" else "bin")
    other_bin = tmp_path / "bin"
    monkeypatch.setattr("sys.prefix", str(venv_path), raising=True)
    monkeypatch.setattr("sys.base_prefix", str(tmp_path / "base"), raising=True)
    monkeypatch.setenv("PATH", os.pathsep.join([str(venv_bin), str(other_bin), str(venv_bin)]))
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_path))
    script = "import os; print(os.environ['PATH']); print('VIRTUAL_ENV' in os.environ)"

    result = run_tool([sys.executable, "-c", script], tmp_path, 1)

    assert result.stdout.splitlines() == [str(other_bin), "False"]


@pytest.mark.parametrize(
    ("tool", "threads"),
    [("ecm", 4), ("rho", 1), ("pm1", 1), ("siqs", 4), ("nfs", 4), ("yafu_direct", 4), ("nfs_cado", 4)],
)
def test_tools_are_given_the_threads_their_method_can_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool: str, threads: int
) -> None:
    """Test that the single-threaded methods are limited to one thread, and every other method to max_threads.

    The tool's own thread option has to agree with the budget, as OMP_NUM_THREADS only limits what uses OpenMP.
    """
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("siqs", 90, 4, 1.0)
    output = " ".join(map(str, C90_SPLIT)) if tool == "nfs_cado" else yafu_output(C90_SPLIT)
    run_tool = Mock(return_value=subprocess.CompletedProcess([], 0, output, ""))
    monkeypatch.setattr("factortool.tools.run_tool", run_tool, raising=True)
    config = make_config(work_path=tmp_path)
    yafu = config.yafu_paths

    if tool == "ecm":
        _ = factor_ecm(C90, 2, config.final_methods, 4, yafu, stats)
    elif tool == "yafu_direct":
        _ = factor_yafu_direct(C90, 4, yafu, stats)
    elif tool == "nfs_cado":
        _ = factor_nfs_cado(C90, 4, Path("cado-nfs.py"), tmp_path, stats)
    else:
        _ = factor_yafu(C90, tool, 4, yafu, stats)

    cmd = cast("list[str]", run_tool.call_args.args[0])
    option = "-t" if tool == "nfs_cado" else "-threads"

    # YAFU uses a single thread unless told otherwise.
    assert (int(cmd[cmd.index(option) + 1]) if option in cmd else 1) == threads
    assert run_tool.call_args.args[2] == threads


def failing_tools(tmp_path: Path, stats: FactoringStats) -> dict[str, tuple[Callable[[], object], type[ToolError]]]:
    """Build a call to each tool-backed factoring function, with the error its repeated failure should give.

    Returns:
        dict[str, tuple[Callable[[], object], type[ToolError]]]: The call and error type by name.
    """
    config = make_config(work_path=tmp_path, yafu_path=tmp_path / "missing-yafu")
    yafu = config.yafu_paths

    return {
        "ecm": (lambda: factor_ecm(C90, 2, config.final_methods, 1, yafu, stats), YafuError),
        "rho": (lambda: factor_yafu(C90, "rho", 1, yafu, stats), YafuError),
        "siqs": (lambda: factor_yafu(C90, "siqs", 1, yafu, stats), YafuError),
        "yafu_direct": (lambda: factor_yafu_direct(C90, 1, yafu, stats), YafuError),
        "nfs_cado": (lambda: factor_nfs_cado(C90, 1, tmp_path / "missing", tmp_path, stats), CadoNfsError),
    }


TOOLS = ["ecm", "rho", "siqs", "yafu_direct", "nfs_cado"]


def recorded_runs(stats: FactoringStats) -> int:
    """Count the runs recorded in the statistics for a 90-digit composite.

    Returns:
        int: The number of recorded runs of any method.
    """
    return (
        sum(stats.get_final_stats(method, 90, 1)[0] for method in ("siqs", "nfs_yafu", "nfs_cado"))
        + sum(stats.get_probability_stats(90, method, 1)[0] for method in ("rho", "pm1", "yafu"))
        + stats.get_ecm_stats(90, 2, 1)[0]
    )


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_exiting_with_an_error_fails_without_recording_statistics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool: str
) -> None:
    """Test that a tool exiting with an error is a failure carrying its output, with no statistics recorded."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("siqs", 90, 1, 1.0)
    failure = subprocess.CalledProcessError(1, [], "", "something broke\n")
    monkeypatch.setattr("factortool.tools.run_tool", Mock(side_effect=failure), raising=True)

    with pytest.raises(ToolFailure, match=r"something broke$"):
        _ = failing_tools(tmp_path, stats)[tool][0]()

    assert recorded_runs(stats) == 1


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_that_cannot_be_started_fails(tmp_path: Path, tool: str) -> None:
    """Test that a missing tool binary is a failure rather than an unhandled OSError."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("siqs", 90, 1, 1.0)

    with pytest.raises(ToolFailure, match="missing"):
        _ = failing_tools(tmp_path, stats)[tool][0]()

    assert list(tmp_path.glob("*-*/")) == []


@pytest.mark.parametrize(
    ("tool", "output", "problem"),
    [
        ("ecm", "eof; done processing batchfile\n", "no factors"),
        ("rho", yafu_output([7, 13]), "do not multiply back"),
        ("rho", yafu_output([1, C90]), "do not multiply back"),
        ("siqs", yafu_output([C90]), "did not find a factor"),
        ("yafu_direct", "", "no factors"),
        ("yafu_direct", yafu_output([C90]), "did not find a factor"),
        ("nfs_cado", "Error occurred, terminating", "could not be understood"),
        ("nfs_cado", "", "no factors"),
        ("nfs_cado", "7 13", "do not multiply back"),
        ("nfs_cado", str(C90), "did not find a factor"),
    ],
)
def test_tool_returning_an_unusable_result_fails_without_recording_statistics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool: str, output: str, problem: str
) -> None:
    """Test that output that isn't a factorization of the number is a failure, with no statistics recorded."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("siqs", 90, 1, 1.0)
    run_tool = Mock(return_value=subprocess.CompletedProcess([], 0, output, ""))
    monkeypatch.setattr("factortool.tools.run_tool", run_tool, raising=True)

    with pytest.raises(ToolFailure, match=problem):
        _ = failing_tools(tmp_path, stats)[tool][0]()

    assert recorded_runs(stats) == 1


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_failing_repeatedly_raises_its_tool_error(tmp_path: Path, tool: str) -> None:
    """Test that a tool that keeps failing raises that tool's error, carrying its exit status."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("siqs", 90, 1, 1.0)
    call, error_type = failing_tools(tmp_path, stats)[tool]

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        with pytest.raises(ToolFailure):
            _ = call()

    with pytest.raises(error_type, match="broken installation"):
        _ = call()


def test_tool_start_failure_names_the_tool_when_the_platform_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a failure to start a tool names its binary, as Windows reports the error without one."""
    failure = FileNotFoundError(2, "The system cannot find the file specified")
    monkeypatch.setattr("factortool.tools.subprocess.Popen", Mock(side_effect=failure), raising=True)

    with pytest.raises(FileNotFoundError, match="missing-tool"):
        _ = run_tool(["missing-tool"], tmp_path, 1)


def test_a_tool_failure_sets_the_number_aside(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a number a tool fails on is left as it was, with nothing further to do for it this run."""
    number = make_number(C90)
    factor_yafu = Mock(side_effect=ToolFailure("YAFU failed"))
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)

    number.factor_rho()

    assert number.tool_failed
    assert not number.active
    assert not number.ecm_needed()
    assert number.composite_factors == [C90]
    assert number.started
    assert number.outcome is Outcome.FAILED


def test_a_number_no_method_was_run_on_is_not_started() -> None:
    """Test that a number is not counted as worked on merely for resuming with factors from a previous run."""
    number = resumed_number(15015, [1001, 3, 5])

    assert not number.started
    assert number.outcome is Outcome.NOT_STARTED


def test_a_method_that_finds_nothing_still_starts_the_number(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a number is unfinished, rather than not started, once a method has run on it without success."""
    number = make_number(C90)
    monkeypatch.setattr("factortool.number.factor_yafu", Mock(return_value=[C90]), raising=True)

    number.factor_rho()

    assert number.methods == []
    assert number.outcome is Outcome.UNFINISHED


def test_a_factored_number_is_factored_whatever_else_happened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a number factored just before its assignment ran out is not reported as expired."""
    number = make_number(C90)
    monkeypatch.setattr("factortool.number.factor_yafu", Mock(return_value=C90_SPLIT), raising=True)
    monkeypatch.setattr("factortool.number.is_prime", is_prime_unless_c90, raising=True)

    number.factor_rho()
    number.expires_at = 0.0

    assert number.outcome is Outcome.FACTORED


@pytest.mark.parametrize(
    ("started", "tool_failed", "outcome"),
    [(False, False, Outcome.EXPIRED), (True, False, Outcome.EXPIRED), (True, True, Outcome.FAILED)],
    ids=["skipped", "cut short", "failed"],
)
def test_an_expired_assignment_outranks_everything_but_a_tool_failure(
    started: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    tool_failed: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    outcome: Outcome,
) -> None:
    """Test that an unfactored number whose assignment ran out is expired unless a tool failed on it."""
    number = make_number(C90)
    number.expires_at = 0.0
    number.started = started
    number.tool_failed = tool_failed

    assert number.outcome is outcome


def final_number(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, results: dict[str, list[int] | Exception]) -> Number:
    """Build a 90-digit number whose final methods give the supplied results, slowest first by statistics key.

    Returns:
        Number: The number, with every final method enabled and holding statistics.
    """
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)

    for execution_time, method in enumerate(sorted(results, reverse=True)):
        stats.update_final(method, 90, 1, float(execution_time + 1))

    def run_final(_self: Number, _n: int, method: str) -> list[int]:
        if isinstance(result := results[method], Exception):
            raise result

        return result

    monkeypatch.setattr("factortool.number.Number._run_final", run_final, raising=True)
    monkeypatch.setattr("factortool.number.is_prime", is_prime_unless_c90, raising=True)

    return Number(C90, make_config(cado_nfs_path="cado-nfs.py", use_nfs_cado=True, use_nfs_yafu=True), stats, None)


def test_a_failed_final_method_falls_back_to_the_next_fastest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a composite a final method fails on is finished with another eligible method."""
    results: dict[str, list[int] | Exception] = {
        "siqs": ToolFailure("SIQS failed"),
        "nfs_yafu": [C90],
        "nfs_cado": C90_SPLIT,
    }
    number = final_number(monkeypatch, tmp_path, results)

    assert number.final_method == "siqs"

    number.factor_final()

    assert number.factored
    assert not number.tool_failed
    assert number.methods == ["CADO-NFS"]
    assert sorted(number.prime_factors) == C90_SPLIT


def test_a_number_is_set_aside_once_every_final_method_has_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a composite is left unfactored only after each eligible final method has failed on it."""
    attempted: list[str] = []

    def fail(method: str) -> ToolFailure:
        return ToolFailure(f"{method} failed")

    number = final_number(monkeypatch, tmp_path, {x: fail(x) for x in ("siqs", "nfs_yafu", "nfs_cado")})
    run_final = Number._run_final

    def record(self: Number, n: int, method: str) -> list[int]:
        attempted.append(method)
        return run_final(self, n, method)

    monkeypatch.setattr("factortool.number.Number._run_final", record, raising=True)

    number.factor_final()

    assert attempted == ["siqs", "nfs_yafu", "nfs_cado"]
    assert number.tool_failed
    assert number.composite_factors == [C90]


def test_a_number_is_set_aside_when_no_final_method_splits_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a composite every final method hands back whole is set aside, rather than silently left as it was."""
    number = final_number(monkeypatch, tmp_path, {x: [C90] for x in ("siqs", "nfs_yafu", "nfs_cado")})

    number.factor_final()

    assert number.tool_failed
    assert not number.factored
    assert number.composite_factors == [C90]
    assert number.started
    assert number.outcome is Outcome.FAILED


def test_composite_cofactors_left_by_a_final_method_are_finished(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a composite cofactor a final method returns is itself factored with a final method."""
    number = make_number(3 * 5 * 7)
    calls: list[int] = []

    def run_final(_self: Number, n: int, _method: str) -> list[int]:
        calls.append(n)
        return {105: [3, 35], 35: [5, 7]}[n]

    monkeypatch.setattr("factortool.number.Number._run_final", run_final, raising=True)

    number.factor_final()

    assert calls == [105, 35]
    assert sorted(number.prime_factors) == [3, 5, 7]


def test_an_immediate_final_run_does_not_rename_the_method_for_later_composites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a composite split by ECM is credited to ECM, even after another was finished for statistics."""
    number = make_number(15 * 77)
    number.composite_factors = [15, 77]
    number._stats = Mock(spec_set=FactoringStats, get_ecm_cutoffs=Mock(return_value=ECMCutoffs(None, 5)))

    def factor_ecm(n: int, *_args: object) -> list[int]:
        if n == 15:  # ruff: ignore[magic-value-comparison]
            method = "siqs"
            raise FinalMethodNeeded(method)

        return [7, 11]

    monkeypatch.setattr("factortool.number.factor_ecm", factor_ecm, raising=True)

    def run_final(*_args: object) -> list[int]:
        return [3, 5]

    monkeypatch.setattr("factortool.number.Number._run_final", run_final, raising=True)

    number.factor_ecm(2)

    assert number.methods == ["SIQS", "ECM"]
    assert sorted(number.prime_factors) == [3, 5, 7, 11]


def test_a_factor_equal_to_another_composite_keeps_the_progress_of_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a split producing a composite already present doesn't lose the progress recorded for it."""
    a, b = 1001, 1003
    number = make_number(a * b * b)
    number.composite_factors = [a * b, b]
    number._progress = {b: CompositeProgress(n=b, completed=frozenset({"rho"}), ecm_level=3)}
    factor_yafu = Mock(return_value=[a, b])
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)
    monkeypatch.setattr("factortool.number.is_prime", never_prime, raising=True)

    number.factor_pm1()

    factor_yafu.assert_called_once_with(a * b, "pm1", *number._yafu_args)
    assert sorted(number.composite_factors) == [a, b, b]
    assert number._progress_of(b) == CompositeProgress(n=b, completed=frozenset({"rho", "pm1"}), ecm_level=3)


def test_factors_found_by_yafu_inherit_the_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that YAFU's factor() runs whatever the progress, and passes it on to any composite factors it leaves."""
    a, b = 1001, 1003
    number = make_number(a * b)
    progress = CompositeProgress(n=a * b, completed=frozenset({"tf", "rho", "pm1"}), ecm_level=10)
    number._progress = {a * b: progress}
    monkeypatch.setattr("factortool.number.factor_yafu_direct", Mock(return_value=[a, b]), raising=True)
    monkeypatch.setattr("factortool.number.is_prime", never_prime, raising=True)

    number.factor_yafu_direct()

    assert [number._progress_of(n) for n in (a, b)] == [progress.for_factor(a), progress.for_factor(b)]


def test_ecm_runs_each_composite_up_to_its_own_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a small composite stops ECM at its own cutoff rather than following the largest one."""
    small, large = 10**30 + 1, 10**50 + 1
    number = make_number(small * large)
    number.composite_factors = [small, large]
    targets = {31: 3, 51: 5}

    def get_ecm_cutoffs(digits: int, *_args: object) -> ECMCutoffs:
        return ECMCutoffs(None, targets[digits])

    number._stats = Mock(spec_set=FactoringStats, get_ecm_cutoffs=Mock(side_effect=get_ecm_cutoffs))
    calls: list[tuple[int, int]] = []

    def factor_ecm(n: int, level: int, *_args: object) -> list[int]:
        calls.append((n, level))
        return [n]

    monkeypatch.setattr("factortool.number.factor_ecm", factor_ecm, raising=True)
    monkeypatch.setattr("factortool.number.is_prime", never_prime, raising=True)

    for level in range(2, 7):
        assert number.ecm_needed(level) == (level <= max(targets.values()))
        number.factor_ecm(level)

    assert calls == [(small, 2), (large, 2), (small, 3), (large, 3), (large, 4), (large, 5)]
    assert not number.ecm_needed()
    assert [number._progress_of(n).ecm_level for n in (small, large)] == [3, 5]


def test_factors_of_a_composite_inherit_its_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that composites split from another keep the work done on it, so it is not repeated on them."""
    a, b, c = 1001, 1003, 1007
    number = make_number(a * b * c)
    splits = {a * b * c: [a, b * c], b * c: [b, c]}
    calls: list[tuple[str, int]] = []

    def factor_yafu(n: int, method: str, *_args: object) -> list[int]:
        calls.append((method, n))
        return [n] if method == "pm1" else splits[n]

    def factor_ecm(n: int, level: int, *_args: object) -> list[int]:
        calls.append((f"ecm{level}", n))
        return splits.get(n, [n])

    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)
    monkeypatch.setattr("factortool.number.factor_ecm", factor_ecm, raising=True)
    monkeypatch.setattr("factortool.number.is_prime", never_prime, raising=True)
    number._stats = Mock(spec_set=FactoringStats, get_ecm_cutoffs=Mock(return_value=ECMCutoffs(None, 3)))

    number.factor_rho()
    number.factor_rho()
    number.factor_pm1()
    number.factor_ecm(2)
    number.factor_ecm(3)

    assert calls == [
        ("rho", a * b * c),
        ("pm1", a),
        ("pm1", b * c),
        ("ecm2", a),
        ("ecm2", b * c),
        ("ecm3", a),
        ("ecm3", b),
        ("ecm3", c),
    ]
    assert {n: number._progress_of(n) for n in number.composite_factors} == {
        a: CompositeProgress(n=a, completed=frozenset({"rho", "pm1"}), ecm_level=3),
        b: CompositeProgress(n=b, completed=frozenset({"rho", "pm1"}), ecm_level=3),
        c: CompositeProgress(n=c, completed=frozenset({"rho", "pm1"}), ecm_level=3),
    }


def test_yafu_mode_ignores_but_keeps_the_saved_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that YAFU's factor() runs on composites whatever standard mode did, keeping the progress if it fails."""
    progress = (CompositeProgress(n=1001, completed=frozenset({"tf", "rho", "pm1"}), ecm_level=10),)
    number = Number(15015, make_config(), FactoringStats(NO_STATS_PATH, read_only=True), None, (3, 5, 1001), progress)
    factor_yafu_direct = Mock(side_effect=ToolFailure("YAFU failed"))
    monkeypatch.setattr("factortool.number.factor_yafu_direct", factor_yafu_direct, raising=True)

    number.factor_yafu_direct()

    factor_yafu_direct.assert_called_once_with(1001, *number._yafu_args)
    assert number.progress == progress


def test_progress_without_a_new_factor_is_checkpointed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a method finding nothing still records that it ran, while a skipped method records nothing."""
    checkpoint = Mock()
    number = Number(1001, make_config(), FactoringStats(NO_STATS_PATH, read_only=True), Mock(checkpoint=checkpoint))
    monkeypatch.setattr("factortool.number.factor_yafu", Mock(return_value=[1001]), raising=True)

    number.factor_rho()
    number.factor_rho()

    checkpoint.assert_called_once_with(number)
    assert number.progress == (CompositeProgress(n=1001, completed=frozenset({"rho"})),)
