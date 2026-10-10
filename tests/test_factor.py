# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Factoring tests for factortool."""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from factortool.constants import MAX_CONSECUTIVE_TOOL_FAILURES
from factortool.number import (
    FinalMethodNeeded,
    Number,
    factor_ecm,
    factor_nfs_cado,
    factor_tf,
    factor_yafu,
    factor_yafu_direct,
)
from factortool.stats import ECMCutoffs, FactoringStats
from factortool.tools import CadoNfsError, ToolError, ToolFailure, YafuError, run_tool

from .helpers import make_config, make_number

# Composites on either side of the YAFU NFS minimum, each with a small factor.
C84 = 10**83 + 1
C84_SPLIT = [11, C84 // 11]
C90 = 10**89 + 7 * 13
C90_SPLIT = [101, C90 // 101]


def yafu_output(factors: list[int]) -> str:
    """Build the part of YAFU's output that lists the given factors.

    Returns:
        str: The list of factors, in YAFU's format.
    """
    return "\n\n***factors found***\n" + "".join(f"C{len(str(x))} = {x}\n" for x in factors) + "\nans = 1\n"


def test_tf() -> None:
    """Test trial factoring."""
    n = 15825810
    stats = FactoringStats(Path("stats.json"), read_only=True)
    assert factor_tf(n, stats) == [2, 3, 5, 7, 11, 13, 17, 31]


def test_small_cofactors_use_statistical_ecm_cutoffs() -> None:
    """Test that a composite too small for CADO-NFS stops ECM at the learned cutoff rather than the maximum level."""
    number = make_number(15 * 10**50 + 7)
    stats = Mock(spec=FactoringStats)
    stats.get_ecm_cutoffs.return_value = ECMCutoffs(2, 5)
    number._stats = stats
    number.composite_factors = [10**40 + 1]
    number._ecm_level = 4

    assert number.ecm_needed

    stats.get_ecm_cutoffs.assert_called_once_with(41, number._config.max_threads, ("siqs",))

    number._ecm_level = 5
    assert not number.ecm_needed


def test_no_ecm_is_needed_when_the_cutoff_is_none() -> None:
    """Test that a number that has done no ECM does none when the cutoff says none is worthwhile."""
    number = make_number(10**59 + 1)
    stats = Mock(spec=FactoringStats)
    stats.get_ecm_cutoffs.return_value = ECMCutoffs(0, 0)
    number._stats = stats

    assert not number.ecm_needed


def test_ecm_cutoff_follows_the_latest_statistics() -> None:
    """Test that the ECM cutoff is reevaluated as statistics arrive, and that a finished number stays finished."""
    number = make_number(10**59 + 1)
    stats = Mock(spec=FactoringStats)
    number._stats = stats
    number._ecm_level = 10

    stats.get_ecm_cutoffs.return_value = ECMCutoffs(None, 20)
    assert number.ecm_needed

    stats.get_ecm_cutoffs.return_value = ECMCutoffs(8, 10)
    assert not number.ecm_needed

    # Resuming would skip the levels the engine ran in the meantime.
    stats.get_ecm_cutoffs.return_value = ECMCutoffs(8, 20)
    assert not number.ecm_needed


def test_yafu_nfs_forces_nfs_and_records_its_own_statistics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that YAFU NFS is forced below YAFU's QS/NFS crossover and recorded separately from CADO-NFS."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    run_tool = Mock(return_value=subprocess.CompletedProcess([], 0, yafu_output(C90_SPLIT), ""))
    monkeypatch.setattr("factortool.tools.run_tool", run_tool, raising=True)

    factors = factor_yafu(C90, "nfs", 4, make_config(work_path=tmp_path).yafu_paths, stats)

    cmd = run_tool.call_args.args[0]
    assert factors == C90_SPLIT
    assert cmd[1] == f"nfs({C90})"
    assert cmd[cmd.index("-xover") + 1] == "1"
    assert stats.get_final_stats("nfs_yafu", 90, 4)[0] == 1
    assert stats.get_final_stats("nfs_cado", 90, 4)[0] == 0


def test_yafu_nfs_is_not_run_below_its_minimum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that YAFU NFS is never started on a composite small enough to stall its polynomial selection."""
    run_tool = Mock()
    monkeypatch.setattr("factortool.tools.run_tool", run_tool, raising=True)

    stats = FactoringStats(tmp_path / "stats.json", read_only=True)

    assert factor_yafu(C84, "nfs", 4, make_config().yafu_paths, stats) == [C84]
    run_tool.assert_not_called()


def test_final_factoring_chooses_a_method_for_each_composite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that each remaining composite is factored with the fastest final method eligible for its own size."""
    config = make_config(use_nfs_cado=True, use_nfs_yafu=True)
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
    monkeypatch.setattr("factortool.number.is_prime", lambda n: n not in splits, raising=True)

    assert number.final_method == "nfs_yafu"

    number.factor_final()

    assert calls == [(C84, "nfs_cado"), (C90, "nfs_yafu")]


def test_ecm_runs_yafu_nfs_immediately_without_statistics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that ECM gives way to an immediate YAFU NFS run while YAFU NFS has no data for the composite's size."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("nfs_cado", 90, 1, 80.0)
    number = Number(C90, make_config(use_nfs_yafu=True, max_siqs_digits=80), stats, None)
    factor_yafu = Mock(return_value=[C90])
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)

    number.factor_ecm(2)

    factor_yafu.assert_called_once_with(C90, "nfs", *number._yafu_args)


def test_tools_inherit_the_environment_with_their_thread_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a tool keeps the surrounding environment, with OpenMP limited to the threads it is allowed."""
    monkeypatch.setenv("FACTORTOOL_TEST", "inherited")
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    script = "import os; print(os.environ['FACTORTOOL_TEST'], os.environ['OMP_NUM_THREADS'], 'PATH' in os.environ)"

    result = run_tool([sys.executable, "-c", script], tmp_path, 3)

    assert result.stdout.split() == ["inherited", "3", "True"]


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
        factor_ecm(C90, 2, config.final_methods, 4, yafu, stats)
    elif tool == "yafu_direct":
        factor_yafu_direct(C90, 4, yafu, stats)
    elif tool == "nfs_cado":
        factor_nfs_cado(C90, 4, config.cado_nfs_path, tmp_path, stats)
    else:
        factor_yafu(C90, tool, 4, yafu, stats)

    cmd = run_tool.call_args.args[0]
    option = "-t" if tool == "nfs_cado" else "-threads"

    # YAFU uses a single thread unless told otherwise.
    assert (int(cmd[cmd.index(option) + 1]) if option in cmd else 1) == threads
    assert run_tool.call_args.args[2] == threads


def failing_tools(tmp_path: Path, stats: FactoringStats) -> dict[str, tuple[Callable[[], object], type[ToolError]]]:
    """Build a call to each tool-backed factoring function, with the error its repeated failure should give.

    Returns:
        dict[str, tuple[Callable[[], object], type[ToolError]]]: The call and error type by name.
    """
    config = make_config(work_path=tmp_path, yafu_path=tmp_path / "missing-yafu", cado_nfs_path=tmp_path / "missing")
    yafu = config.yafu_paths

    return {
        "ecm": (lambda: factor_ecm(C90, 2, config.final_methods, 1, yafu, stats), YafuError),
        "rho": (lambda: factor_yafu(C90, "rho", 1, yafu, stats), YafuError),
        "siqs": (lambda: factor_yafu(C90, "siqs", 1, yafu, stats), YafuError),
        "yafu_direct": (lambda: factor_yafu_direct(C90, 1, yafu, stats), YafuError),
        "nfs_cado": (lambda: factor_nfs_cado(C90, 1, config.cado_nfs_path, tmp_path, stats), CadoNfsError),
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
        failing_tools(tmp_path, stats)[tool][0]()

    assert recorded_runs(stats) == 1


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_that_cannot_be_started_fails(tmp_path: Path, tool: str) -> None:
    """Test that a missing tool binary is a failure rather than an unhandled OSError."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("siqs", 90, 1, 1.0)

    with pytest.raises(ToolFailure, match="missing"):
        failing_tools(tmp_path, stats)[tool][0]()

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
        failing_tools(tmp_path, stats)[tool][0]()

    assert recorded_runs(stats) == 1


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_failing_repeatedly_raises_its_tool_error(tmp_path: Path, tool: str) -> None:
    """Test that a tool that keeps failing raises that tool's error, carrying its exit status."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    stats.update_final("siqs", 90, 1, 1.0)
    call, error_type = failing_tools(tmp_path, stats)[tool]

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        with pytest.raises(ToolFailure):
            call()

    with pytest.raises(error_type, match="broken installation") as raised:
        call()

    assert raised.value.exit_status == {YafuError: 5, CadoNfsError: 4}[error_type]


def test_tool_start_failure_names_the_tool_when_the_platform_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a failure to start a tool names its binary, as Windows reports the error without one."""
    failure = FileNotFoundError(2, "The system cannot find the file specified")
    monkeypatch.setattr("factortool.tools.subprocess.Popen", Mock(side_effect=failure), raising=True)

    with pytest.raises(FileNotFoundError, match="missing-tool"):
        run_tool(["missing-tool"], tmp_path, 1)


def test_a_tool_failure_sets_the_number_aside(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a number a tool fails on is left as it was, with nothing further to do for it this run."""
    number = make_number(C90)
    factor_yafu = Mock(side_effect=ToolFailure("YAFU failed"))
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)

    number.factor_rho()

    assert number.tool_failed
    assert not number.active
    assert not number.ecm_needed
    assert number.composite_factors == [C90]
    assert not number.attempted


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
    monkeypatch.setattr("factortool.number.is_prime", lambda n: n != C90, raising=True)

    return Number(C90, make_config(use_nfs_cado=True, use_nfs_yafu=True), stats, None)


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

    def factor_ecm(n: int, *_args: object) -> list[int]:
        if n == 15:  # ruff: ignore[magic-value-comparison]
            method = "siqs"
            raise FinalMethodNeeded(method)

        return [7, 11]

    monkeypatch.setattr("factortool.number.factor_ecm", factor_ecm, raising=True)
    monkeypatch.setattr("factortool.number.Number._run_final", lambda *_args: [3, 5], raising=True)

    number.factor_ecm(2)

    assert number.methods == ["SIQS", "ECM"]
    assert sorted(number.prime_factors) == [3, 5, 7, 11]
