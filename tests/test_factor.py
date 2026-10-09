# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Factoring tests for factortool."""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    import pytest

from factortool.number import Number, factor_tf, factor_yafu
from factortool.stats import FactoringStats

from .helpers import make_config, make_number

# Composites on either side of the YAFU NFS minimum.
C84 = 10**83 + 1
C90 = 10**89 + 7 * 13


def test_tf() -> None:
    """Test trial factoring."""
    n = 15825810
    stats = FactoringStats(Path("stats.json"), read_only=True)
    assert factor_tf(n, stats) == [2, 3, 5, 7, 11, 13, 17, 31]


def test_small_cofactors_use_statistical_ecm_cutoffs() -> None:
    """Test that a composite too small for CADO-NFS stops ECM at the learned cutoff rather than the maximum level."""
    number = make_number(15 * 10**50 + 7)
    stats = Mock(spec=FactoringStats)
    stats.get_ecm_cutoffs.return_value = (2, 5)
    number._stats = stats
    number.composite_factors = [10**40 + 1]
    number._ecm_level = 4

    assert number.ecm_needed

    stats.get_ecm_cutoffs.assert_called_once_with(41, number._config.max_threads, ("siqs",))

    number._ecm_level = 5
    assert not number.ecm_needed


def test_ecm_cutoff_follows_the_latest_statistics() -> None:
    """Test that the ECM cutoff is reevaluated as statistics arrive, and that a finished number stays finished."""
    number = make_number(10**59 + 1)
    stats = Mock(spec=FactoringStats)
    number._stats = stats
    number._ecm_level = 10

    stats.get_ecm_cutoffs.return_value = (None, 20)
    assert number.ecm_needed

    stats.get_ecm_cutoffs.return_value = (8, 10)
    assert not number.ecm_needed

    # Resuming would skip the levels the engine ran in the meantime.
    stats.get_ecm_cutoffs.return_value = (8, 20)
    assert not number.ecm_needed


def test_yafu_nfs_forces_nfs_and_records_its_own_statistics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that YAFU NFS is forced below YAFU's QS/NFS crossover and recorded separately from CADO-NFS."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)
    output = f"P1 = 7\nP1 = 13\nC88 = {10**87 + 1}\n"
    run_tool = Mock(return_value=subprocess.CompletedProcess([], 0, output, ""))
    monkeypatch.setattr("factortool.number.run_tool", run_tool, raising=True)
    monkeypatch.setenv("FACTORTOOL_TEST", "1")

    factors = factor_yafu(C90, "nfs", 4, make_config(work_path=tmp_path).yafu_paths, stats)

    cmd = run_tool.call_args.args[0]
    assert factors == [7, 13, 10**87 + 1]
    assert cmd[1] == f"nfs({C90})"
    assert cmd[cmd.index("-xover") + 1] == "1"
    assert run_tool.call_args.kwargs["env"]["FACTORTOOL_TEST"] == "1"
    assert stats.get_final_stats("nfs_yafu", 90, 4)[0] == 1
    assert stats.get_final_stats("nfs_cado", 90, 4)[0] == 0


def test_yafu_nfs_is_not_run_below_its_minimum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that YAFU NFS is never started on a composite small enough to stall its polynomial selection."""
    run_tool = Mock()
    monkeypatch.setattr("factortool.number.run_tool", run_tool, raising=True)

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

    def factor_yafu(n: int, method: str, *_args: object) -> list[int]:
        calls.append((n, method + "_yafu"))
        return [n]

    def factor_nfs_cado(n: int, *_args: object) -> list[int]:
        calls.append((n, "nfs_cado"))
        return [n]

    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)
    monkeypatch.setattr("factortool.number.factor_nfs_cado", factor_nfs_cado, raising=True)

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
