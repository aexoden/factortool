# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the analyzer."""

from __future__ import annotations

import json
import sys

from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from factortool.analyzer.main import main
from factortool.constants import ECM_MIN_LEVEL
from factortool.stats import FactoringStats

DIGITS = 50


def run_analyzer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, stats: FactoringStats
) -> list[str]:
    """Run the analyzer for DIGITS against the given statistics.

    Returns:
        list[str]: The lines of the ECM table, without their leading and trailing whitespace.
    """
    stats.save_data()

    config = {"backend": "factordb", "max_threads": 1, "stats_path": str(tmp_path / "stats.json"), "yafu_path": "yafu"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["analyzer", "--config_path", str(config_path), "--digits", str(DIGITS)])
    monkeypatch.setattr("factortool.analyzer.main.setup_logger", Mock(), raising=True)

    main()

    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]

    return lines[lines.index("Stopping ECM after doing the given level averages:") + 2 :]


def add_ecm_runs(stats: FactoringStats, level: int, runs: int = 1024) -> None:
    """Record ECM runs at a level that take one second and find a factor half of the time."""
    for run in range(runs):
        stats.update_ecm(DIGITS, level, 1, 1.0, success=run % 2 == 0)


@pytest.mark.parametrize("digits", ["0", "-5", "many"])
def test_an_unusable_digit_count_is_rejected(monkeypatch: pytest.MonkeyPatch, digits: str) -> None:
    """Test that a digit count no number could have is refused, with the status of a configuration error."""
    monkeypatch.setattr(sys, "argv", ["analyzer", "--digits", digits])
    monkeypatch.setattr("factortool.analyzer.main.setup_logger", Mock(), raising=True)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 1


def test_ecm_levels_beyond_a_gap_are_shown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Test that a level with no data is marked as skipped, rather than ending the table."""
    stats = FactoringStats(tmp_path / "stats.json")
    stats.update_final("siqs", DIGITS, 1, 10.0)
    add_ecm_runs(stats, ECM_MIN_LEVEL)
    add_ecm_runs(stats, ECM_MIN_LEVEL + 1)
    add_ecm_runs(stats, ECM_MIN_LEVEL + 5, runs=4)

    rows = [line.split() for line in run_analyzer(monkeypatch, capsys, tmp_path, stats)]

    # The first unmeasured level is being done to collect data, so it is shown as the current cutoff.
    assert rows[:6] == [
        ["none", "-", "-", "-", "10.000s", "N/A"],
        ["2", "1024", "1.000s", "50.000%", "6.000s", "N/A"],
        ["3", "1024", "1.000s", "50.000%", "4.000s", "N/A", "<-", "optimal"],
        ["..."],
        ["7", "4", "1.000s", "50.000%", "N/A", "N/A"],
        ["..."],
    ]
    assert rows[6][:2] == ["13", "0"]
    assert rows[6][-2:] == ["<-", "current"]


def test_ecm_levels_are_shown_without_data_for_the_first_levels(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Test that data starting partway through the levels is analyzed from there, noting what is missing."""
    stats = FactoringStats(tmp_path / "stats.json")
    stats.update_final("siqs", DIGITS, 1, 10.0)
    add_ecm_runs(stats, 20)
    add_ecm_runs(stats, 21)

    lines = run_analyzer(monkeypatch, capsys, tmp_path, stats)

    assert [line.split() for line in lines[:4]] == [
        ["none", "-", "-", "-", "10.000s", "N/A"],
        ["..."],
        ["20", "1024", "1.000s", "50.000%", "6.000s", "N/A"],
        ["21", "1024", "1.000s", "50.000%", "4.000s", "N/A", "<-", "optimal"],
    ]
    assert "There is no ECM data below level 20. The From ECM times assume the levels before it are done." in lines
    assert "The Overall times can't be estimated without data for those levels." in lines
    assert "Optimal ECM cutoff: 21" in lines
