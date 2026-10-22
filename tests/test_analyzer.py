# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the analyzer."""

from __future__ import annotations

import json
import sys

from typing import TYPE_CHECKING
from unittest.mock import Mock, call

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from factortool.analyzer.main import NO_DATA_EXIT_STATUS, main
from factortool.constants import ECM_MIN_LEVEL
from factortool.exit_status import ExitStatus
from factortool.stats import FactoringStats

DIGITS = 50


def run_analyzer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    stats: FactoringStats,
    *arguments: str,
) -> list[str]:
    """Run the analyzer for DIGITS against the given statistics, with any extra arguments.

    Returns:
        list[str]: The lines of the ECM table, without their leading and trailing whitespace.
    """
    stats.save_data()

    config = {"backend": "factordb", "max_threads": 1, "state_path": str(tmp_path), "yafu_path": "yafu"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv", ["analyzer", "--config_path", str(config_path), "--digits", str(DIGITS), *arguments]
    )
    monkeypatch.setattr("factortool.analyzer.main.setup_logger", Mock(), raising=True)

    main()

    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]

    return lines[lines.index("Stopping ECM after doing the given level averages:") + 2 :]


def add_ecm_runs(stats: FactoringStats, level: int, runs: int = 1024, threads: int = 1) -> None:
    """Record ECM runs at a level that take one second and find a factor half of the time."""
    for run in range(runs):
        stats.update_ecm(DIGITS, level, threads, 1.0, success=run % 2 == 0)


@pytest.mark.parametrize("digits", ["0", "-5", "many"])
def test_an_unusable_digit_count_is_rejected(monkeypatch: pytest.MonkeyPatch, digits: str) -> None:
    """Test that a digit count no number could have is refused, with the status of a configuration error."""
    monkeypatch.setattr(sys, "argv", ["analyzer", "--digits", digits])
    monkeypatch.setattr("factortool.analyzer.main.setup_logger", Mock(), raising=True)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == ExitStatus.INVALID_INPUT


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], [call(None), call("WARNING")]), (["--log_level", "debug"], [call("DEBUG"), call("DEBUG")])],
    ids=["configured", "overridden"],
)
def test_the_log_level_option_overrides_the_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], expected: list[object]
) -> None:
    """Test that logging is set up before the configuration is read, then at the configured level unless overridden."""
    config = {
        "backend": "factordb",
        "log_level": "warning",
        "max_threads": 1,
        "state_path": str(tmp_path),
        "yafu_path": "yafu",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    setup_logger = Mock()
    monkeypatch.setattr(sys, "argv", ["analyzer", "--config_path", str(config_path), "--digits", str(DIGITS), *argv])
    monkeypatch.setattr("factortool.analyzer.main.setup_logger", setup_logger, raising=True)

    # Without any statistics, the analyzer exits once the configuration has been read.
    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == NO_DATA_EXIT_STATUS
    assert setup_logger.call_args_list == expected


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


def test_the_thread_count_option_selects_the_statistics_analyzed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Test that the statistics gathered with the given thread count are analyzed, rather than the configured one."""
    stats = FactoringStats(tmp_path / "stats.json")
    stats.update_final("siqs", DIGITS, 4, 10.0)
    add_ecm_runs(stats, ECM_MIN_LEVEL, threads=4)

    rows = [line.split() for line in run_analyzer(monkeypatch, capsys, tmp_path, stats, "--max_threads", "4")]

    assert rows[:2] == [
        ["none", "-", "-", "-", "10.000s", "N/A"],
        ["2", "1024", "1.000s", "50.000%", "6.000s", "N/A", "<-", "optimal"],
    ]


@pytest.mark.parametrize("max_threads", ["0", "-3", "many"])
def test_an_unusable_thread_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, max_threads: str
) -> None:
    """Test that a thread count the configuration rejects is refused before the statistics are opened."""
    config = {"backend": "factordb", "max_threads": 1, "state_path": str(tmp_path), "yafu_path": "yafu"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    factoring_stats = Mock()
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyzer", "--config_path", str(config_path), "--digits", str(DIGITS), "--max_threads", max_threads],
    )
    monkeypatch.setattr("factortool.analyzer.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.analyzer.main.FactoringStats", factoring_stats, raising=True)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == ExitStatus.INVALID_INPUT
    factoring_stats.assert_not_called()
