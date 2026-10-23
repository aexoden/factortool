# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for running the external factoring tools."""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time

from pathlib import Path
from unittest.mock import Mock

import pytest

from factortool.constants import MAX_CONSECUTIVE_TOOL_FAILURES
from factortool.tools import (
    CadoNfsError,
    ToolDeadlineExceeded,
    ToolFailure,
    ToolFailures,
    ToolState,
    YafuError,
    parse_yafu_factors,
    run_cado_nfs,
    run_tool,
    run_yafu,
    tool_deadline,
    tool_failures,
)

from .helpers import TOOL_STOP_TIMEOUT, heartbeat_stopped, make_config, make_tool_with_helper


def test_yafu_factors_are_read_whatever_their_primality_label() -> None:
    """Test that factors YAFU labels as probable primes or of unknown primality are not dropped."""
    output = "***factors found***\nP1 = 7\nPRP2 = 11\nC2 = 15\nU2 = 13\n\nans = 1\n"

    assert parse_yafu_factors(output) == [7, 11, 13, 15]


def test_only_the_last_list_of_yafu_factors_is_read() -> None:
    """Test that a list of factors YAFU printed earlier in its output is not counted a second time."""
    output = "***factors found***\nP1 = 7\nC2 = 15\n\n***factors found***\nP1 = 3\nP1 = 5\nP1 = 7\n"

    assert parse_yafu_factors(output) == [3, 5, 7]


def test_yafu_output_without_factors_has_none() -> None:
    """Test that output from a YAFU that ignored its expression yields no factors."""
    assert parse_yafu_factors("no variable indicator (@)\n\n\neof; done processing batchfile\n") == []


def test_yafu_is_given_its_expression_on_standard_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that the expression is also passed on standard input, where YAFU looks when not run from a terminal."""
    run_tool = Mock(return_value=subprocess.CompletedProcess([], 0, "***factors found***\nP1 = 3\nP1 = 5\n", ""))
    monkeypatch.setattr("factortool.tools.run_tool", run_tool, raising=True)

    factors, _ = run_yafu("Rho", 15, "rho(15)", 1, make_config(work_path=tmp_path).yafu_paths)

    assert factors == [3, 5]
    assert run_tool.call_args.args[0][1] == "rho(15)"
    assert run_tool.call_args.kwargs["stdin"] == "rho(15)\n"


def test_cado_nfs_works_in_the_managed_working_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that CADO-NFS is told to use its working directory, as it otherwise creates one of its own elsewhere.

    CADO-NFS refuses a relative path, so the directory is given as an absolute one even when work_path is relative.
    """
    run_tool = Mock(return_value=subprocess.CompletedProcess([], 0, "3 5\n", ""))
    monkeypatch.setattr("factortool.tools.run_tool", run_tool, raising=True)
    monkeypatch.chdir(tmp_path)

    factors, _ = run_cado_nfs("CADO-NFS", 15, 2, Path("cado-nfs.py"), Path("work"))

    cmd = run_tool.call_args.args[0]
    work_dir = Path(cmd[cmd.index("--workdir") + 1])

    assert factors == [3, 5]
    assert work_dir.is_absolute()
    assert work_dir == run_tool.call_args.args[1].absolute()
    assert work_dir.parent == tmp_path / "work"
    assert "stdin" not in run_tool.call_args.kwargs


def test_a_tool_given_no_input_reads_from_the_null_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a tool given no input doesn't inherit the terminal, which it would be stopped for reading."""
    popen = Mock(wraps=subprocess.Popen)
    monkeypatch.setattr("factortool.tools.subprocess.Popen", popen, raising=True)

    result = run_tool([sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"], tmp_path, 1)

    assert popen.call_args.kwargs["stdin"] == subprocess.DEVNULL
    assert result.stdout.strip() == "''"


def fail(failures: ToolFailures, method: str, tool: type[YafuError | CadoNfsError] = YafuError) -> None:
    """Record a failure that is expected to be an isolated one."""
    with pytest.raises(ToolFailure, match="it broke"):
        failures.failed(tool, method, "it broke")


def test_a_success_ends_a_run_of_failures() -> None:
    """Test that failures only count toward a broken installation while nothing succeeds in between."""
    failures = ToolFailures()

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES):
        fail(failures, "ECM")
        fail(failures, "ECM")
        failures.succeeded("ECM")


def test_failures_are_counted_separately_for_each_method() -> None:
    """Test that a method that always fails is noticed even while the same tool's other methods succeed."""
    failures = ToolFailures()

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        fail(failures, "YAFU NFS")
        failures.succeeded("SIQS")

    with pytest.raises(YafuError, match=r"it broke \(YAFU NFS has failed 3 times in a row"):
        failures.failed(YafuError, "YAFU NFS", "it broke")


def test_a_run_of_failures_continues_from_one_run_to_the_next(tmp_path: Path) -> None:
    """Test that a tool run too rarely to fail repeatedly within a batch is still noticed across batches."""
    state_path = tmp_path / "tool_state.json"

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        failures = ToolFailures()
        failures.load(state_path)
        failures.succeeded("ECM")
        fail(failures, "CADO-NFS", CadoNfsError)
        failures.save(state_path)

    failures = ToolFailures()
    failures.load(state_path)

    with pytest.raises(CadoNfsError, match="CADO-NFS has failed 3 times in a row"):
        failures.failed(CadoNfsError, "CADO-NFS", "it broke")


def test_a_broken_tool_is_reported_again_on_its_next_failure(tmp_path: Path) -> None:
    """Test that a method reported as broken gets no fresh allowance of failures in the next run."""
    state_path = tmp_path / "tool_state.json"
    failures = ToolFailures()

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        fail(failures, "ECM")

    with pytest.raises(YafuError):
        failures.failed(YafuError, "ECM", "it broke")

    failures.save(state_path)
    failures = ToolFailures()
    failures.load(state_path)

    with pytest.raises(YafuError, match="ECM has failed 4 times in a row"):
        failures.failed(YafuError, "ECM", "it broke")


def test_a_success_in_a_later_run_ends_a_saved_run_of_failures(tmp_path: Path) -> None:
    """Test that a method that works again is forgiven the failures saved by earlier runs."""
    state_path = tmp_path / "tool_state.json"
    failures = ToolFailures()

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        fail(failures, "ECM")

    failures.save(state_path)
    failures = ToolFailures()
    failures.load(state_path)
    failures.succeeded("ECM")
    failures.save(state_path)

    assert ToolState.model_validate_json(state_path.read_text(encoding="utf-8")).streaks == {}

    failures = ToolFailures()
    failures.load(state_path)
    fail(failures, "ECM")


@pytest.mark.parametrize("content", [b"not json", b'{"streaks": {"ECM": -1}}', b'{"streaks": []}', b"\xff"])
def test_unreadable_tool_state_is_ignored(tmp_path: Path, content: bytes) -> None:
    """Test that a damaged state file costs the saved failures rather than the run."""
    state_path = tmp_path / "tool_state.json"
    state_path.write_bytes(content)
    failures = ToolFailures()
    fail(failures, "ECM")
    fail(failures, "ECM")

    failures.load(state_path)

    fail(failures, "ECM")


def test_resetting_forgets_earlier_failures() -> None:
    """Test that a reset clears every run of failures."""
    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        fail(tool_failures, "ECM")

    tool_failures.reset()

    fail(tool_failures, "ECM")


def test_a_tool_still_running_at_its_deadline_is_killed_with_its_helpers(tmp_path: Path) -> None:
    """Test that the deadline kills the whole process tree, rather than waiting for the tool to finish."""
    heartbeat = tmp_path / "heartbeat"
    start = time.monotonic()

    with tool_deadline(start + 1.0), pytest.raises(ToolDeadlineExceeded, match="the tool"):
        run_tool(make_tool_with_helper(heartbeat), tmp_path, 1, description="the tool")

    assert time.monotonic() - start < TOOL_STOP_TIMEOUT
    assert heartbeat_stopped(heartbeat)


def test_a_tool_that_finishes_before_its_deadline_is_unaffected(tmp_path: Path) -> None:
    """Test that a deadline does not disturb the input or output of a tool that finishes in time."""
    script = "import sys; print(sys.stdin.read().upper())"

    with tool_deadline(time.monotonic() + TOOL_STOP_TIMEOUT):
        result = run_tool([sys.executable, "-c", script], tmp_path, 1, stdin="factors")

    assert result.stdout.strip() == "FACTORS"


def test_a_deadline_only_applies_inside_its_block(tmp_path: Path) -> None:
    """Test that a tool started after the block has ended is not held to the block's deadline."""
    with tool_deadline(time.monotonic() - 1.0):
        pass

    assert run_tool([sys.executable, "-c", "print(1)"], tmp_path, 1).stdout.strip() == "1"


def test_a_long_running_tool_is_warned_about_until_it_finishes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that a tool that keeps running is reported periodically, so that a hung tool can be noticed."""
    warning = Mock()
    monkeypatch.setattr("factortool.tools.STILL_RUNNING_INTERVAL", 0.2, raising=True)
    monkeypatch.setattr("factortool.tools.logger.warning", warning, raising=True)
    script = "import sys, time; time.sleep(0.7); print(sys.stdin.read())"

    result = run_tool([sys.executable, "-c", script], tmp_path, 1, stdin="factors", description="SIQS on 15")

    assert result.stdout.strip() == "factors"
    assert len(warning.call_args_list) >= 2  # ruff: ignore[magic-value-comparison]
    assert warning.call_args.args[1] == "SIQS on 15"
