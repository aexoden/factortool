# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for running the external factoring tools."""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]

from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from factortool.constants import MAX_CONSECUTIVE_TOOL_FAILURES
from factortool.tools import (
    CadoNfsError,
    ToolFailure,
    ToolFailures,
    YafuError,
    parse_yafu_factors,
    run_yafu,
    tool_failures,
)

from .helpers import make_config


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
        failures.succeeded(YafuError, "ECM")


def test_failures_are_counted_separately_for_each_method() -> None:
    """Test that a method that always fails is noticed even while the same tool's other methods succeed."""
    failures = ToolFailures()

    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        fail(failures, "YAFU NFS")
        failures.succeeded(YafuError, "SIQS")

    with pytest.raises(YafuError, match=r"it broke \(YAFU NFS has failed 3 times in a row"):
        failures.failed(YafuError, "YAFU NFS", "it broke")


def test_a_tool_that_never_succeeded_is_reported() -> None:
    """Test that a tool failing every time it was run is an error, however few times that was."""
    failures = ToolFailures()
    failures.succeeded(YafuError, "ECM")
    fail(failures, "CADO-NFS", CadoNfsError)

    with pytest.raises(CadoNfsError, match=r"CADO-NFS failed every time it was run \(1 time\)") as raised:
        failures.check()

    assert raised.value.exit_status == CadoNfsError.exit_status


def test_a_tool_that_failed_and_succeeded_is_not_reported() -> None:
    """Test that an isolated failure of a tool that otherwise works is not an error."""
    failures = ToolFailures()
    fail(failures, "Rho")
    failures.succeeded(YafuError, "P-1")

    failures.check()


def test_resetting_forgets_earlier_failures() -> None:
    """Test that the failures of one run do not count toward the next."""
    for _ in range(MAX_CONSECUTIVE_TOOL_FAILURES - 1):
        fail(tool_failures, "ECM")

    tool_failures.reset()

    fail(tool_failures, "ECM")
    tool_failures.reset()
    tool_failures.check()
