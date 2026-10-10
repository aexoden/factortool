# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""External tool management for factortool."""

from __future__ import annotations

import contextlib
import math
import os
import re
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
import time

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from pathlib import Path

from factortool.constants import MAX_CONSECUTIVE_TOOL_FAILURES
from factortool.interrupt import Interrupted
from factortool.util import get_work_dir

if TYPE_CHECKING:
    from factortool.config import YafuPaths


class ToolFailure(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Exception raised when a single run of an external tool fails."""


class ToolError(Exception):
    """Exception raised when an external tool fails repeatedly."""

    exit_status: int
    tool_name: str


class YafuError(ToolError):
    """Exception raised when YAFU continues to fail."""

    exit_status = 5
    tool_name = "YAFU"


class CadoNfsError(ToolError):
    """Exception raised when CADO-NFS continues to fail."""

    exit_status = 4
    tool_name = "CADO-NFS"


class ToolFailures:
    """Record of how the external tools have fared."""

    def __init__(self) -> None:
        """Initialize the record."""
        self._lock = threading.Lock()
        self._streaks: dict[str, int] = {}
        self._succeeded: set[type[ToolError]] = set()
        self._failures: dict[type[ToolError], int] = {}

    def reset(self) -> None:
        """Clear all recorded tool outcomes."""
        with self._lock:
            self._streaks.clear()
            self._succeeded.clear()
            self._failures.clear()

    def succeeded(self, tool: type[ToolError], method: str) -> None:
        """Record a run of a tool that returned a usable result for the given factoring method."""
        with self._lock:
            self._streaks[method] = 0
            self._succeeded.add(tool)

    def failed(self, tool: type[ToolError], method: str, message: str) -> NoReturn:
        """Record a failed run of a tool for the given factoring method, and raise the failure.

        Failures are counted per method rather than per tool.

        Raises:
            ToolFailure: If this is an isolated failure.
            ToolError: If the method has failed repeatedly.
        """
        with self._lock:
            streak = self._streaks[method] = self._streaks.get(method, 0) + 1
            self._failures[tool] = self._failures.get(tool, 0) + 1

        if streak >= MAX_CONSECUTIVE_TOOL_FAILURES:
            msg = f"{message} ({method} has failed {streak} times in a row, which suggests a broken installation)"
            raise tool(msg)

        raise ToolFailure(message)

    def check(self) -> None:
        """Check for a tool that failed every time it was run.

        Raises:
            ToolError: If any tool has failed every time it was run.
        """
        with self._lock:
            broken = [(tool, count) for tool, count in self._failures.items() if tool not in self._succeeded]

        for tool, count in broken:
            msg = (
                f"{tool.tool_name} failed every time it was run ({count} time{'s' if count != 1 else ''}), "
                "which suggests a broken installation"
            )
            raise tool(msg)


tool_failures = ToolFailures()


def _describe_tool_failure(error: OSError | subprocess.CalledProcessError) -> str:
    """Describe why a tool failed, preferring its own error output when it ran at all.

    Returns:
        str: The tool's error output, or the reason it could not be started.
    """
    if isinstance(error, subprocess.CalledProcessError):
        return str(error.stderr).strip() or f"exit status {error.returncode}"

    return str(error)


# External tools currently running in any thread.
_running_tools: set[subprocess.Popen[str]] = set()

# External tools that abandon_tools has killed.
_abandoned_tools: set[subprocess.Popen[str]] = set()

_tools_lock = threading.Lock()


def _kill_tool(process: subprocess.Popen[str]) -> None:
    """Kill a tool along with every helper process it may have started."""
    if sys.platform == "win32":
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],  # ruff: ignore[start-process-with-partial-path]
            capture_output=True,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def abandon_tools() -> None:
    """Kill every running external tool."""
    with _tools_lock:
        for process in _running_tools:
            _abandoned_tools.add(process)
            _kill_tool(process)


def run_tool(cmd: list[str], cwd: Path, threads: int, *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run an external tool in its own process group, capturing its output.

    The separate process group ensures that the process isn't immediately killed in the event of a terminal interrupt.

    The tool inherits the environment, apart from OMP_NUM_THREADS, which is set to the number of threads the tool is
    allowed, as anything built with OpenMP otherwise uses every available core.

    Returns:
        subprocess.CompletedProcess[str]: The finished process and its output.

    Raises:
        OSError: If the tool cannot be started.
        subprocess.CalledProcessError: If the tool exits with a non-zero status.
        Interrupted: If abandon_tools killed the tool while it was running.
    """
    if sys.platform == "win32":
        creationflags, process_group = subprocess.CREATE_NEW_PROCESS_GROUP, None
    else:
        creationflags, process_group = 0, 0

    try:
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            cmd,
            cwd=cwd,
            env={**os.environ, "OMP_NUM_THREADS": str(threads)},
            stdin=subprocess.PIPE if stdin is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
            process_group=process_group,
        )
    except OSError as e:
        # Windows doesn't say which file it failed to start.
        if e.filename is None:
            e.filename = cmd[0]

        raise

    with process:
        with _tools_lock:
            _running_tools.add(process)

        try:
            stdout, stderr = process.communicate(stdin)
        except BaseException:
            _kill_tool(process)
            raise
        finally:
            with _tools_lock:
                _running_tools.discard(process)
                abandoned = process in _abandoned_tools
                _abandoned_tools.discard(process)

    if abandoned:
        raise Interrupted

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


def _check_factors(n: int, factors: list[int], *, must_split: bool) -> str | None:
    """Check that a tool's factors are valid for the given number.

    Returns:
        str | None: A message describing why the factors are invalid, or None if they are valid.
    """
    if not factors:
        return "it reported no factors"

    if min(factors) < 2 or math.prod(factors) != n:  # ruff: ignore[magic-value-comparison]
        return f"its factors do not multiply back to the number: {' * '.join(map(str, factors))}"

    if must_split and len(factors) == 1:
        return "it did not find a factor"

    return None


# The start of each list of factors in YAFU's output.
YAFU_FACTORS_HEADER = "***factors found***"

# A factor in YAFU's output, labelled as prime, probably prime, composite or of unknown primality.
YAFU_FACTOR_PATTERN = re.compile(r"(?:P|PRP|C|U)[0-9]+ = (?P<factor>[0-9]+)\s*$")


def parse_yafu_factors(output: str) -> list[int]:
    """Extract the factors from YAFU's output.

    Returns:
        list[int]: The sorted factors.
    """
    factors: list[int] = []

    for line in output.rpartition(YAFU_FACTORS_HEADER)[2].splitlines():
        matches = YAFU_FACTOR_PATTERN.match(line)

        if matches:
            factors.append(int(matches["factor"]))

    return sorted(factors)


def run_yafu(  # ruff: ignore[too-many-arguments]
    method: str, n: int, expression: str, threads: int, yafu: YafuPaths, *options: str, must_split: bool = False
) -> tuple[list[int], float]:
    """Evaluate an expression that factors a number using YAFU, allowing it to use the given number of threads.

    The factors are checked before they are returned.

    Returns:
        tuple[list[int], float]: A tuple containing the sorted list of factors and the execution time in seconds.

    Raises:
        ToolFailure: If YAFU cannot be started, exits with an error or doesn't return a usable factorization.
        YafuError: If the method has now failed many times in a row.
    """
    cmd = [str(yafu.binary), expression, *options]
    start_time = time.perf_counter_ns()

    try:
        with get_work_dir(yafu.work, yafu.ini, "yafu-") as work_dir:
            # YAFU ignores the expression on the command line whenever standard input isn't a terminal, except on
            # Windows, where it doesn't recognize a pipe. Providing both covers either case.
            result = run_tool(cmd, work_dir, threads, stdin=f"{expression}\n")
    except (OSError, subprocess.CalledProcessError) as e:
        tool_failures.failed(YafuError, method, f"YAFU failed for {expression}: {_describe_tool_failure(e)}")

    execution_time = (time.perf_counter_ns() - start_time) / 1_000_000_000.0
    factors = parse_yafu_factors(result.stdout)

    if (problem := _check_factors(n, factors, must_split=must_split)) is not None:
        tool_failures.failed(YafuError, method, f"YAFU failed for {expression}: {problem}")

    tool_failures.succeeded(YafuError, method)

    return factors, execution_time


def run_cado_nfs(method: str, n: int, threads: int, cado_nfs_path: Path, work_path: Path) -> tuple[list[int], float]:
    """Factor a number using CADO-NFS, allowing it to use the given number of threads.

    The factors are checked before they are returned.

    Returns:
        tuple[list[int], float]: A tuple containing the sorted list of factors and the execution time in seconds.

    Raises:
        ToolFailure: If CADO-NFS cannot be started, exits with an error or doesn't return a usable factorization.
        CadoNfsError: If CADO-NFS has now failed too many times in a row.
    """
    cmd = [str(cado_nfs_path.absolute()), str(n), "-t", str(threads)]
    start_time = time.perf_counter_ns()

    try:
        with get_work_dir(work_path, None, "cado-nfs-") as work_dir:
            result = run_tool(cmd, work_dir, threads, stdin=str(n))
    except (OSError, subprocess.CalledProcessError) as e:
        tool_failures.failed(CadoNfsError, method, f"CADO-NFS failed for {n}: {_describe_tool_failure(e)}")

    execution_time = (time.perf_counter_ns() - start_time) / 1_000_000_000.0

    try:
        factors = sorted(map(int, result.stdout.split()))
    except ValueError:
        tool_failures.failed(CadoNfsError, method, f"CADO-NFS failed for {n}: its output could not be understood")

    if (problem := _check_factors(n, factors, must_split=True)) is not None:
        tool_failures.failed(CadoNfsError, method, f"CADO-NFS failed for {n}: {problem}")

    tool_failures.succeeded(CadoNfsError, method)

    return factors, execution_time
