# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""External tool management for factortool."""

from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import math
import os
import re
import shlex
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
import time

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Generator

from loguru import logger
from pydantic import BaseModel, NonNegativeInt, ValidationError

from factortool.constants import MAX_CONSECUTIVE_TOOL_FAILURES
from factortool.interrupt import Interrupted
from factortool.util import format_number, get_work_dir, safe_write

if TYPE_CHECKING:
    from factortool.config import YafuPaths


class ToolFailure(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Exception raised when a single run of an external tool fails."""


class ToolDeadlineExceeded(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Exception raised when an external tool is killed for still running at its deadline."""


class ToolError(Exception):
    """Exception raised when an external tool fails repeatedly."""

    tool_name: str


class YafuError(ToolError):
    """Exception raised when YAFU continues to fail."""

    tool_name = "YAFU"


class CadoNfsError(ToolError):
    """Exception raised when CADO-NFS continues to fail."""

    tool_name = "CADO-NFS"


class ToolState(BaseModel):
    """Persistent state for external tools."""

    streaks: dict[str, NonNegativeInt] = {}


class ToolFailures:
    """Record of how the external tools have fared."""

    def __init__(self) -> None:
        """Initialize the record."""
        self._lock = threading.Lock()
        self._streaks: dict[str, int] = {}

    def reset(self) -> None:
        """Clear all recorded tool outcomes."""
        with self._lock:
            self._streaks.clear()

    def load(self, state_path: Path) -> None:
        """Replace the record with the one saved by a previous run, if any."""
        streaks: dict[str, int] = {}

        if state_path.exists():
            try:
                streaks = ToolState.model_validate_json(state_path.read_bytes()).streaks
            except (OSError, ValidationError) as e:
                logger.warning("Ignoring unreadable tool state at {}: {}", state_path, e)

        with self._lock:
            self._streaks = streaks

    def save(self, state_path: Path) -> None:
        """Save the record for the next run.

        Raises:
            OSError: If the state file cannot be written.
        """
        with self._lock:
            state = ToolState(streaks={method: streak for method, streak in self._streaks.items() if streak > 0})

        safe_write(state_path, state.model_dump_json().encode("utf-8"))

    def succeeded(self, method: str) -> None:
        """Record a run of a tool that returned a usable result for the given factoring method."""
        with self._lock:
            self._streaks[method] = 0

    def failed(self, tool: type[ToolError], method: str, message: str) -> NoReturn:
        """Record a failed run of a tool for the given factoring method, and raise the failure.

        Failures are counted per method rather than per tool.

        Raises:
            ToolFailure: If this is an isolated failure.
            ToolError: If the method has failed repeatedly.
        """
        with self._lock:
            streak = self._streaks[method] = self._streaks.get(method, 0) + 1

        if streak >= MAX_CONSECUTIVE_TOOL_FAILURES:
            msg = f"{message} ({method} has failed {streak} times in a row, which suggests a broken installation)"
            raise tool(msg)

        raise ToolFailure(message)


tool_failures = ToolFailures()


def _describe_tool_failure(error: OSError | subprocess.CalledProcessError) -> str:
    """Describe why a tool failed, preferring its own error output when it ran at all.

    Returns:
        str: The tool's error output, or the reason it could not be started.
    """
    if isinstance(error, subprocess.CalledProcessError):
        return str(error.stderr).strip() or f"exit status {error.returncode}"

    return str(error)


# External tools currently running in any thread, each with the handle of the Windows job it was placed in, if any.
_running_tools: dict[subprocess.Popen[str], int | None] = {}

# External tools that abandon_tools has killed.
_abandoned_tools: set[subprocess.Popen[str]] = set()

_tools_lock = threading.Lock()

if sys.platform == "win32":
    import ctypes

    from ctypes import wintypes

    # Starts a process without letting it run.
    CREATE_SUSPENDED = 0x00000004

    # The access needed to place a process in a job and then resume it.
    PROCESS_TERMINATE = 0x0001
    PROCESS_SET_QUOTA = 0x0100
    PROCESS_SUSPEND_RESUME = 0x0800

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    _ntdll.NtResumeProcess.restype = wintypes.LONG

    def _create_job(process_handle: int) -> int | None:
        """Place a process in a new job, which every process it spawns will also be part of.

        Returns:
            int | None: A handle to the job, or None if the process could not be placed in one.
        """
        job: int | None = _kernel32.CreateJobObjectW(None, None)

        if job is not None and _kernel32.AssignProcessToJobObject(job, process_handle):
            return job

        error = ctypes.WinError(ctypes.get_last_error())

        if job is not None:
            _kernel32.CloseHandle(job)

        logger.warning("Failed to place a tool in a job, so its helper processes may outlive it: {}", error)

        return None

    def _start_in_job(process: subprocess.Popen[str]) -> int | None:
        """Let a tool that was started suspended run, having first placed it in a job.

        Killing a process tree by walking it can miss a helper that is started while the walk is underway.

        Returns:
            int | None: A handle to the job, or None if the tool is running outside of one.

        Raises:
            OSError: If the tool could not be resumed.
        """
        access = PROCESS_TERMINATE | PROCESS_SET_QUOTA | PROCESS_SUSPEND_RESUME
        process_handle: int | None = _kernel32.OpenProcess(access, False, process.pid)  # ruff: ignore[boolean-positional-value-in-call]

        if process_handle is None:
            error = ctypes.WinError(ctypes.get_last_error())
            raise error

        try:
            job = _create_job(process_handle)

            # The subprocess module keeps no handle to the suspended thread, leaving this as the way to resume it.
            status: int = _ntdll.NtResumeProcess(process_handle)

            if status < 0:
                if job is not None:
                    _kernel32.CloseHandle(job)

                msg = f"Failed to resume {process.args!r}: NTSTATUS {status & 0xFFFFFFFF:#010x}"
                raise OSError(msg)
        finally:
            _kernel32.CloseHandle(process_handle)

        return job


def _kill_tool(process: subprocess.Popen[str], job: int | None) -> None:
    """Kill a tool along with every helper process it may have started."""
    if sys.platform == "win32":
        if job is not None:
            _kernel32.TerminateJobObject(job, 1)
        else:
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
        for process, job in _running_tools.items():
            _abandoned_tools.add(process)
            _kill_tool(process, job)


# How long a tool runs between warnings that it is still running.
STILL_RUNNING_INTERVAL = 1800.0

# When the tools started by the current thread are to be killed.
_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("tool_deadline", default=None)


@contextlib.contextmanager
def tool_deadline(deadline: float | None) -> Generator[None]:
    """Kill any tool the current thread runs inside this block that is still running at the deadline.

    The deadline is a time.monotonic() value, or None for no deadline. The run of a killed tool raises
    ToolDeadlineExceeded.

    Yields:
        None: Control, for the duration of the work the deadline applies to.
    """
    token = _deadline.set(deadline)

    try:
        yield
    finally:
        _deadline.reset(token)


def _communicate(process: subprocess.Popen[str], stdin: str | None, description: str) -> tuple[str, str]:
    """Wait for a tool to finish, warning periodically that it is still running.

    Returns:
        tuple[str, str]: The tool's standard output and standard error.

    Raises:
        ToolDeadlineExceeded: If the tool is still running when the deadline is reached.
    """
    deadline = _deadline.get()
    start_time = time.monotonic()
    next_warning = start_time + STILL_RUNNING_INTERVAL

    while True:
        now = time.monotonic()

        if deadline is not None and now >= deadline:
            raise ToolDeadlineExceeded(description)

        if now >= next_warning:
            logger.warning("{} has been running for {}", description, dt.timedelta(seconds=round(now - start_time)))
            next_warning += STILL_RUNNING_INTERVAL

        wake = next_warning if deadline is None else min(next_warning, deadline)

        try:
            return process.communicate(stdin, timeout=wake - now)
        except subprocess.TimeoutExpired:
            # The input has already been sent, so we can't send it again.
            stdin = None


def _tool_environment(threads: int) -> dict[str, str]:
    """Build the environment for an external tool allowed the given number of threads.

    Returns:
        dict[str, str]: The environment for the tool.
    """
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}

    if sys.prefix != sys.base_prefix:
        venv_bin = os.path.realpath(Path(sys.prefix) / ("Scripts" if sys.platform == "win32" else "bin"))
        path = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(entry for entry in path if not entry or os.path.realpath(entry) != venv_bin)
        env.pop("VIRTUAL_ENV", None)
        env.pop("VIRTUAL_ENV_PROMPT", None)

    return env


def run_tool(
    cmd: list[str], cwd: Path, threads: int, *, stdin: str | None = None, description: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run an external tool in its own process group, capturing its output.

    The separate process group ensures that the process isn't immediately killed in the event of a terminal interrupt.
    On Windows, the tool is also placed in a job, so that it can be killed along with its helper processes.

    The tool inherits the environment, apart from OMP_NUM_THREADS, which is set to the number of threads the tool is
    allowed, as anything built with OpenMP otherwise uses every available core, and factortool's virtual environment,
    which is removed from PATH.

    Returns:
        subprocess.CompletedProcess[str]: The finished process and its output.

    Raises:
        OSError: If the tool cannot be started.
        subprocess.CalledProcessError: If the tool exits with a non-zero status.
        Interrupted: If abandon_tools killed the tool while it was running.
        ToolDeadlineExceeded: If the tool was killed for still running at the deadline set by tool_deadline.
    """
    if sys.platform == "win32":
        creationflags, process_group = subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED, None
    else:
        creationflags, process_group = 0, 0

    description = description or Path(cmd[0]).name
    logger.debug("Running {} in {}", shlex.join(cmd), cwd)
    start_time = time.monotonic()

    try:
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            cmd,
            cwd=cwd,
            env=_tool_environment(threads),
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

    job: int | None = None

    with process:
        try:
            if sys.platform == "win32":
                job = _start_in_job(process)

            with _tools_lock:
                _running_tools[process] = job

            stdout, stderr = _communicate(process, stdin, description)
        except BaseException:
            _kill_tool(process, job)
            raise
        finally:
            with _tools_lock:
                _running_tools.pop(process, None)
                abandoned = process in _abandoned_tools
                _abandoned_tools.discard(process)

            if sys.platform == "win32" and job is not None:
                _kernel32.CloseHandle(job)

    if abandoned:
        raise Interrupted

    logger.debug(
        "{} exited with status {} after {:.2f}s", description, process.returncode, time.monotonic() - start_time
    )

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
        YafuError: If the method has now failed too many times in a row.
    """
    cmd = [str(yafu.binary), expression, *options]
    start_time = time.perf_counter_ns()

    try:
        with get_work_dir(yafu.work, yafu.ini, "yafu-") as work_dir:
            # YAFU ignores the expression on the command line whenever standard input isn't a terminal, except on
            # Windows, where it doesn't recognize a pipe. Providing both covers either case.
            description = f"{method} on {format_number(n)}"
            result = run_tool(cmd, work_dir, threads, stdin=f"{expression}\n", description=description)
    except (OSError, subprocess.CalledProcessError) as e:
        tool_failures.failed(YafuError, method, f"YAFU failed for {expression}: {_describe_tool_failure(e)}")

    execution_time = (time.perf_counter_ns() - start_time) / 1_000_000_000.0
    factors = parse_yafu_factors(result.stdout)

    if (problem := _check_factors(n, factors, must_split=must_split)) is not None:
        tool_failures.failed(YafuError, method, f"YAFU failed for {expression}: {problem}")

    tool_failures.succeeded(method)

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
            result = run_tool(cmd, work_dir, threads, stdin=str(n), description=f"{method} on {format_number(n)}")
    except (OSError, subprocess.CalledProcessError) as e:
        tool_failures.failed(
            CadoNfsError, method, f"CADO-NFS failed for {format_number(n, truncate=False)}: {_describe_tool_failure(e)}"
        )

    execution_time = (time.perf_counter_ns() - start_time) / 1_000_000_000.0

    try:
        factors = sorted(map(int, result.stdout.split()))
    except ValueError:
        tool_failures.failed(
            CadoNfsError,
            method,
            f"CADO-NFS failed for {format_number(n, truncate=False)}: its output could not be understood",
        )

    if (problem := _check_factors(n, factors, must_split=True)) is not None:
        tool_failures.failed(CadoNfsError, method, f"CADO-NFS failed for {format_number(n, truncate=False)}: {problem}")

    tool_failures.succeeded(method)

    return factors, execution_time
