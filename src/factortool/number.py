# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Number factorization methods for factortool."""

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

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

from loguru import logger

from factortool.assignments import assignment_expired
from factortool.constants import ECM_CURVES, FINAL_METHOD_NAMES, NFS_CADO_MIN_DIGITS, NFS_YAFU_MIN_DIGITS
from factortool.interrupt import Interrupted
from factortool.util import SMALL_PRIMES, format_number, get_work_dir, is_prime, log_factor_result

if TYPE_CHECKING:
    from factortool.backend import Backend
    from factortool.config import Config, FinalMethods, YafuPaths
    from factortool.stats import ECMCutoffs, FactoringStats


class FinalMethodNeeded(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Exception indicating that a final factoring method (SIQS or NFS) is needed for statistics."""

    def __init__(self, method: str) -> None:
        """Initialize the exception with the statistics key of the needed method."""
        super().__init__(method)
        self.method = method


class ToolError(Exception):
    """Exception raised when an external tool cannot be started or exits with an error."""

    exit_status: int


class YafuError(ToolError):
    """Exception raised when YAFU fails."""

    exit_status = 5


class CadoNfsError(ToolError):
    """Exception raised when CADO-NFS fails."""

    exit_status = 4


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


def _run_yafu(expression: str, threads: int, yafu: YafuPaths, *options: str) -> tuple[list[int], float]:
    """Evaluate an expression using YAFU, allowing it to use the given number of threads.

    Returns:
        tuple[list[int], float]: A tuple containing the sorted list of factors and the execution time in seconds.

    Raises:
        YafuError: If YAFU cannot be started or exits with an error.
    """
    cmd = [str(yafu.binary), expression, *options]
    start_time = time.perf_counter_ns()

    try:
        with get_work_dir(yafu.work, yafu.ini, "yafu-") as work_dir:
            result = run_tool(cmd, work_dir, threads)
    except (OSError, subprocess.CalledProcessError) as e:
        msg = f"YAFU failed for {expression}: {_describe_tool_failure(e)}"
        raise YafuError(msg) from e

    factors: list[int] = []

    for line in result.stdout.strip().split("\n"):
        matches = re.match(r"(P|C)([0-9]*) = (?P<factor>[0-9]*)", line)

        if matches:
            factors.append(int(matches["factor"]))

    end_time = time.perf_counter_ns()

    return sorted(factors), (end_time - start_time) / 1_000_000_000.0


@cache
def factor_ecm(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    n: int, level: int, final_methods: FinalMethods, max_threads: int, yafu: YafuPaths, stats: FactoringStats
) -> list[int]:
    """Factor a number using ECM via YAFU.

    Returns:
        list[int]: List of factors found.

    Raises:
        FinalMethodNeeded: If a final factoring method (SIQS or NFS) is needed for statistics.
    """
    # If any eligible final method has no statistics data, signal doing an immediate run of it.
    digits = len(str(n))

    for method in final_methods.for_digits(digits):
        run_count, _ = stats.get_final_stats(method, digits, max_threads)

        if run_count == 0:
            raise FinalMethodNeeded(method)

    # Determine the number of curves and B1.
    curves, b1 = ECM_CURVES[level]

    # Perform the ECM using YAFU.
    factors, execution_time = _run_yafu(
        f"ecm({n}, {curves})", max_threads, yafu, "-threads", str(max_threads), "-B1ecm", str(b1)
    )

    stats.update_ecm(digits, level, max_threads, execution_time, success=len(factors) > 1)

    if len(factors) > 1:
        log_factor_result(["ECM"], n, factors)

    return factors


@cache
def factor_yafu(n: int, method: str, max_threads: int, yafu: YafuPaths, stats: FactoringStats) -> list[int]:
    """Factor a number using YAFU with a specified method.

    Returns:
        list[int]: List of factors found.
    """
    digits = len(str(n))

    # Abort if the number of digits is too small for YAFU NFS.
    if method == "nfs" and digits < NFS_YAFU_MIN_DIGITS:
        return [n]

    options = ["-inmem", "200"]
    threads = 1

    # Rho and P-1 are single-threaded, so the engine runs several of them at once instead.
    if method not in {"pm1", "rho"}:
        options.extend(["-threads", str(max_threads)])
        threads = max_threads

    if method == "nfs":
        # YAFU silently runs SIQS instead of NFS below its QS/NFS crossover.
        options.extend(["-xover", "1"])

    factors, execution_time = _run_yafu(f"{method}({n})", threads, yafu, *options)

    if method == "siqs":
        stats.update_final("siqs", digits, threads, execution_time)
    elif method == "nfs":
        stats.update_final("nfs_yafu", digits, threads, execution_time)
    else:
        stats.update_probability(digits, method, threads, execution_time, success=len(factors) > 1)

    methods = {
        "rho": "Rho",
        "pm1": "P-1",
        "siqs": "SIQS",
        "nfs": "YAFU NFS",
    }

    if len(factors) > 1:
        log_factor_result([methods[method]], n, factors)

    return factors


@cache
def factor_yafu_direct(n: int, max_threads: int, yafu: YafuPaths, stats: FactoringStats) -> list[int]:
    """Factor a number using YAFU's automatic method selection.

    Returns:
        list[int]: List of factors found.
    """
    factors, execution_time = _run_yafu(f"factor({n})", max_threads, yafu, "-threads", str(max_threads))

    stats.update_probability(len(str(n)), "yafu", max_threads, execution_time, success=len(factors) > 1)

    if len(factors) > 1:
        log_factor_result(["YAFU"], n, factors)

    return factors


@cache
def factor_nfs_cado(n: int, max_threads: int, cado_nfs_path: Path, work_path: Path, stats: FactoringStats) -> list[int]:
    """Factor a number using CADO-NFS.

    Returns:
        list[int]: List of factors found.

    Raises:
        CadoNfsError: If CADO-NFS cannot be started or exits with an error.
    """
    # Abort if the number of digits is too small for CADO-NFS.
    digits = len(str(n))

    if digits < NFS_CADO_MIN_DIGITS:
        return [n]

    # Factor the number using CADO-NFS.
    cmd = [str(cado_nfs_path.absolute()), str(n), "-t", str(max_threads)]

    start_time = time.perf_counter_ns()

    try:
        with get_work_dir(work_path, None, "nfs-cado-") as work_dir:
            result = run_tool(cmd, work_dir, max_threads, stdin=str(n))
    except (OSError, subprocess.CalledProcessError) as e:
        msg = f"CADO-NFS failed for {n}: {_describe_tool_failure(e)}"
        raise CadoNfsError(msg) from e

    end_time = time.perf_counter_ns()
    execution_time = (end_time - start_time) / 1_000_000_000.0
    stats.update_final("nfs_cado", digits, max_threads, execution_time)

    factors = list(map(int, result.stdout.strip().split()))

    if len(factors) > 1:
        log_factor_result(["CADO-NFS"], n, sorted(factors))

    return sorted(factors)


@cache
def factor_tf(n: int, stats: FactoringStats) -> list[int]:
    """Factor a number using trial factoring.

    Returns:
        list[int]: List of factors found.
    """
    original_n = n
    factors: list[int] = []

    start_time = time.perf_counter_ns()

    for p in SMALL_PRIMES:
        while n % p == 0:
            factors.append(p)
            n //= p
        if n == 1:
            break

    # Check if n is a perfect square.
    if n > 1:
        sqrt_n = math.isqrt(n)
        if sqrt_n * sqrt_n == n:
            factors.extend([sqrt_n, sqrt_n])
            n = 1

    # Include the remaining n if it's greater than one.
    if n > 1:
        factors.append(n)

    # Log the factoring result.
    if len(factors) > 1:
        log_factor_result(["TF"], original_n, factors)

    end_time = time.perf_counter_ns()
    execution_time = (end_time - start_time) / 1_000_000_000.0
    stats.update_probability(len(str(original_n)), "tf", 1, execution_time, success=len(factors) > 1)

    return factors


class Number:
    """Representation of a number to be factored."""

    n: int
    prime_factors: list[int]
    composite_factors: list[int]
    methods: list[str]

    expires_at: float | None

    _ecm_level: int
    _ecm_finished: bool
    _stats: FactoringStats
    _config: Config
    _backend: Backend | None

    def __init__(self, n: int, config: Config, stats: FactoringStats, backend: Backend | None) -> None:
        """Initialize the number object."""
        self.n = n
        self._stats = stats
        self._config = config
        self._backend = backend
        self._submitted = False
        self.expires_at = None

        self._ecm_level = 0
        self._ecm_finished = False

        if is_prime(n):
            self.composite_factors = []
            self.prime_factors = [n]
        else:
            self.composite_factors = [n]
            self.prime_factors = []

        self.methods = []

    def __lt__(self, other: object) -> bool:
        """Less-than comparison based on the number value.

        Returns:
            bool: True if self.n < other.n, False otherwise.
        """
        return isinstance(other, self.__class__) and self.n < other.n

    def __eq__(self, other: object) -> bool:
        """Equality comparison based on the number value.

        Returns:
            bool: True if self.n == other.n, False otherwise.
        """
        return isinstance(other, self.__class__) and self.n == other.n

    def __hash__(self) -> int:
        """Hash based on the number value.

        Returns:
            int: Hash of the number.
        """
        return self.n.__hash__()

    @property
    def assignment_expired(self) -> bool:
        """Whether this number's assignment is too close to expiry to continue factoring."""
        return self.expires_at is not None and assignment_expired(self.expires_at)

    @property
    def attempted(self) -> bool:
        """Whether any factoring method has been run against this number."""
        return len(self.methods) > 0

    def report_partial(self) -> None:
        """Report the factorization found so far, even if incomplete."""
        if self._submitted or self._backend is None:
            return

        if len(self.prime_factors) == 0:
            return

        self._backend.submit([self])
        self._submitted = True

    @property
    def ecm_needed(self) -> bool:
        """Determine if further ECM factoring is needed, based on the latest statistics.

        Once a number has finished ECM, it stays finished even if the cutoff shifts later.
        """
        if self._ecm_finished or self.factored:
            return False

        if self._ecm_level < self.ecm_cutoffs.target:
            return True

        self._ecm_finished = True
        return False

    @property
    def ecm_cutoffs(self) -> ECMCutoffs:
        """The optimal and target ECM cutoffs for the largest remaining composite factor."""
        # We use the largest remaining composite factor, as that's the largest number we're still actually factoring.
        digits = len(str(max(self.composite_factors)))
        methods = self._config.final_methods.for_digits(digits)

        return self._stats.get_ecm_cutoffs(digits, self._config.max_threads, methods)

    @property
    def final_method(self) -> str:
        """The statistics key of the final factoring method for the largest remaining composite."""
        return self._choose_final_method(max(self.composite_factors, default=1))

    @property
    def factored(self) -> bool:
        """Determine if the number has been fully factored."""
        return len(self.composite_factors) == 0

    def _choose_final_method(self, n: int) -> str:
        digits = len(str(n))
        methods = self._config.final_methods.for_digits(digits)

        return self._stats.get_final_method(digits, self._config.max_threads, methods)

    @property
    def _yafu_args(self) -> tuple[int, YafuPaths, FactoringStats]:
        """Trailing arguments shared by every YAFU-backed factoring function."""
        return (self._config.max_threads, self._config.yafu_paths, self._stats)

    @property
    def _nfs_cado_args(self) -> tuple[int, Path, Path, FactoringStats]:
        """Trailing arguments for the CADO-NFS factoring function."""
        return (self._config.max_threads, self._config.cado_nfs_path, self._config.work_path, self._stats)

    def _factor_final(self, n: int, method: str) -> list[int]:
        """Factor a composite using the final method with the given statistics key.

        Returns:
            list[int]: List of factors found.
        """
        if method == "siqs":
            return factor_yafu(n, "siqs", *self._yafu_args)

        if method == "nfs_yafu":
            return factor_yafu(n, "nfs", *self._yafu_args)

        return factor_nfs_cado(n, *self._nfs_cado_args)

    def _factor_generic(
        self,
        method: str,
        factor_func: Callable[..., list[int]],
        *args: int | str | Path | YafuPaths | FinalMethods | FactoringStats,
        composites: Iterable[int] | None = None,
    ) -> None:
        for n in list(self.composite_factors if composites is None else composites):
            try:
                factors = factor_func(n, *args)
            except FinalMethodNeeded as e:
                method = FINAL_METHOD_NAMES[e.method]
                logger.info("Immediately doing {} on {} for statistics", method, format_number(n))
                factors = self._factor_final(n, e.method)

            if len(factors) > 1:
                self.methods.append(method)

            self.composite_factors.remove(n)

            for factor in factors:
                if is_prime(factor):
                    self.prime_factors.append(factor)
                else:
                    self.composite_factors.append(factor)

        if self.factored and len(self.methods) > 1:
            log_factor_result(set(self.methods), self.n, self.prime_factors)

        if not self._submitted and self.factored and self._backend is not None:
            self._backend.submit([self])
            self._submitted = True

    def factor_yafu_direct(self) -> None:
        """Factor using YAFU's automatic method selection."""
        self._factor_generic("YAFU", factor_yafu_direct, *self._yafu_args)

    def factor_tf(self) -> None:
        """Factor using trial factoring."""
        self._factor_generic("TF", factor_tf, self._stats)

    def factor_rho(self) -> None:
        """Factor using Pollard's Rho algorithm."""
        self._factor_generic("Rho", factor_yafu, "rho", *self._yafu_args)

    def factor_pm1(self) -> None:
        """Factor using Pollard's P-1 algorithm."""
        self._factor_generic("P-1", factor_yafu, "pm1", *self._yafu_args)

    def factor_ecm(self, level: int) -> None:
        """Factor using ECM at the specified level."""
        self._factor_generic("ECM", factor_ecm, level, self._config.final_methods, *self._yafu_args)
        self._ecm_level = level

    def factor_final(self) -> None:
        """Factor each remaining composite using its own fastest final method."""
        for n in self.composite_factors.copy():
            method = self._choose_final_method(n)
            self._factor_generic(FINAL_METHOD_NAMES[method], self._factor_final, method, composites=[n])


def format_factorization(number: Number, separator: str) -> str:
    """Format a number's known factorization as "n=<factors>".

    Every factor is listed individually, including repeats and any remaining composite factors, so the product of the
    listed factors is always the original number.

    Returns:
        str: Formatted factorization of the number.
    """
    return f"{number.n}={separator.join(map(str, number.prime_factors + number.composite_factors))}"


def format_results(numbers: Iterable[Number]) -> str:
    """Format the factoring results for output.

    Returns:
        str: Formatted factoring results.
    """
    return "\n".join(format_factorization(x, " ") for x in numbers)
