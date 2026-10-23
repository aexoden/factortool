# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Number factorization methods for factortool."""

from __future__ import annotations

import math
import time

from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

from loguru import logger

from factortool.assignments import assignment_expired
from factortool.constants import (
    ECM_CURVES,
    FINAL_METHOD_NAMES,
    MIN_FACTORABLE,
    NFS_CADO_MIN_DIGITS,
    NFS_YAFU_MIN_DIGITS,
)
from factortool.tools import ToolFailure, run_cado_nfs, run_yafu
from factortool.util import format_number, is_prime, log_factor_result, small_primes

if TYPE_CHECKING:
    from factortool.backend import Backend
    from factortool.config import Config, FinalMethods, YafuPaths
    from factortool.stats import ECMCutoffs, FactoringStats
    from factortool.submissions import Submission

YAFU_METHOD_NAMES: dict[str, str] = {
    "rho": "Rho",
    "pm1": "P-1",
    "siqs": FINAL_METHOD_NAMES["siqs"],
    "nfs": FINAL_METHOD_NAMES["nfs_yafu"],
}


class Outcome(Enum):
    """How a number stands at the end of a factoring run."""

    # Every factor found is prime.
    FACTORED = "factored"

    # A tool failed on the number, so it was set aside.
    FAILED = "failed"

    # The number's assignment ran out before it was factored.
    EXPIRED = "expired"

    # At least one method was run on the number, but the run ended before it was factored.
    UNFINISHED = "unfinished"

    # The run ended before any method was run on the number.
    NOT_STARTED = "not started"


class FinalMethodNeeded(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Exception indicating that a final factoring method (SIQS or NFS) is needed for statistics."""

    def __init__(self, method: str) -> None:
        """Initialize the exception with the statistics key of the needed method."""
        super().__init__(method)
        self.method = method


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
    factors, execution_time = run_yafu(
        "ECM", n, f"ecm({n}, {curves})", max_threads, yafu, "-threads", str(max_threads), "-B1ecm", str(b1)
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

    Raises:
        ValueError: If the number is too small for YAFU NFS.
    """
    digits = len(str(n))

    # YAFU NFS can stall indefinitely on a number this small.
    if method == "nfs" and digits < NFS_YAFU_MIN_DIGITS:
        msg = f"YAFU NFS needs at least {NFS_YAFU_MIN_DIGITS} digits, but {n} has {digits}"
        raise ValueError(msg)

    options = ["-inmem", "200"]
    threads = 1

    # Rho and P-1 are single-threaded, so the engine runs several of them at once instead.
    if method not in {"pm1", "rho"}:
        options.extend(["-threads", str(max_threads)])
        threads = max_threads

    if method == "nfs":
        # YAFU silently runs SIQS instead of NFS below its QS/NFS crossover.
        options.extend(["-xover", "1"])

    # The final methods are expected to find a factor, so failure is considered an error.
    factors, execution_time = run_yafu(
        YAFU_METHOD_NAMES[method], n, f"{method}({n})", threads, yafu, *options, must_split=method in {"siqs", "nfs"}
    )

    if method == "siqs":
        stats.update_final("siqs", digits, threads, execution_time)
    elif method == "nfs":
        stats.update_final("nfs_yafu", digits, threads, execution_time)
    else:
        stats.update_probability(digits, method, threads, execution_time, success=len(factors) > 1)

    if len(factors) > 1:
        log_factor_result([YAFU_METHOD_NAMES[method]], n, factors)

    return factors


@cache
def factor_yafu_direct(n: int, max_threads: int, yafu: YafuPaths, stats: FactoringStats) -> list[int]:
    """Factor a number using YAFU's automatic method selection.

    Returns:
        list[int]: List of factors found.
    """
    factors, execution_time = run_yafu(
        "YAFU", n, f"factor({n})", max_threads, yafu, "-threads", str(max_threads), must_split=True
    )

    stats.update_probability(len(str(n)), "yafu", max_threads, execution_time, success=True)
    log_factor_result(["YAFU"], n, factors)

    return factors


@cache
def factor_nfs_cado(n: int, max_threads: int, cado_nfs_path: Path, work_path: Path, stats: FactoringStats) -> list[int]:
    """Factor a number using CADO-NFS.

    Returns:
        list[int]: List of factors found.

    Raises:
        ValueError: If the number is too small for CADO-NFS.
    """
    digits = len(str(n))

    if digits < NFS_CADO_MIN_DIGITS:
        msg = f"CADO-NFS needs at least {NFS_CADO_MIN_DIGITS} digits, but {n} has {digits}"
        raise ValueError(msg)

    factors, execution_time = run_cado_nfs(FINAL_METHOD_NAMES["nfs_cado"], n, max_threads, cado_nfs_path, work_path)

    stats.update_final("nfs_cado", digits, max_threads, execution_time)
    log_factor_result([FINAL_METHOD_NAMES["nfs_cado"]], n, factors)

    return factors


@cache
def factor_tf(n: int, stats: FactoringStats) -> list[int]:
    """Factor a number using trial factoring.

    Returns:
        list[int]: List of factors found.
    """
    original_n = n
    factors: list[int] = []

    # The primes are generated on first use, so do that before beginning timing.
    primes = small_primes()
    start_time = time.perf_counter_ns()

    for p in primes:
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
    started: bool
    tool_failed: bool

    _ecm_level: int
    _ecm_finished: bool
    _stats: FactoringStats
    _config: Config
    _backend: Backend | None

    def __init__(
        self, n: int, config: Config, stats: FactoringStats, backend: Backend | None, factors: Sequence[int] = ()
    ) -> None:
        """Initialize the number object, optionally resuming from factors found by a previous run.

        Raises:
            ValueError: If the number is below the minimum factorable value.
        """
        if n < MIN_FACTORABLE:
            msg = f"{n} cannot be factored, as it is below {MIN_FACTORABLE}"
            raise ValueError(msg)

        self.n = n
        self._stats = stats
        self._config = config
        self._backend = backend
        self._submitted = False
        self.expires_at = None
        self.started = False
        self.tool_failed = False

        self._ecm_level = 0
        self._ecm_finished = False

        if factors and (math.prod(factors) != n or min(factors) <= 1):
            logger.warning(
                "Ignoring the factors saved for {}, as they are not a factorization of it",
                format_number(n, truncate=False),
            )
            factors = ()

        known_factors = factors or (n,)
        self.prime_factors = [x for x in known_factors if is_prime(x)]
        self.composite_factors = [x for x in known_factors if not is_prime(x)]

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
    def active(self) -> bool:
        """Whether this number is still actively being factored.

        Numbers are considered active until they're fully factored or until a tool fails on it.
        """
        return not self.factored and not self.tool_failed

    @property
    def outcome(self) -> Outcome:
        """The outcome of this number's factoring process."""
        if self.factored:
            return Outcome.FACTORED

        if self.tool_failed:
            return Outcome.FAILED

        if self.assignment_expired:
            return Outcome.EXPIRED

        return Outcome.UNFINISHED if self.started else Outcome.NOT_STARTED

    @property
    def known_factors(self) -> list[int]:
        """Every factor found so far, prime or composite, whose product is the number itself."""
        return [*self.prime_factors, *self.composite_factors]

    @property
    def partially_factored(self) -> bool:
        """Whether at least one factor has been found without the number being fully factored."""
        return not self.factored and len(self.known_factors) > 1

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
        if self._ecm_finished or not self.active:
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
        # CADO-NFS is only ever eligible if the path is provided in the configuration.
        cado_nfs_path = cast("Path", self._config.cado_nfs_path)

        return (self._config.max_threads, cado_nfs_path, self._config.work_path, self._stats)

    def _run_final(self, n: int, method: str) -> list[int]:
        """Factor a composite using the final method with the given statistics key.

        Returns:
            list[int]: List of factors found.
        """
        if method == "siqs":
            return factor_yafu(n, "siqs", *self._yafu_args)

        if method == "nfs_yafu":
            return factor_yafu(n, "nfs", *self._yafu_args)

        return factor_nfs_cado(n, *self._nfs_cado_args)

    def _set_aside(self, failure: ToolFailure) -> None:
        """Give up on this number for the rest of the run, after a tool failed on it."""
        logger.warning("{}. Leaving {} unfactored", failure, format_number(self.n))
        self.tool_failed = True

    def _replace_composite(self, n: int, method: str, factors: list[int]) -> list[int]:
        """Replace a composite factor with the factors a method found for it.

        Progress on a number that remains unfinished is checkpointed at once.

        Returns:
            list[int]: The factors that are themselves composite.
        """
        if len(factors) > 1:
            self.methods.append(method)

        self.composite_factors.remove(n)
        composites: list[int] = []

        for factor in factors:
            if is_prime(factor):
                self.prime_factors.append(factor)
            else:
                self.composite_factors.append(factor)
                composites.append(factor)

        if len(factors) > 1 and not self.factored and self._backend is not None:
            self._backend.checkpoint(self)

        return composites

    def _finish(self) -> None:
        """Log and submit the factorization if complete."""
        if not self.factored:
            return

        if len(self.methods) > 1:
            log_factor_result(set(self.methods), self.n, self.prime_factors)

        if not self._submitted and self._backend is not None:
            self._backend.submit([self])
            self._submitted = True

    def _factor_final(self, n: int, first_method: str | None = None) -> None:
        """Completely factor a composite using the final methods.

        The fastest eligible method is tried first, unless another is given. If a method fails or does not fully factor
        the number, the remaining eligible methods are tried in turn, and the number is set aside once none are left.
        Any factors that are themselves composite are then factored in the same way.
        """
        digits = len(str(n))
        methods = list(self._config.final_methods.for_digits(digits))
        method = first_method

        while self.active and methods:
            if method is None:
                method = self._stats.get_final_method(digits, self._config.max_threads, methods)

            methods.remove(method)

            try:
                factors = self._run_final(n, method)
            except ToolFailure as e:
                failure = e
            else:
                if len(factors) > 1:
                    for composite in self._replace_composite(n, FINAL_METHOD_NAMES[method], factors):
                        self._factor_final(composite)

                    return

                failure = ToolFailure(f"{FINAL_METHOD_NAMES[method]} did not find a factor of {format_number(n)}")

            if not methods:
                self._set_aside(failure)
                return

            logger.warning("{}. Trying another method", failure)
            method = None

    def _factor_generic(
        self,
        method: str,
        factor_func: Callable[..., list[int]],
        *args: int | str | Path | YafuPaths | FinalMethods | FactoringStats,
    ) -> None:
        self.started = True

        for n in list(self.composite_factors):
            try:
                self._replace_composite(n, method, factor_func(n, *args))
            except FinalMethodNeeded as e:
                logger.info("Immediately doing {} on {} for statistics", FINAL_METHOD_NAMES[e.method], format_number(n))
                self._factor_final(n, e.method)
            except ToolFailure as e:
                self._set_aside(e)

            if not self.active:
                break

        self._finish()

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
        """Factor each remaining composite using its own fastest final method, falling back to the others."""
        self.started = True

        for n in self.composite_factors.copy():
            self._factor_final(n)

        self._finish()


def format_factorization(number: Number | Submission, separator: str) -> str:
    """Format a number's known factorization as "n=<factors>".

    Every factor is listed individually, including repeats and any remaining composite factors, so the product of the
    listed factors is always the original number.

    Returns:
        str: Formatted factorization of the number.
    """
    return f"{number.n}={separator.join(map(str, (*number.prime_factors, *number.composite_factors)))}"


def format_results(numbers: Iterable[Number]) -> str:
    """Format the factoring results for output, one number per line in ascending order.

    Returns:
        str: Formatted factoring results.
    """
    return "".join(f"{format_factorization(x, ' ')}\n" for x in sorted(numbers))
