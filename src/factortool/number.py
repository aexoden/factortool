# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import math
import multiprocessing
import random
import subprocess
import sys
import time

from collections.abc import Callable, Iterable
from functools import cache
from multiprocessing.pool import ApplyResult  # noqa: TCH003 (conflicts with isort, which is more useful for the moment)
from pathlib import Path

from loguru import logger

from factortool.config import Config
from factortool.constants import (
    CADO_NFS_MIN_DIGITS,
    DECAY_RATE,
    ECM_CURVES,
    INITIAL_EPSILON,
    P_FACTOR_DECAY_RATE,
    P_FACTOR_INITIAL,
    UTILITY_SCALING_FACTOR,
)
from factortool.stats import FactoringStats
from factortool.util import SMALL_PRIMES, format_number, is_prime, log_factor_result


class NFSNeeded(Exception):  # noqa: N818
    pass


def factor_ecm_single(n: int, curves: int, b1: int, gmp_ecm_path: Path) -> list[int]:
    cmd: list[str] = [str(gmp_ecm_path), "-q", "-c", str(curves), str(b1)]
    result = subprocess.run(cmd, input=str(n), capture_output=True, text=True, check=False, process_group=0)

    if result.returncode & 0x01:
        logger.critical("ECM failed for {n}: {result.stderr}")  # noqa: RUF027
        sys.exit(3)

    if result.returncode & 0x02:
        factors: list[int] = list(map(int, result.stdout.strip().split()))
        return factors

    return []


@cache
def factor_ecm(n: int, level: int, max_threads: int, gmp_ecm_path: Path, stats: FactoringStats) -> list[int]:
    # If there is no NFS statistics data, signal doing an immediate NFS run.
    digits = len(str(n))
    nfs_run_count, _ = stats.get_nfs_stats(digits, max_threads)

    if digits >= CADO_NFS_MIN_DIGITS and nfs_run_count == 0:
        raise NFSNeeded

    # Determine the number of curves for each process.
    curves, b1 = ECM_CURVES[level]
    thread_count = min(max_threads, curves)
    curves_per_thread = curves // thread_count
    remaining_curves = curves % thread_count

    # Divide the curve sinto several parallel tasks.
    factors: set[int] = set()
    start_time = time.perf_counter_ns()

    with multiprocessing.Pool(processes=thread_count) as pool:
        tasks: list[ApplyResult[list[int]]] = []

        for i in range(thread_count):
            thread_curves = curves_per_thread + (1 if i < remaining_curves else 0)
            tasks.append(pool.apply_async(factor_ecm_single, (n, thread_curves, b1, gmp_ecm_path)))

        for task in tasks:
            factors = factors.union(task.get())

    end_time = time.perf_counter_ns()
    execution_time = (end_time - start_time) / 1_000_000_000.0
    stats.update_ecm(len(str(n)), level, thread_count, execution_time, success=len(factors) > 1)

    if len(factors) == 0:
        factors.add(n)

    if len(factors) > 1:
        log_factor_result(["ECM"], n, sorted(factors))

    return sorted(factors)


@cache
def factor_nfs(n: int, max_threads: int, cado_nfs_path: Path, stats: FactoringStats) -> list[int]:
    # Abort if the number of digits is too small for CADO-NFS.
    digits = len(str(n))

    if digits < CADO_NFS_MIN_DIGITS:
        return [n]

    # Factor the number using CADO-NFS.
    cmd = [str(cado_nfs_path), str(n), "-t", str(max_threads)]

    try:
        start_time = time.perf_counter_ns()

        result = subprocess.run(
            cmd,
            input=str(n),
            capture_output=True,
            text=True,
            check=True,
            process_group=0,
        )

        end_time = time.perf_counter_ns()
        execution_time = (end_time - start_time) / 1_000_000_000.0
        stats.update_nfs(digits, max_threads, execution_time)

        factors = list(map(int, result.stdout.strip().split()))

        if len(factors) > 1:
            log_factor_result(["NFS"], n, sorted(factors))
    except subprocess.CalledProcessError as e:
        logger.critical("NFS failed for {}: {}", n, e.stderr)
        sys.exit(4)
    else:
        return sorted(factors)


@cache
def factor_tf(n: int) -> list[int]:
    original_n = n
    factors: list[int] = []

    for p in SMALL_PRIMES:
        while n % p == 0:
            factors.append(p)
            n //= p
        if n == 1:
            break

    # Check if n is a perfect square.
    if n > 1:
        sqrt_n = int(math.isqrt(n))
        if sqrt_n * sqrt_n == n:
            factors.extend([sqrt_n, sqrt_n])
            n = 1

    # Include the remaining n if it's greater than one.
    if n > 1:
        factors.append(n)

    # Log the factoring result.
    if len(factors) > 1:
        log_factor_result(["TF"], original_n, factors)

    return factors


class Number:
    n: int
    prime_factors: list[int]
    composite_factors: list[int]
    methods: list[str]

    _ecm_overage: float
    _ecm_needed: bool
    _stats: FactoringStats
    _config: Config

    def __init__(self, n: int, config: Config, stats: FactoringStats) -> None:
        self.n = n
        self._stats = stats
        self._config = config

        self._ecm_needed = True
        self._ecm_overage = 0.0

        if is_prime(n):
            self.composite_factors = []
            self.prime_factors = [n]
        else:
            self.composite_factors = [n]
            self.prime_factors = []

        self.methods = []

    def __lt__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.n < other.n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.n == other.n

    def __hash__(self) -> int:
        return self.n.__hash__()

    @property
    def ecm_needed(self) -> bool:
        return self._ecm_needed

    @property
    def factored(self) -> bool:
        return len(self.composite_factors) == 0

    def _update_ecm_needed(self, level: int) -> None:
        self._ecm_needed, overage = self._calculate_ecm_needed(level)
        self._ecm_overage += overage

    # Some of this function is a stopgap until I implement a more global solution. Working level-by-level is not
    # particularly robust and can never work for certain factor distributions. (It could be that if you look only at the
    # next level, it's better to do NFS immediately, but that several ECM levels up, there is significant savings to
    # be had.)
    def _calculate_ecm_needed(self, level: int) -> tuple[bool, float]:  # noqa: PLR0911
        if self.factored:
            return (False, 0.0)

        # Retrieve statistics from the database. We use the largest remaining composite factor, as that's the largest
        # number we're actually factoring at this point.
        largest_composite_factor = max(self.composite_factors)
        digits = len(str(largest_composite_factor))
        smallest_composite_factor_digits = len(str(min(self.composite_factors)))

        nfs_count, nfs_time = self._stats.get_nfs_stats(digits, self._config.max_threads)
        ecm_count, ecm_time, ecm_p_factor = self._stats.get_ecm_stats(
            digits,
            level + 1,
            self._config.max_threads,
        )

        # If either of the run counts is zero, continue doing ECM. The actual ECM factoring code will force an initial
        # NFS run if the number of digits is compatible.
        if nfs_count == 0 or ecm_count == 0:
            return (True, 0.0)

        # If the run count is non-zero, the other values should never be None. Doing this check will satisfy static type
        # checking tools, however. We'll nonetheless output a log message as this would indicate a bug.
        if nfs_time is None or ecm_time is None or ecm_p_factor is None:
            logger.warning(
                "Unexpected None value encountered in Number._update_ecm_needed. Please report this as a bug."
            )
            return (True, 0.0)

        # Adjust the probability of finding a factor based on the number of samples. We'll arbitrarily assume a base
        # probability, and then apply an exponential decay factor. This way, we don't incorrectly make decisions based
        # on a zero probability that's merely the result in insufficient data.
        ecm_p_factor_multiplier = math.exp(-P_FACTOR_DECAY_RATE * ecm_count)
        ecm_p_factor = ecm_p_factor_multiplier * P_FACTOR_INITIAL + (1 - ecm_p_factor_multiplier) * ecm_p_factor

        # Calculate the expected utility of doing the additional round of ECM.
        ecm_expected_time = ecm_time + (1 - ecm_p_factor) * nfs_time
        utility = nfs_time - ecm_expected_time
        overage = -utility if utility < 0 else 0.0

        # Calculate the probability of doing exploration.
        p_exploration = INITIAL_EPSILON * math.exp(-DECAY_RATE * ecm_count)
        p_exploration *= math.exp(-abs(utility) * UTILITY_SCALING_FACTOR)

        # Temporary logging message for debugging.
        # logger.info("level: {}  digits: {}  scf_digits: {}  nfs_count: {}  nfs_time: {}  ecm_count: {}  ecm_time: {}  ecm_p_factor: {}  epf_multiplier: {}  ecm_expected: {}  utility: {}  p_exploration: {}",  # noqa: E501
        #    level, digits, smallest_composite_factor_digits, nfs_count, nfs_time, ecm_count, ecm_time, ecm_p_factor, ecm_p_factor_multiplier, ecm_expected_time, utility, p_exploration)  # noqa: E501

        # If the smallest remaining composite factor is too small for CADO-NFS, continue doing ECM.
        if smallest_composite_factor_digits < CADO_NFS_MIN_DIGITS:
            # logger.info("factor too small")  # noqa: ERA001
            return (True, overage)

        # If the current ECM level is already well beyond the maximum possible factor size, abort doing ECM.
        if level >= digits // 2 + 10:
            self._ecm_needed = False
            # logger.info("level too high for digit count")  # noqa: ERA001
            return (True, overage)

        # If our current overage is more than 0.25 times the NFS time, just abort and do NFS.
        if self._ecm_overage > 0.25 * nfs_time:
            # logger.info("too much overage")  # noqa: ERA001
            return (False, 0.0)

        # Randomly choose to do additional ECM with a probability based on the number of samples we already have and the
        # calculated utility.
        if random.random() < p_exploration:  # noqa: S311
            # logger.info("choosing to explore")  # noqa: ERA001
            return (True, overage)

        # If the expected utility of another round of ECM is positive, do ECM.
        if utility > 0:
            # logger.info("positive utility")  # noqa: ERA001
            return (True, overage)

        # If the potential overage is very small, just do ECM. This is one of the stopgaps.
        if overage < 1.0:
            # logger.info("limited overage")  # noqa: ERA001
            return (True, overage)

        # Otherwise, move on to NFS.
        # logger.info("Moving to NFS")  # noqa: ERA001
        return (False, 0.0)

    def _factor_generic(
        self,
        method: str,
        factor_func: Callable[..., list[int]],
        *args: int | Path | FactoringStats,
    ) -> None:
        composite_factors = self.composite_factors.copy()
        self.composite_factors = []

        for n in composite_factors:
            try:
                factors = factor_func(n, *args)
            except NFSNeeded:
                logger.info("Immediately doing NFS on {} for statistics", format_number(n))
                method = "NFS"
                factors = factor_nfs(n, self._config.max_threads, self._config.cado_nfs_path, self._stats)

            if len(factors) > 1:
                self.methods.append(method)

            for factor in factors:
                if is_prime(factor):
                    self.prime_factors.append(factor)
                else:
                    self.composite_factors.append(factor)

        if self.factored and len(self.methods) > 1:
            log_factor_result(set(self.methods), self.n, self.prime_factors)

    def factor_tf(self) -> None:
        self._factor_generic("TF", factor_tf)

    def factor_ecm(self, level: int) -> None:
        self._factor_generic("ECM", factor_ecm, level, self._config.max_threads, self._config.gmp_ecm_path, self._stats)
        self._update_ecm_needed(level)

    def factor_nfs(self) -> None:
        self._factor_generic("NFS", factor_nfs, self._config.max_threads, self._config.cado_nfs_path, self._stats)


def format_results(numbers: Iterable[Number]) -> str:
    return "\n".join([f"{x.n}={' '.join(map(str, x.prime_factors + x.composite_factors))}" for x in numbers])
