# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import concurrent.futures
import math
import re
import subprocess
import sys
import time

from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path

from loguru import logger

from factortool.config import Config
from factortool.constants import CADO_NFS_MIN_DIGITS, ECM_CURVES
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        tasks: list[concurrent.futures.Future[list[int]]] = []

        for i in range(thread_count):
            thread_curves = curves_per_thread + (1 if i < remaining_curves else 0)
            tasks.append(executor.submit(factor_ecm_single, n, thread_curves, b1, gmp_ecm_path))

        # Only accept the factors from one run, as otherwise we may end up with extra factors in some cases. This could
        # be mitigated by manually checking each factor for divisibility into the composite and producing our own
        # remaining cofactor, but that may be more effort than is needed.
        for task in tasks:
            task_factors = task.result()

            if len(task_factors) > 0:
                factors = factors.union(task_factors)
                break

    end_time = time.perf_counter_ns()
    execution_time = (end_time - start_time) / 1_000_000_000.0
    stats.update_ecm(len(str(n)), level, thread_count, execution_time, success=len(factors) > 1)

    if len(factors) == 0:
        factors.add(n)

    if len(factors) > 1:
        log_factor_result(["ECM"], n, sorted(factors))

    return sorted(factors)


@cache
def factor_yafu(n: int, method: str, max_threads: int, yafu_path: Path) -> list[int]:
    cmd = [str(yafu_path), f"{method}({n})", "-threads", str(max_threads)]

    try:
        result = subprocess.run(
            cmd,
            cwd=yafu_path.parent,
            capture_output=True,
            text=True,
            check=True,
            process_group=0,
        )

        factors: list[int] = []

        for line in result.stdout.strip().split("\n"):
            matches = re.match(r"(P|C)([0-9]*) = (?P<factor>[0-9]*)", line)

            if matches:
                factors.append(int(matches["factor"]))
    except subprocess.CalledProcessError as e:
        logger.critical("YAFU failed for {} with method {}: {}", n, method, e.stderr)
        sys.exit(5)
    else:
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

    _ecm_level: int
    _maximum_ecm_level: int
    _stats: FactoringStats
    _config: Config

    def __init__(self, n: int, config: Config, stats: FactoringStats) -> None:
        self.n = n
        self._stats = stats
        self._config = config

        self._ecm_level = 0

        if is_prime(n):
            self.composite_factors = []
            self.prime_factors = [n]
        else:
            self.composite_factors = [n]
            self.prime_factors = []

        self._set_maximum_ecm_level()
        self.methods = []

    def __lt__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.n < other.n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.n == other.n

    def __hash__(self) -> int:
        return self.n.__hash__()

    @property
    def ecm_needed(self) -> bool:
        return self._ecm_level < self._maximum_ecm_level

    @property
    def factored(self) -> bool:
        return len(self.composite_factors) == 0

    def _set_maximum_ecm_level(self) -> None:
        # If factored, there is no need for any ECM.
        if self.factored:
            self._maximum_ecm_level = self._ecm_level
            return

        # Retrieve statistics from the database. We use the largest remaining composite factor, as that's the largest
        # number we're actually factoring at this point.
        largest_composite_factor = max(self.composite_factors)
        digits = len(str(largest_composite_factor))
        smallest_composite_factor_digits = len(str(min(self.composite_factors)))

        # If the smallest composite factor is smaller than supported by CADO-NFS, just use the true maximum since we
        # can't do NFS anyway.
        if smallest_composite_factor_digits < CADO_NFS_MIN_DIGITS:
            self._maximum_ecm_level = max(ECM_CURVES.keys())
            return

        # Establish a semi-arbitrary limit on our maximum ECM level. The smallest factors should never have more than
        # about half the digits of the number, so do a few levels beyond that (as ECM may miss factors).
        self._maximum_ecm_level = digits // 2 + 10

        # Collect data on the statistics based on stopping ECM at a given level. Once we've reached the first level with
        # no data, simply abort. Along the way, we'll note which ECM level was fastest on average.
        ecm_data: dict[int, tuple[int, float | None]] = {}
        best_maximum_ecm_level = None
        best_maximum_ecm_level_time = 0.0

        for ecm_level in range(min(ECM_CURVES.keys()), self._maximum_ecm_level + 1):
            ecm_count, average_time = self._stats.get_average_time(digits, ecm_level, self._config.max_threads)

            if ecm_count == 0:
                break

            assert average_time is not None  # noqa: S101

            if best_maximum_ecm_level is None or average_time < best_maximum_ecm_level_time:
                best_maximum_ecm_level = ecm_level
                best_maximum_ecm_level_time = average_time

            ecm_data[ecm_level] = (ecm_count, average_time)

        # If no data at all was collected, limit the ECM work to one-third the digit count. This is probably too little
        # for smaller numbers, but it's only for the first run, at which point the other metrics will take over. This
        # prevents trying to do way too much ECM on the first run, which will be more important with medium and large
        # numbers.
        if best_maximum_ecm_level is None:
            self._maximum_ecm_level = digits // 3
            return

        # If the highest level with data is the fastest, there's no evidence ever doing NFS is useful, so just return,
        # as we've already set a reasonable maximum ECM level above.
        if max(ecm_data.keys()) == best_maximum_ecm_level:
            return

        # Otherwise, we'll balance collecting more data with taking advantage of what we already know, based on how many
        # samples have been collected. The function as defined here will do an extra number of levels based on the
        # number of samples for the level following the one with the minimum time. The parameters as chosen here will do
        # eight extra levels when there are 4 samples, decaying to zero extra levels when 1024 samples are reached.
        # These numbers are, of course, arbitrary, but should work reasonably well enough. We need to check each level
        # from the minimum up to the test level because in certain situations, the numbers won't be monotonically
        # decreasing.
        test_ecm_level = best_maximum_ecm_level + 1
        lowest_ecm_count = None

        for ecm_level in range(min(ECM_CURVES.keys()), test_ecm_level + 1):
            test_ecm_count, _ = ecm_data.get(ecm_level, (0, None))

            if lowest_ecm_count is None or test_ecm_count < lowest_ecm_count:
                lowest_ecm_count = test_ecm_count

        if lowest_ecm_count is None:
            lowest_ecm_count = 0

        extra_ecm_levels = math.ceil(-math.log2(lowest_ecm_count) + 10)

        self._maximum_ecm_level = min(best_maximum_ecm_level + extra_ecm_levels, self._maximum_ecm_level)

    def _factor_generic(
        self,
        method: str,
        factor_func: Callable[..., list[int]],
        *args: int | str | Path | FactoringStats,
    ) -> bool:
        composite_factors = self.composite_factors.copy()
        self.composite_factors = []
        found_factors = False

        for n in composite_factors:
            try:
                factors = factor_func(n, *args)
            except NFSNeeded:
                logger.info("Immediately doing NFS on {} for statistics", format_number(n))
                method = "NFS"
                factors = factor_nfs(n, self._config.max_threads, self._config.cado_nfs_path, self._stats)

            if len(factors) > 1:
                self.methods.append(method)
                found_factors = True

            for factor in factors:
                if is_prime(factor):
                    self.prime_factors.append(factor)
                else:
                    self.composite_factors.append(factor)

        if self.factored and len(self.methods) > 1:
            log_factor_result(set(self.methods), self.n, self.prime_factors)

        return found_factors

    def factor_tf(self) -> None:
        if self._factor_generic("TF", factor_tf):
            self._set_maximum_ecm_level()

    def factor_rho(self) -> None:
        if self._factor_generic("Rho", factor_yafu, "rho", self._config.max_threads, self._config.yafu_path):
            self._set_maximum_ecm_level()

    def factor_pm1(self) -> None:
        if self._factor_generic("P-1", factor_yafu, "pm1", self._config.max_threads, self._config.yafu_path):
            self._set_maximum_ecm_level()

    def factor_ecm(self, level: int) -> None:
        found_factors = self._factor_generic(
            "ECM", factor_ecm, level, self._config.max_threads, self._config.gmp_ecm_path, self._stats
        )

        self._ecm_level = level

        if found_factors:
            self._set_maximum_ecm_level()

    def factor_nfs(self) -> None:
        self._factor_generic("NFS", factor_nfs, self._config.max_threads, self._config.cado_nfs_path, self._stats)


def format_results(numbers: Iterable[Number]) -> str:
    return "\n".join([f"{x.n}={' '.join(map(str, x.prime_factors + x.composite_factors))}" for x in numbers])
