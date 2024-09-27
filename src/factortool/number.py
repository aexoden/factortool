# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import math
import multiprocessing
import subprocess
import sys
import time

from functools import cache
from multiprocessing.pool import ApplyResult  # noqa: TCH003 (conflicts with isort, which is more useful for the moment)
from pathlib import Path

from loguru import logger

from factortool.util import SMALL_PRIMES, is_prime, log_factor_result


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


def factor_ecm(n: int, curves: int, b1: int, max_threads: int, gmp_ecm_path: Path) -> list[int]:
    # Determine the number of curves for each process.
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

    # TODO: Log the execution time of this run.
    end_time = time.perf_counter_ns()
    _execution_time = (end_time - start_time) / 1_0000_000_000.0

    if len(factors) == 0:
        factors.add(n)

    if len(factors) > 1:
        log_factor_result("ECM", n, sorted(factors))

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
        log_factor_result("TF", original_n, factors)

    return factors


class Number:
    n: int
    prime_factors: list[int]
    composite_factors: list[int]
    methods: set[str]

    def __init__(self, n: int) -> None:
        if is_prime(n):
            self.composite_factors = []
            self.prime_factors = [n]
        else:
            self.composite_factors = [n]
            self.prime_factors = []

        self.methods = set()

    def factor_tf(self) -> None:
        composite_factors = self.composite_factors.copy()
        self.composite_factors = []

        for n in composite_factors:
            factors = factor_tf(n)

            if len(factors) > 1:
                self.methods.add("TF")

            for factor in factors:
                if is_prime(factor):
                    self.prime_factors.append(factor)
                else:
                    self.composite_factors.append(factor)

    def factor_ecm(self, curves: int, b1: int, max_threads: int, gmp_ecm_path: Path) -> None:
        composite_factors = self.composite_factors.copy()
        self.composite_factors = []

        for n in composite_factors:
            factors = factor_ecm(n, curves, b1, max_threads, gmp_ecm_path)

            if len(factors) > 1:
                self.methods.add("ECM")

            for factor in factors:
                if is_prime(factor):
                    self.prime_factors.append(factor)
                else:
                    self.composite_factors.append(factor)
