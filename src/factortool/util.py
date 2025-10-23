# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Utility functions for factortool."""

from __future__ import annotations

import math
import os
import sys
import tempfile

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

import gmpy2

from loguru import logger


def generate_primes(limit: int = 10**6) -> list[int]:
    """Generate a list of prime numbers up to the specified limit.

    Returns:
        list[int]: List of prime numbers less than the limit.
    """
    prime_flags = [True for _ in range(limit)]

    for i in range(2, limit):
        if prime_flags[i]:
            for j in range(2 * i, limit, i):
                prime_flags[j] = False

    return [i for i in range(2, limit) if prime_flags[i]]


SMALL_PRIMES: list[int] = generate_primes()


def format_number(n: int, max_width: int = 32) -> str:
    """Format a number for display, truncating if necessary.

    Returns:
        str: Formatted number string.
    """
    str_n = str(n)
    digits = len(str_n)

    if digits > max_width:
        to_keep = max_width - 3 - 3 - len(str(digits))
        left = math.ceil(to_keep / 2)
        right = to_keep - left

        return f"{str_n[:left]}...{str_n[-right:]} <{digits}>"

    return f"{str_n}"


@cache
def is_prime(n: int) -> bool:
    """Check if a number is prime using trial division and Miller-Rabin tests.

    This function is deterministic for numbers less than 3,317,044,064,679,887,385,961,981. Beyond that, it may be
    probabilistic.

    Returns:
        bool: True if n is a probable prime, False otherwise.
    """
    # Return False for n = 1
    if n < 2:  # noqa: PLR2004
        return False

    # Check for small prime factors.
    for p in SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False

    # Do a series of Miller-Rabin tests, based on bases reported by https://www.wikiwand.com/en/articles/Miller-Rabin
    test_thresholds: dict[int, list[int]] = {
        2_047: [2],
        1_373_653: [2, 3],
        25_326_001: [2, 3, 5],
        3_215_031_751: [2, 3, 5, 7],
        2_152_302_898_747: [2, 3, 5, 7, 11],
        3_474_749_660_383: [2, 3, 5, 7, 11, 13],
        341_550_071_728_321: [2, 3, 5, 7, 11, 13, 17],
        3_825_123_056_546_413_051: [2, 3, 57, 11, 13, 17, 19, 23],
        318_665_857_834_031_151_167_461: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
        3_317_044_064_679_887_385_961_981: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
    }

    test_bases = []

    for threshold, bases in test_thresholds.items():
        if n < threshold:
            test_bases = bases
            break

    # If the number is larger, just do a few extra bases. If we accidentally label a composite number prime, it doesn't
    # matter that much, as FactorDB will ultimately catch the composite factor. Searching for alternate lists of
    # required bases might be useful.
    if not test_bases:
        test_bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49]

    return all(gmpy2.is_strong_prp(n, base) for base in test_bases)


def log_factor_result(methods: Iterable[str], n: int, factors: list[int]) -> None:
    """Log the result of a factorization."""
    logger.info("{} -> {} = {}", ", ".join(methods), format_number(n), " * ".join(map(format_number, sorted(factors))))


def setup_logger() -> None:
    """Set up the logger for the application."""
    logger.remove(0)

    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <magenta>{elapsed}</magenta> | "
        "<level>{level: <8}</level> | <level>{message}</level>"
    )

    logger.add(sys.stdout, format=logger_format)


def safe_write(path: Path, data: bytes) -> None:
    """Safely write data to a file by using a temporary file and renaming it."""
    target_path = path.parent

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=target_path, suffix=".tmp") as f:
        f.write(data)

        temp_path = Path(f.name)
        f.flush()
        os.fsync(f.fileno())

    temp_path.rename(path)
