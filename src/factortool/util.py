# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Utility functions for factortool."""

from __future__ import annotations

import contextlib
import math
import os
import sys
import tempfile

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

import gmpy2

from loguru import logger
from tap import Tap


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

# The exit status of a run that was given an invalid configuration or invalid arguments.
INVALID_USAGE_EXIT_STATUS = 1


class ArgumentParser(Tap):
    """An argument parser that reports invalid arguments with the same exit status as an invalid configuration."""

    def error(self, message: str) -> NoReturn:
        """Report invalid arguments and exit, without argparse's exit status of 2."""
        self.print_usage(sys.stderr)
        self.exit(INVALID_USAGE_EXIT_STATUS, f"{self.prog}: error: {message}\n")


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
    if n < 2:  # ruff:ignore[magic-value-comparison]
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
        3_825_123_056_546_413_051: [2, 3, 5, 7, 11, 13, 17, 19, 23],
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
        test_bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

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


# Options in yafu.ini that represent file or directory paths.
YAFU_INI_PATH_OPTIONS = frozenset({"cado_dir", "convert_poly_path", "ecm_path", "ggnfs_dir"})


def rewrite_yafu_ini(ini_text: str, ini_dir: Path) -> str:
    """Rewrite relative path options in a yafu.ini file to still be valid from a different working directory.

    Returns:
        str: The rewritten ini text with updated relative paths.
    """
    lines: list[str] = []

    for line in ini_text.splitlines():
        key, separator, raw_value = line.partition("=")
        value = raw_value.strip()

        # Paths with an anchor are left alone, even on Windows.
        if separator and key.strip() in YAFU_INI_PATH_OPTIONS and value and not Path(value).anchor:
            # YAFU requires directory options to retain their trailing separator.
            trailing = os.sep if value.endswith(("/", "\\")) else ""
            lines.append(f"{key}={(ini_dir / value).resolve()}{trailing}")
        else:
            lines.append(line)

    return "\n".join(lines) + "\n"


def _place_yafu_ini(ini_path: Path, work_dir: Path) -> None:
    """Write a working-directory-independent copy of yafu.ini into a YAFU working directory.

    The ini is rewritten rather than symlinked because its relative path options are only meaningful in the original
    location.
    """
    try:
        ini_text = ini_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to read {}: {}. YAFU will use its built-in defaults", ini_path, e)
        return

    try:
        (work_dir / "yafu.ini").write_text(rewrite_yafu_ini(ini_text, ini_path.parent), encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to write yafu.ini in {}: {}. YAFU will use its built-in defaults", work_dir, e)


@contextlib.contextmanager
def get_work_dir(base_path: Path, ini_path: Path | None, prefix: str) -> Generator[Path]:
    """Provide an isolated working directory for a single YAFU or CADO-NFS invocation.

    YAFU writes session.log, factor.log, factor.json, siqs.dat and assorted NFS artifacts into its current working
    directory. Giving every invocation its own directory keeps those out of the YAFU installation and, because
    factortool can run several YAFU processes concurrently, prevents them from clobbering each other's files.

    CADO-NFS also writes artifacts into its working directory, but it doesn't need yafu.ini.

    Yields:
        Path: The working directory, removed once the context exits.
    """
    base_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=base_path, prefix=prefix) as name:
        work_dir = Path(name)

        if ini_path is not None and ini_path.is_file():
            _place_yafu_ini(ini_path, work_dir)

        yield work_dir


def safe_write(path: Path, data: bytes) -> None:
    """Safely write data to a file by using a temporary file and renaming it."""
    target_path = path.parent

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=target_path, suffix=".tmp") as f:
        f.write(data)

        temp_path = Path(f.name)
        f.flush()
        os.fsync(f.fileno())

    temp_path.replace(path)
