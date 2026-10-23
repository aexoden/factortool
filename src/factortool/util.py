# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Utility functions for factortool."""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import sys
import tempfile

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

import gmpy2

from loguru import logger
from tap import Tap

# Trial factoring divides by every prime below this.
SMALL_PRIME_LIMIT = 10**6


@cache
def small_primes() -> tuple[int, ...]:
    """List the prime numbers below SMALL_PRIME_LIMIT, generating them on first use.

    Returns:
        tuple[int, ...]: Tuple of prime numbers below SMALL_PRIME_LIMIT, in ascending order.
    """
    primes: list[int] = []
    p = gmpy2.mpz(2)

    while p < SMALL_PRIME_LIMIT:
        primes.append(int(p))
        p = gmpy2.next_prime(p)

    return tuple(primes)


# The log levels that can be selected, from the most to the least verbose.
LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_LOG_LEVEL: LogLevel = "INFO"


class ConfigArguments(Tap):
    """An argument parser for a utility that reads the configuration file, with options that override its settings."""

    config_path: Path = Path("config.json")  # Path to the configuration file
    log_level: LogLevel | None = None  # Minimum level of log messages to show (overrides log_level in config)
    max_threads: int | None = None  # Maximum number of threads to use (overrides max_threads in config)

    def configure(self) -> None:
        """Accept log levels in any case, as the configuration file does."""
        self.add_argument("--log_level", type=str.upper, choices=get_args(LogLevel))


# The width of a number truncated for display, including its digit count.
NUMBER_DISPLAY_WIDTH = 32


def format_number(n: int, *, truncate: bool = True) -> str:
    """Format a number for display, followed by its digit count.

    If truncate is True, a number that exceeds NUMBER_DISPLAY_WIDTH characters, including the digit count suffix, will
    be truncated in the middle.

    Returns:
        str: Formatted number string.
    """
    str_n = str(n)
    digits = len(str_n)
    suffix = f" <{digits}>"

    if truncate and digits + len(suffix) > NUMBER_DISPLAY_WIDTH:
        to_keep = NUMBER_DISPLAY_WIDTH - len("...") - len(suffix)
        left = math.ceil(to_keep / 2)
        right = to_keep - left

        return f"{str_n[:left]}...{str_n[-right:]}{suffix}"

    return f"{str_n}{suffix}"


@cache
def is_prime(n: int) -> bool:
    """Check if a number is prime using GMP's trial division, Baillie-PSW and Miller-Rabin tests.

    The result is certain for numbers below 2**64. Above that, there is no known composite counterexample.

    Only GMP versions of 6.2.0 and above include the BPSW portion of the test.

    Returns:
        bool: True if n is a probable prime, False otherwise.
    """
    return gmpy2.is_prime(n)


def log_factor_result(methods: Iterable[str], n: int, factors: list[int]) -> None:
    """Log the result of a factorization."""
    logger.info("{} -> {} = {}", ", ".join(methods), format_number(n), " * ".join(map(format_number, sorted(factors))))


def setup_logger(level: LogLevel | None = None) -> None:
    """Send log messages at or above the given level to standard error, replacing any previous setup.

    The level defaults to DEFAULT_LOG_LEVEL.
    """
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <magenta>{elapsed}</magenta> | "
        "<level>{level: <8}</level> | <level>{message}</level>"
    )

    logger.configure(handlers=[{"sink": sys.stderr, "format": logger_format, "level": level or DEFAULT_LOG_LEVEL}])


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


# The prefixes of the working directories each tool is given under work_path.
YAFU_WORK_DIR_PREFIX = "yafu-"
CADO_NFS_WORK_DIR_PREFIX = "cado-nfs-"

# Marks a working directory as one factortool created.
WORK_DIR_MARKER = ".factortool-work"


@contextlib.contextmanager
def get_work_dir(base_path: Path, ini_path: Path | None, prefix: str) -> Generator[Path]:
    """Provide an isolated working directory for a single YAFU or CADO-NFS invocation.

    YAFU writes session.log, factor.log, factor.json, siqs.dat and assorted NFS artifacts into its current working
    directory. Giving every invocation its own directory keeps those out of the YAFU installation and, because
    factortool can run several YAFU processes concurrently, prevents them from clobbering each other's files.

    CADO-NFS is pointed at the directory with --workdir, as it otherwise creates its own under the system temporary
    directory and leaves it behind whenever it fails or is killed. It doesn't need yafu.ini.

    Yields:
        Path: The working directory, removed once the context exits.

    Raises:
        OSError: If the directory cannot be created or marked as factortool's.
    """
    base_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=base_path, prefix=prefix) as name:
        work_dir = Path(name)
        (work_dir / WORK_DIR_MARKER).touch()

        if ini_path is not None and ini_path.is_file():
            _place_yafu_ini(ini_path, work_dir)

        yield work_dir


def remove_stale_work_dirs(base_path: Path) -> None:
    """Remove the working directories left behind by previous runs."""
    for prefix in (YAFU_WORK_DIR_PREFIX, CADO_NFS_WORK_DIR_PREFIX):
        for path in base_path.glob(f"{prefix}*"):
            if path.is_symlink() or path.is_junction() or not (path / WORK_DIR_MARKER).is_file():
                continue

            try:
                shutil.rmtree(path)
            except OSError as e:
                logger.warning("Failed to remove stale working directory {}: {}", path, e)
            else:
                logger.info("Removed stale working directory {}", path)


def safe_write(path: Path, data: bytes) -> None:
    """Safely write data to a file by using a temporary file and renaming it."""
    target_path = path.parent

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=target_path, suffix=".tmp") as f:
        f.write(data)

        temp_path = Path(f.name)
        f.flush()
        os.fsync(f.fileno())

    temp_path.replace(path)
