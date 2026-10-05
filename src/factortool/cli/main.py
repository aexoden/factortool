# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024-2026 Jason Lynch <jason@aexoden.com>
"""Utility for factoring numbers using various methods."""

from __future__ import annotations

import datetime
import sys
import time

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

from loguru import logger
from tap import Tap

from factortool.backend import FetchCriteria, create_backend
from factortool.batch import BatchController, BatchKey
from factortool.config import read_config
from factortool.engine import ExitStatus, FactorEngine
from factortool.number import Number, format_results
from factortool.stats import FactoringStats
from factortool.util import setup_logger


class Arguments(Tap):
    """Utility for factoring numbers using various methods."""

    config_path: Path = Path("config.json")  # Path to the JSON-formatted configuration file
    min_digits: int = 1  # Minimum number of digits fetched composite numbers should have
    max_digits: int = 0  # Maximum number of digits fetched composite numbers should have (required by mersenne.ca)
    batch_size: int = 0  # Number of composite numbers to work on at a time (0 for automatic)
    target_duration: float = 600.0  # Target duration in seconds for each batch (only used when batch_size is 0)
    skip_count: int = 0  # Skip this many numbers when fetching from FactorDB (to hopefully avoid conflict)


def validate_arguments(args: Arguments, backend_name: str) -> None:
    """Reject argument combinations the selected backend cannot honor."""
    if args.max_digits < 0:
        logger.error("--max_digits ({}) cannot be negative", args.max_digits)
        sys.exit(1)

    if 0 < args.max_digits < args.min_digits:
        logger.error("--max_digits ({}) is below --min_digits ({})", args.max_digits, args.min_digits)
        sys.exit(1)

    if backend_name == "mersenne_ca":
        if args.max_digits <= 0:
            logger.error("The mersenne.ca backend requires --max_digits")
            sys.exit(1)

        if args.skip_count != 0:
            logger.error("The mersenne.ca backend does not support --skip_count, since it assigns distinct work")
            sys.exit(1)


def report_summary(numbers: Collection[Number]) -> None:
    """Log which methods were used and which numbers were left unfactored."""
    method_counts: dict[str, int] = {}
    failed_numbers: set[Number] = set()

    for number in numbers:
        for method in set(number.methods):
            if method not in method_counts:
                method_counts[method] = 0

            method_counts[method] += 1

        if not number.factored:
            failed_numbers.add(number)

    if len(failed_numbers) > 0:
        logger.warning(
            "{} numbers failed to factor: {}",
            len(failed_numbers),
            ", ".join(str(x.n) for x in sorted(failed_numbers)),
        )

    logger.info(
        "Factored {} numbers and there were {} failures",
        len(numbers) - len(failed_numbers),
        len(failed_numbers),
    )

    logger.info(
        "The following methods were used: {}",
        ", ".join(f"{method} ({count})" for method, count in method_counts.items()),
    )


def write_results(numbers: Collection[Number], output_path: Path) -> None:
    """Write the run's factorizations to a timestamped file."""
    if not output_path.exists():
        output_path.mkdir(parents=True)

    output_filename = f"{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d-%H%M%S')}.txt"

    with output_path.joinpath(output_filename).open("w", encoding="utf-8") as f:
        f.write(format_results(numbers) + "\n")


def main() -> None:
    """Factor numbers using various methods."""
    setup_logger()

    args = Arguments().parse_args()

    try:
        config = read_config(args.config_path)
    except FileNotFoundError:
        logger.error("Configuration file not found")
        sys.exit(1)

    validate_arguments(args, config.backend)

    stats = FactoringStats(config.stats_path)
    backend = create_backend(config, stats)
    engine = FactorEngine(config, args.target_duration)

    logger.info("Using backend: {}", config.backend)
    logger.info("Using factoring mode: {}", config.factoring_mode)

    max_digits = args.max_digits if args.max_digits > 0 else None
    batch_controller = BatchController(
        args.target_duration,
        BatchKey(backend=config.backend, min_digits=args.min_digits, max_digits=max_digits, skip_count=args.skip_count),
        config.batch_state_path,
    )
    batch_size = args.batch_size if args.batch_size > 0 else batch_controller.batch_size

    logger.info("Fetching {} composite numbers from {}", batch_size, config.backend)

    numbers = backend.fetch(
        FetchCriteria(count=batch_size, min_digits=args.min_digits, max_digits=max_digits, skip_count=args.skip_count)
    )

    if not numbers:
        logger.warning("No numbers to factor")
        backend.close()
        sys.exit(0)

    start_time = time.monotonic()

    status = engine.run(sorted(numbers))

    duration = time.monotonic() - start_time
    factored_count = len([number for number in numbers if number.factored])

    logger.info("Factored {} numbers in {:.2f} seconds", factored_count, duration)

    batch_controller.record_batch(factored_count, duration)

    report_summary(numbers)
    write_results(numbers, config.result_output_path)

    stats.save_data()
    backend.close()

    if status == ExitStatus.SUCCESS:
        sys.exit(0)
    elif status == ExitStatus.INTERRUPTED:
        sys.exit(2)
    elif status == ExitStatus.TIME_LIMIT_EXCEEDED:
        sys.exit(3)
