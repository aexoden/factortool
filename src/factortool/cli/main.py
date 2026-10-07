# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024-2026 Jason Lynch <jason@aexoden.com>
"""Utility for factoring numbers using various methods."""

from __future__ import annotations

import datetime
import sys
import time

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

import requests

from loguru import logger
from tap import Tap

from factortool.__about__ import __version__
from factortool.assignments import ASSIGNMENT_EXPIRY_FUDGE_FACTOR, AssignmentStore, select_unfinished
from factortool.backend import Backend, FetchCriteria, create_backend
from factortool.batch import BatchController, BatchKey
from factortool.config import read_config
from factortool.engine import ExitStatus, FactorEngine
from factortool.http import PermanentHttpError
from factortool.interrupt import InterruptState
from factortool.number import Number, format_results
from factortool.stats import FactoringStats
from factortool.util import setup_logger

if TYPE_CHECKING:
    from factortool.config import Config


class Arguments(Tap):
    """Utility for factoring numbers using various methods."""

    version: bool = False  # Show the version of the utility and exit
    config_path: Path = Path("config.json")  # Path to the JSON-formatted configuration file
    min_digits: int = 1  # Minimum number of digits fetched composite numbers should have
    max_digits: int = 0  # Maximum number of digits fetched composite numbers should have (required by mersenne.ca)
    batch_size: int = 0  # Number of composite numbers to work on at a time (0 for automatic)
    target_duration: float = 600.0  # Target duration in seconds for each batch (only used when batch_size is 0)
    skip_count: int = 0  # Skip this many numbers when fetching from FactorDB (to hopefully avoid conflict)
    no_new_work: bool = False  # Do not fetch new work from the backend. Only supported by mersenne.ca.


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
    elif args.no_new_work:
        # FactorDB doesn't assign work, and we don't retain unfinished work, so this option would have no effect.
        logger.error("The FactorDB backend does not support --no_new_work, since it does not assign work.")
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


def acquire_numbers(  # ruff: ignore[too-many-arguments]
    backend: Backend,
    assignments: AssignmentStore,
    config: Config,
    stats: FactoringStats,
    criteria: FetchCriteria,
    *,
    fetch: bool = True,
) -> set[Number]:
    """Collect a batch of numbers to work on by resuming retained assignments and optionally fetching new work.

    Returns:
        set[Number]: A set of numbers to factor, which may be empty.
    """
    numbers: set[Number] = set()

    if backend.assigns_work:
        numbers = {Number(n, config, stats, backend) for n in assignments.load()}

    remaining = criteria.count - len(numbers)

    if fetch and remaining > 0:
        logger.info("Fetching {} composite numbers from {}", remaining, config.backend)
        fetched = fetch_numbers(backend, replace(criteria, count=remaining))

        if backend.assigns_work:
            assignments.note_assigned(x.n for x in fetched)

        numbers |= fetched

    if backend.assigns_work:
        for number in numbers:
            number.expires_at = assignments.expires_at(number.n)

    return numbers


def start_backend(config: Config, stats: FactoringStats, interrupts: InterruptState) -> Backend:
    """Create the configured backend, exiting if it cannot start.

    Returns:
        Backend: The configured backend.
    """
    try:
        return create_backend(config, stats, interrupts)
    except requests.RequestException as e:
        logger.error("Unable to start the {} backend: {}", config.backend, e)
        sys.exit(6)


def fetch_numbers(backend: Backend, criteria: FetchCriteria) -> set[Number]:
    """Fetch new work, exiting if the backend reports a permanent error.

    Returns:
        set[Number]: The fetched numbers.
    """
    try:
        return backend.fetch(criteria)
    except PermanentHttpError as e:
        logger.error("Unable to fetch numbers: {}", e)
        backend.close()
        sys.exit(6)


def warn_if_assignments_may_expire(backend: Backend, target_duration: float) -> None:
    """Warn if a run may outlast the assignments it fetches."""
    if backend.assigns_work and 2.0 * target_duration + ASSIGNMENT_EXPIRY_FUDGE_FACTOR > backend.assignment_lifetime:
        logger.warning("With a target duration of {:.0f}s, assignments may expire before completion", target_duration)


def preserve_unfinished(backend: Backend, assignments: AssignmentStore, numbers: Collection[Number]) -> None:
    """Submit or preserve unfinished assignments.

    Partial work is submitted, and untouched assignments are preserved for the next run if the backend assigns work.
    """
    partial, untouched = select_unfinished(numbers)

    for number in partial:
        number.report_partial()

    if backend.assigns_work:
        assignments.save(x.n for x in untouched)


def main() -> None:
    """Factor numbers using various methods."""
    setup_logger()

    args = Arguments().parse_args()

    if args.version:
        print(f"Factortool version: {__version__}")  # ruff: ignore[print]
        sys.exit(0)

    try:
        config = read_config(args.config_path)
    except FileNotFoundError:
        logger.error("Configuration file not found")
        sys.exit(1)

    validate_arguments(args, config.backend)

    interrupts = InterruptState()
    interrupts.install()

    stats = FactoringStats(config.stats_path)
    backend = start_backend(config, stats, interrupts)
    engine = FactorEngine(config, args.target_duration, interrupts)

    logger.info("Using backend: {}", config.backend)
    logger.info("Using factoring mode: {}", config.factoring_mode)

    max_digits = args.max_digits if args.max_digits > 0 else None
    batch_controller = BatchController(
        args.target_duration,
        BatchKey(backend=config.backend, min_digits=args.min_digits, max_digits=max_digits, skip_count=args.skip_count),
        config.batch_state_path,
    )
    batch_size = args.batch_size if args.batch_size > 0 else batch_controller.batch_size

    if args.no_new_work:
        logger.info("As requested, not fetching new work")

    warn_if_assignments_may_expire(backend, args.target_duration)

    assignments = AssignmentStore(config.assignment_state_path, config.backend, backend.assignment_lifetime)
    numbers = acquire_numbers(
        backend,
        assignments,
        config,
        stats,
        FetchCriteria(count=batch_size, min_digits=args.min_digits, max_digits=max_digits, skip_count=args.skip_count),
        fetch=not args.no_new_work,
    )

    if not numbers:
        logger.warning("No numbers to factor")
        backend.close()
        sys.exit(2 if interrupts.interrupted else 0)

    start_time = time.monotonic()

    # Each number retains its own state, so we can safely process them independently, regardless of what happens in the
    # engine.
    try:
        status = engine.run(sorted(numbers))
    finally:
        duration = time.monotonic() - start_time
        factored_count = len([number for number in numbers if number.factored])

        logger.info("Factored {} numbers in {:.2f} seconds", factored_count, duration)

        # Record the batch only if new work was fetched.
        if not args.no_new_work:
            batch_controller.record_batch(factored_count, duration)

        preserve_unfinished(backend, assignments, numbers)

        report_summary(numbers)
        write_results(numbers, config.result_output_path)

        stats.save_data()
        backend.close()

    if status == ExitStatus.INTERRUPTED or interrupts.interrupted:
        sys.exit(2)

    if status == ExitStatus.TIME_LIMIT_EXCEEDED:
        sys.exit(3)
