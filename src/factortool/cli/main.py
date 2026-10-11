# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024-2026 Jason Lynch <jason@aexoden.com>
"""Utility for factoring numbers using various methods."""

from __future__ import annotations

import datetime
import math
import sys
import time

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

import requests

from loguru import logger

from factortool.__about__ import __version__
from factortool.assignments import ASSIGNMENT_EXPIRY_FUDGE_FACTOR, AssignmentStore, select_unfinished
from factortool.backend import Backend, FetchCriteria, create_backend
from factortool.batch import BatchController, BatchKey
from factortool.config import read_config
from factortool.engine import ExitStatus, FactorEngine
from factortool.http import PermanentHttpError
from factortool.interrupt import EXIT_STATUS as INTERRUPTED_EXIT_STATUS
from factortool.interrupt import Interrupted, InterruptState
from factortool.number import Number, format_results
from factortool.stats import FactoringStats, InvalidStatsError
from factortool.tools import ToolError
from factortool.util import ArgumentParser, setup_logger

if TYPE_CHECKING:
    from factortool.config import Config

# How far past the target duration an automatically sized batch may run before the run is cut short.
TIME_LIMIT_FACTOR = 2.0

# The exit status of a run that is cut short by its time limit.
TIME_LIMIT_EXIT_STATUS = 3

# The exit status of a run with a permanent HTTP error.
PERMANENT_HTTP_ERROR_EXIT_STATUS = 6

# The exit status of a run that could not save its state or results or shut down its backend.
CLEANUP_FAILED_EXIT_STATUS = 7

# Exit statuses that do not report an error, and so give way to a cleanup failure.
NON_ERROR_EXIT_STATUSES = frozenset({0, INTERRUPTED_EXIT_STATUS, TIME_LIMIT_EXIT_STATUS})


class Arguments(ArgumentParser):
    """Utility for factoring numbers using various methods."""

    version: bool = False  # Show the version of the utility and exit
    config_path: Path = Path("config.json")  # Path to the JSON-formatted configuration file
    min_digits: int = 1  # Minimum number of digits fetched composite numbers should have
    max_digits: int = 0  # Maximum number of digits fetched composite numbers should have (required by mersenne.ca)
    batch_size: int = 0  # Number of composite numbers to work on at a time (0 for automatic, with a time limit)
    target_duration: float = 600.0  # Target duration in seconds for each batch (only used when batch_size is 0)
    skip_count: int = 0  # Skip this many numbers when fetching from FactorDB (to hopefully avoid conflict)
    no_new_work: bool = False  # Do not fetch new work from the backend. Only supported by mersenne.ca.


class Cleanup:
    """Runs cleanup steps independently."""

    def __init__(self) -> None:
        """Initialize the cleanup."""
        self.failed = False

    def run[**P](self, description: str, step: Callable[P, object], /, *args: P.args, **kwargs: P.kwargs) -> None:
        """Run a step, logging a failure instead of raising it."""
        try:
            step(*args, **kwargs)
        except OSError as e:
            logger.error("Failed to {}: {}", description, e)
            self.failed = True
        except Exception:  # ruff: ignore[blind-except] (A bug in one step must not skip the rest)
            logger.exception("Failed to {}", description)
            self.failed = True


@dataclass(frozen=True)
class Session:
    """State that lasts for a factoring session, rather than a single batch."""

    config: Config
    stats: FactoringStats
    backend: Backend
    interrupts: InterruptState
    cleanup: Cleanup = field(default_factory=Cleanup)


def validate_argument_ranges(args: Arguments) -> None:
    """Reject arguments that are out of range."""
    if args.min_digits < 1:
        logger.error("--min_digits ({}) must be at least 1", args.min_digits)
        sys.exit(1)

    if args.max_digits < 0:
        logger.error("--max_digits ({}) cannot be negative", args.max_digits)
        sys.exit(1)

    if 0 < args.max_digits < args.min_digits:
        logger.error("--max_digits ({}) is below --min_digits ({})", args.max_digits, args.min_digits)
        sys.exit(1)

    if args.batch_size < 0:
        logger.error("--batch_size ({}) cannot be negative", args.batch_size)
        sys.exit(1)

    if not (math.isfinite(args.target_duration) and args.target_duration > 0):
        logger.error("--target_duration ({}) must be a positive number of seconds", args.target_duration)
        sys.exit(1)

    if args.skip_count < 0:
        logger.error("--skip_count ({}) cannot be negative", args.skip_count)
        sys.exit(1)


def validate_arguments(args: Arguments, backend_name: str) -> None:
    """Reject arguments that are out of range and combinations the selected backend cannot honor."""
    validate_argument_ranges(args)

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


def validate_tools(config: Config) -> None:
    """Reject a configuration whose external tools are missing, before any work is fetched."""
    problems = config.find_tool_problems()

    for problem in problems:
        logger.error("Configuration error: {}", problem)

    if problems:
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

    Raises:
        PermanentHttpError: If the backend reports a permanent error while fetching.
    """
    numbers: set[Number] = set()

    if backend.assigns_work:
        numbers = {Number(n, config, stats, backend) for n in assignments.load()}

    remaining = criteria.count - len(numbers)

    if fetch and remaining > 0:
        logger.info("Fetching {} composite numbers from {}", remaining, config.backend)
        fetched = backend.fetch(replace(criteria, count=remaining))

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
    except Interrupted:
        logger.warning("Interrupted while starting the {} backend", config.backend)
        sys.exit(INTERRUPTED_EXIT_STATUS)
    except requests.RequestException as e:
        logger.error("Unable to start the {} backend: {}", config.backend, e)
        sys.exit(6)


def get_time_limit(args: Arguments) -> float | None:
    """Determine how long the factoring may run before the run is cut short.

    Returns:
        float | None: The time limit in seconds, or None if there is no time limit.
    """
    if args.batch_size > 0:
        return None

    return TIME_LIMIT_FACTOR * args.target_duration


def warn_if_assignments_may_expire(backend: Backend, time_limit: float | None) -> None:
    """Warn if a run may outlast the assignments it fetches."""
    if time_limit is None or not backend.assigns_work:
        return

    if time_limit + ASSIGNMENT_EXPIRY_FUDGE_FACTOR > backend.assignment_lifetime:
        logger.warning("With a time limit of {:.0f}s, assignments may expire before completion", time_limit)


def preserve_unfinished(backend: Backend, assignments: AssignmentStore, numbers: Collection[Number]) -> None:
    """Submit or preserve unfinished assignments.

    Partial work is submitted, and untouched assignments are preserved for the next run if the backend assigns work.
    """
    partial, untouched = select_unfinished(numbers)

    for number in partial:
        number.report_partial()

    if backend.assigns_work:
        assignments.save(x.n for x in untouched)


def run_batch(args: Arguments, session: Session) -> int:
    """Acquire a batch of numbers, factor them, and save state and results.

    Returns:
        int: The exit status for the batch, not counting any cleanup failure.
    """
    config, stats, backend = session.config, session.stats, session.backend
    time_limit = get_time_limit(args)

    max_digits = args.max_digits if args.max_digits > 0 else None
    batch_controller = BatchController(
        args.target_duration,
        BatchKey(backend=config.backend, min_digits=args.min_digits, max_digits=max_digits, skip_count=args.skip_count),
        config.batch_state_path,
    )
    batch_size = args.batch_size if args.batch_size > 0 else batch_controller.batch_size

    if args.no_new_work:
        logger.info("As requested, not fetching new work")

    warn_if_assignments_may_expire(backend, time_limit)

    assignments = AssignmentStore(config.assignment_state_path, config.backend, backend.assignment_lifetime)

    try:
        numbers = acquire_numbers(
            backend,
            assignments,
            config,
            stats,
            FetchCriteria(
                count=batch_size, min_digits=args.min_digits, max_digits=max_digits, skip_count=args.skip_count
            ),
            fetch=not args.no_new_work,
        )
    except PermanentHttpError as e:
        logger.error("Unable to fetch numbers: {}", e)
        return PERMANENT_HTTP_ERROR_EXIT_STATUS

    if not numbers:
        logger.warning("No numbers to factor")
        return INTERRUPTED_EXIT_STATUS if session.interrupts.interrupted else 0

    status = ExitStatus.SUCCESS
    tool_error: ToolError | None = None
    start_time = time.monotonic()

    # Each number retains its own state, so we can safely process them independently, regardless of what happens in the
    # engine.
    try:
        status = FactorEngine(config, session.interrupts).run(sorted(numbers), time_limit)
    except ToolError as e:
        logger.critical("{}", e)
        tool_error = e
    finally:
        duration = time.monotonic() - start_time
        factored_count = len([number for number in numbers if number.factored])

        logger.info("Factored {} numbers in {:.2f} seconds", factored_count, duration)

        cleanup = session.cleanup

        # Record the batch only if new work was fetched.
        if not args.no_new_work:
            cleanup.run("record the batch", batch_controller.record_batch, factored_count, duration)

        cleanup.run("preserve unfinished work", preserve_unfinished, backend, assignments, numbers)
        cleanup.run("report summary", report_summary, numbers)
        cleanup.run("write results", write_results, numbers, config.result_output_path)
        cleanup.run("save statistics", stats.save_data)

    if tool_error is not None:
        return tool_error.exit_status

    if status == ExitStatus.INTERRUPTED or session.interrupts.interrupted:
        return INTERRUPTED_EXIT_STATUS

    if status == ExitStatus.TIME_LIMIT_EXCEEDED:
        return TIME_LIMIT_EXIT_STATUS

    return 0


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
    validate_tools(config)

    interrupts = InterruptState()
    interrupts.install()

    try:
        stats = FactoringStats(config.stats_path)
    except InvalidStatsError as e:
        logger.error("{}", e)
        sys.exit(1)

    session = Session(config, stats, start_backend(config, stats, interrupts), interrupts)

    logger.info("Using backend: {}", config.backend)
    logger.info("Using factoring mode: {}", config.factoring_mode)

    # The backend is closed last, as flushing its pending submissions can take some time.
    try:
        status = run_batch(args, session)
    finally:
        session.cleanup.run("close the backend", session.backend.close)

    # An interrupt may have arrived while the backend was flushing its submissions.
    if interrupts.interrupted and status in NON_ERROR_EXIT_STATUSES:
        status = INTERRUPTED_EXIT_STATUS

    if session.cleanup.failed and status in NON_ERROR_EXIT_STATUSES:
        status = CLEANUP_FAILED_EXIT_STATUS

    if status != 0:
        sys.exit(status)
