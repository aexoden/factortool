# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024-2026 Jason Lynch <jason@aexoden.com>
"""Utility for factoring numbers using various methods."""

from __future__ import annotations

import datetime as dt
import math
import sys
import time

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import contextlib

    from collections.abc import Callable, Collection
    from pathlib import Path

from loguru import logger

from factortool.__about__ import __version__
from factortool.backend import Backend, FetchCriteria, create_backend
from factortool.batch import BatchController, BatchKey
from factortool.config import BackendName, FactoringMode, load_config
from factortool.engine import EngineStatus, FactorEngine, TimeLimits
from factortool.exit_status import ExitStatus
from factortool.http import PermanentHttpError
from factortool.interrupt import Interrupted, InterruptState
from factortool.lock import DirectoryInUseError, lock_directories
from factortool.number import Number, Outcome, format_results
from factortool.stats import FactoringStats, InvalidStatsError
from factortool.tools import ToolError, tool_failures
from factortool.util import ConfigArguments, format_number, remove_stale_work_dirs, setup_logger

if TYPE_CHECKING:
    from factortool.config import Config

# How far past the target duration an automatically sized batch may run by default before no more work is started.
TIME_LIMIT_FACTOR = 2.0

# Exit statuses that do not report an error, and so give way to a cleanup failure.
NON_ERROR_EXIT_STATUSES = frozenset({ExitStatus.SUCCESS, ExitStatus.INTERRUPTED})


class Arguments(ConfigArguments):
    """Utility for factoring numbers using various methods."""

    version: bool = False  # Show the version of the utility and exit
    backend: BackendName | None = None  # Backend to fetch numbers from and submit results to (overrides config backend)
    factoring_mode: FactoringMode | None = None  # Factoring mode to use (overrides config factoring_mode)
    min_digits: int = 1  # Minimum number of digits fetched composite numbers should have
    max_digits: int = 0  # Maximum number of digits fetched composite numbers should have (required by mersenne.ca)
    batch_size: int = 0  # Number of composite numbers to work on at a time (0 for automatic)
    target_duration: float = 600.0  # Target duration in seconds for each batch (only used when batch_size is 0)
    time_limit: float | None = None  # Seconds after which no new factorizations are started (0 for no limit)
    hard_time_limit: float = 0.0  # Seconds after which running factorizations are abandoned (0 for no limit)
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
        sys.exit(ExitStatus.INVALID_INPUT)

    if args.max_digits < 0:
        logger.error("--max_digits ({}) cannot be negative", args.max_digits)
        sys.exit(ExitStatus.INVALID_INPUT)

    if 0 < args.max_digits < args.min_digits:
        logger.error("--max_digits ({}) is below --min_digits ({})", args.max_digits, args.min_digits)
        sys.exit(ExitStatus.INVALID_INPUT)

    if args.batch_size < 0:
        logger.error("--batch_size ({}) cannot be negative", args.batch_size)
        sys.exit(ExitStatus.INVALID_INPUT)

    if not (math.isfinite(args.target_duration) and args.target_duration > 0):
        logger.error("--target_duration ({}) must be a positive number of seconds", args.target_duration)
        sys.exit(ExitStatus.INVALID_INPUT)

    for option, limit in (("--time_limit", args.time_limit), ("--hard_time_limit", args.hard_time_limit)):
        if limit is not None and not (math.isfinite(limit) and limit >= 0):
            logger.error("{} ({}) must be a number of seconds, or 0 for no limit", option, limit)
            sys.exit(ExitStatus.INVALID_INPUT)

    if args.skip_count < 0:
        logger.error("--skip_count ({}) cannot be negative", args.skip_count)
        sys.exit(ExitStatus.INVALID_INPUT)


def validate_arguments(args: Arguments, backend_name: str) -> None:
    """Reject arguments that are out of range and combinations the selected backend cannot honor."""
    validate_argument_ranges(args)

    if backend_name == "mersenne_ca":
        if args.max_digits <= 0:
            logger.error("The mersenne.ca backend requires --max_digits")
            sys.exit(ExitStatus.INVALID_INPUT)

        if args.skip_count != 0:
            logger.error("The mersenne.ca backend does not support --skip_count, since it assigns distinct work")
            sys.exit(ExitStatus.INVALID_INPUT)
    elif args.no_new_work:
        # FactorDB doesn't assign work, and we don't retain unfinished work, so this option would have no effect.
        logger.error("The FactorDB backend does not support --no_new_work, since it does not assign work.")
        sys.exit(ExitStatus.INVALID_INPUT)


def validate_tools(config: Config) -> None:
    """Reject a configuration whose external tools are missing, before any work is fetched."""
    problems = config.find_tool_problems()

    for problem in problems:
        logger.error("Configuration error: {}", problem)

    if problems:
        sys.exit(ExitStatus.INVALID_INPUT)


def report_summary(numbers: Collection[Number]) -> None:
    """Log a summary of the factoring results, including method usage and failures."""
    method_counts: dict[str, int] = {}

    for number in numbers:
        for method in set(number.methods):
            if method not in method_counts:
                method_counts[method] = 0

            method_counts[method] += 1

    outcomes = Counter(number.outcome for number in numbers)
    failed_numbers = sorted(number for number in numbers if number.outcome is Outcome.FAILED)

    if len(failed_numbers) > 0:
        logger.warning(
            "{} number{} failed to factor: {}",
            len(failed_numbers),
            "s" if len(failed_numbers) != 1 else "",
            ", ".join(format_number(x.n, truncate=False) for x in failed_numbers),
        )

    unfactored = [
        f"{outcomes[outcome]} {outcome.value}"
        for outcome in Outcome
        if outcome is not Outcome.FACTORED and outcomes[outcome]
    ]

    logger.info(
        "Factored {} of {} number{}{}",
        outcomes[Outcome.FACTORED],
        len(numbers),
        "s" if len(numbers) != 1 else "",
        f" ({', '.join(unfactored)})" if unfactored else "",
    )

    if method_counts:
        logger.info(
            "The following methods were used: {}",
            ", ".join(f"{method} ({count})" for method, count in method_counts.items()),
        )


def write_results(numbers: Collection[Number], output_path: Path) -> None:
    """Write the factorizations the run made progress on to a timestamped file."""
    results = format_results(number for number in numbers if number.methods)

    if not results:
        return

    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d-%H%M%S")
    attempt = 0

    # Instances with separate state can share an output directory.
    while True:
        output_filename = f"{timestamp}-{attempt}.txt" if attempt > 0 else f"{timestamp}.txt"

        try:
            with output_path.joinpath(output_filename).open("x", encoding="utf-8") as f:
                f.write(results)
        except FileExistsError:
            attempt += 1
        else:
            return


def acquire_numbers(backend: Backend, criteria: FetchCriteria, *, fetch: bool = True) -> set[Number]:
    """Collect a batch of numbers to work on by resuming retained assignments and optionally fetching new work.

    Returns:
        set[Number]: A set of numbers to factor, which may be empty.

    Raises:
        PermanentHttpError: If the backend reports a permanent error while fetching.
    """
    numbers = backend.resume()
    remaining = criteria.count - len(numbers)

    if fetch and remaining > 0:
        numbers |= backend.fetch(replace(criteria, count=remaining))

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
        sys.exit(ExitStatus.INTERRUPTED)
    except PermanentHttpError as e:
        logger.error("Unable to start the {} backend: {}", config.backend, e)
        sys.exit(ExitStatus.PERMANENT_HTTP_ERROR)


def lock_instance(config: Config) -> contextlib.ExitStack:
    """Claim the state and working directories, exiting if another instance is using either.

    Returns:
        contextlib.ExitStack: Releases the directories when closed.
    """
    try:
        return lock_directories([config.state_path, config.work_path])
    except DirectoryInUseError as e:
        logger.error("{}", e)
        sys.exit(ExitStatus.ALREADY_RUNNING)
    except OSError as e:
        logger.error("Unable to lock the state and working directories: {}", e)
        sys.exit(ExitStatus.INVALID_INPUT)


def get_time_limits(args: Arguments, backend: Backend) -> TimeLimits:
    """Determine how long the factoring may run before the run is cut short.

    Unless a time limit is given, an automatically sized batch stops starting factorizations at a multiple of its target
    duration. A backend that assigns work needs no such limit.

    Returns:
        TimeLimits: The time limits for the factoring run, including the soft and hard limits.
    """
    hard = args.hard_time_limit or None

    if args.time_limit is not None:
        return TimeLimits(soft=args.time_limit or None, hard=hard)

    if args.batch_size > 0 or backend.assigns_work:
        return TimeLimits(hard=hard)

    return TimeLimits(soft=TIME_LIMIT_FACTOR * args.target_duration, hard=hard)


def warn_if_assignments_may_expire(backend: Backend, args: Arguments) -> None:
    """Warn if a run may outlast the assignments it fetches."""
    if args.batch_size > 0 or not backend.assigns_work:
        return

    if args.target_duration > backend.assignment_lifetime:
        logger.warning(
            "With a target duration of {:.0f}s, assignments may expire before completion", args.target_duration
        )


def report_unfinished(backend: Backend, numbers: Collection[Number]) -> None:
    """Report the partial factorizations that cannot be resumed if the backend accepts them, and discard them otherwise.

    A number whose assignment is still held is left alone, as the backend retains it for the next run.
    """
    for number in numbers:
        if not number.partially_factored or (number.expires_at is not None and not number.assignment_expired):
            continue

        if backend.accepts_partial_results:
            number.report_partial()
        else:
            logger.warning(
                "Discarding the partial factorization of {}, as it was not completed before its assignment expired",
                format_number(number.n, truncate=False),
            )


def run_batch(args: Arguments, session: Session) -> ExitStatus:
    """Acquire a batch of numbers, factor them, and save state and results.

    Returns:
        ExitStatus: The exit status for the batch, not counting any cleanup failure.
    """
    config, stats, backend = session.config, session.stats, session.backend
    limits = get_time_limits(args, backend)

    max_digits = args.max_digits if args.max_digits > 0 else None
    batch_controller = BatchController(
        args.target_duration,
        BatchKey(
            backend=config.backend,
            factoring_mode=config.factoring_mode,
            max_threads=config.max_threads,
            min_digits=args.min_digits,
            max_digits=max_digits,
            skip_count=args.skip_count,
        ),
        config.batch_state_path,
    )
    batch_size = args.batch_size if args.batch_size > 0 else batch_controller.batch_size

    if args.no_new_work:
        logger.info("As requested, not fetching new work")

    warn_if_assignments_may_expire(backend, args)

    try:
        numbers = acquire_numbers(
            backend,
            FetchCriteria(
                count=batch_size, min_digits=args.min_digits, max_digits=max_digits, skip_count=args.skip_count
            ),
            fetch=not args.no_new_work,
        )
    except PermanentHttpError as e:
        logger.error("Unable to fetch numbers: {}", e)
        return ExitStatus.PERMANENT_HTTP_ERROR

    if not numbers:
        logger.warning("No numbers to factor")
        return ExitStatus.INTERRUPTED if session.interrupts.interrupted else ExitStatus.SUCCESS

    status = EngineStatus.SUCCESS
    tool_failed = False
    start_time = time.monotonic()

    # Each number retains its own state, so we can safely process them independently, regardless of what happens in the
    # engine.
    try:
        status = FactorEngine(config, session.interrupts).run(sorted(numbers), limits)
    except ToolError as e:
        logger.critical("{}", e)
        tool_failed = True
    finally:
        duration = time.monotonic() - start_time
        factored_count = len([number for number in numbers if number.factored])

        logger.info("Factored {} numbers in {:.2f} seconds", factored_count, duration)

        cleanup = session.cleanup

        # Record the batch only if new work was fetched.
        if not args.no_new_work:
            cleanup.run("record the batch", batch_controller.record_batch, factored_count, duration)

        cleanup.run("report unfinished work", report_unfinished, backend, numbers)
        cleanup.run("report summary", report_summary, numbers)
        cleanup.run("write results", write_results, numbers, config.result_output_path)
        cleanup.run("save statistics", stats.save_data)
        cleanup.run("save tool state", tool_failures.save, config.tool_state_path)

    if tool_failed:
        return ExitStatus.TOOL_FAILURE

    if status == EngineStatus.INTERRUPTED or session.interrupts.interrupted:
        return ExitStatus.INTERRUPTED

    # Reaching the time limit is the expected end of a batch that overran its target, and it's not an error.
    return ExitStatus.SUCCESS


def run_session(args: Arguments, config: Config) -> ExitStatus:
    """Run a factoring session against state that no other instance is using.

    Returns:
        ExitStatus: The exit status for the session.
    """
    interrupts = InterruptState()
    interrupts.install()

    try:
        stats = FactoringStats(config.stats_path)
    except InvalidStatsError as e:
        logger.error("{}", e)
        sys.exit(ExitStatus.INVALID_INPUT)

    tool_failures.load(config.tool_state_path)

    session = Session(config, stats, start_backend(config, stats, interrupts), interrupts)

    logger.info("Using backend: {}", config.backend)
    logger.info("Using factoring mode: {}", config.factoring_mode)
    logger.info("Using up to {} thread{}", config.max_threads, "s" if config.max_threads != 1 else "")

    # The backend is closed last, as flushing its pending submissions can take some time.
    try:
        status = run_batch(args, session)
    finally:
        session.cleanup.run("close the backend", session.backend.close)

    # An interrupt may have arrived while the backend was flushing its submissions.
    if interrupts.interrupted and status in NON_ERROR_EXIT_STATUSES:
        status = ExitStatus.INTERRUPTED

    if session.cleanup.failed and status in NON_ERROR_EXIT_STATUSES:
        status = ExitStatus.CLEANUP_FAILED

    return status


def main() -> None:
    """Factor numbers using various methods."""
    args = Arguments().parse_args()

    if args.version:
        print(f"Factortool version: {__version__}")  # ruff: ignore[print]
        sys.exit(ExitStatus.SUCCESS)

    setup_logger(args.log_level)

    config = load_config(
        args.config_path,
        backend=args.backend,
        factoring_mode=args.factoring_mode,
        log_level=args.log_level,
        max_threads=args.max_threads,
    )

    setup_logger(config.log_level)

    validate_arguments(args, config.backend)
    validate_tools(config)

    with lock_instance(config):
        remove_stale_work_dirs(config.work_path)
        status = run_session(args, config)

    if status != ExitStatus.SUCCESS:
        sys.exit(status)
