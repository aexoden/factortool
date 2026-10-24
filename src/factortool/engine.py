# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024-2026 Jason Lynch <jason@aexoden.com>
"""Factorization engine for processing numbers."""

from __future__ import annotations

import concurrent.futures
import threading
import time

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

from loguru import logger

from factortool.constants import ECM_CURVES, ECM_MAX_LEVEL, ECM_MIN_LEVEL, FINAL_METHOD_NAMES
from factortool.interrupt import Interrupted, InterruptState
from factortool.number import Number
from factortool.tools import ToolDeadlineExceeded, abandon_tools, tool_deadline
from factortool.util import format_count, format_number

if TYPE_CHECKING:
    from factortool.config import Config


class EngineStatus(Enum):
    """Status codes for the factorization engine."""

    SUCCESS = 0
    INTERRUPTED = 1
    TIME_LIMIT_EXCEEDED = 2


@dataclass(frozen=True)
class TimeLimits:
    """How long a run may take, in seconds from its start. None means no limit."""

    # Once this has passed, no further factorizations are started.
    soft: float | None = None

    # Once this has passed, the factorizations still running are abandoned.
    hard: float | None = None

    @property
    def earliest(self) -> float | None:
        """The limit that ends the run, as nothing is started once either limit has passed."""
        return min((limit for limit in (self.soft, self.hard) if limit is not None), default=None)


class FactorEngine:
    """Engine for managing factorization tasks."""

    def __init__(self, config: Config, interrupts: InterruptState | None = None) -> None:
        """Initialize the factorization engine."""
        self._config = config

        self._limits = TimeLimits()
        self._start_time = time.monotonic()
        self._expired: set[Number] = set()
        self._expired_lock = threading.Lock()

        if interrupts is None:
            interrupts = InterruptState()
            interrupts.install()

        self._interrupts = interrupts

    def _skip_expired(self, number: Number) -> bool:
        """Check if a number's assignment is expired, logging the first time it is skipped.

        Returns:
            bool: True if the number's assignment is expired, False otherwise.
        """
        with self._expired_lock:
            if number in self._expired:
                return True

            if not number.assignment_expired:
                return False

            self._expired.add(number)
            logger.warning("Skipping {} as its assignment has expired", format_number(number.n))

        return True

    def _is_time_limit_exceeded(self) -> bool:
        """Check if the current run has exceeded its time limit.

        Returns:
            bool: True if the run has a time limit and has reached it, False otherwise.
        """
        time_limit = self._limits.earliest

        return time_limit is not None and time.monotonic() - self._start_time > time_limit

    def _is_hard_time_limit_exceeded(self) -> bool:
        """Check if the current run has exceeded its hard time limit.

        Returns:
            bool: True if the run has a hard time limit and has exceeded it, False otherwise.
        """
        return self._limits.hard is not None and time.monotonic() - self._start_time >= self._limits.hard

    def _tool_deadline(self, number: Number) -> float | None:
        """Compute the earliest deadline for a number, considering both the hard time limit and the number's expiration.

        Returns:
            float | None: The earliest deadline as a monotonic timestamp, or None if no deadlines are set.
        """
        deadlines: list[float] = []

        if self._limits.hard is not None:
            deadlines.append(self._start_time + self._limits.hard)

        if number.expires_at is not None:
            deadlines.append(time.monotonic() + number.expires_at - time.time())

        return min(deadlines, default=None)

    def _attempt(self, number: Number, factor: Callable[[Number], None]) -> None:
        """Apply a factoring method to a number, abandoning it if a tool is still running at its deadline."""
        try:
            with tool_deadline(self._tool_deadline(number)):
                factor(number)
        except ToolDeadlineExceeded as e:
            if self._is_hard_time_limit_exceeded():
                logger.warning("Abandoned {}, as the hard time limit was reached", e)
                return

            with self._expired_lock:
                self._expired.add(number)

            logger.warning("Abandoned {}, as the assignment for {} has expired", e, format_number(number.n))

    def _stop_status(self, stage: str | None = None) -> EngineStatus | None:
        """Check if the run should end early, either due to an interrupt or the time limit.

        If a stage is named, an interrupt is logged as leaving that stage's remaining factorizations unfinished.

        Returns:
            EngineStatus | None: The status to end the run with, or None if it should continue.
        """
        if self._interrupts.stop_factoring:
            if stage is not None:
                logger.info("Not finishing remaining {} factorizations due to interrupt", stage)

            return EngineStatus.INTERRUPTED

        if self._is_time_limit_exceeded():
            logger.warning(
                "Time limit exceeded: runtime ({:.1f}s) exceeds ({:.1f}s)",
                time.monotonic() - self._start_time,
                self._limits.earliest,
            )
            return EngineStatus.TIME_LIMIT_EXCEEDED

        return None

    def _factor_concurrently(self, numbers: Collection[Number], factor: Callable[[Number], None]) -> None:
        """Apply a factoring method to each unfactored number using a pool of worker threads.

        An exception in any worker ends the stage. The numbers still queued are not started, the tools already running
        are left to finish, and the exception is then raised here as if it had happened in the calling thread. If the
        work is abandoned meanwhile, a tool failure is still raised in preference to Interrupted.

        Raises:
            Interrupted: If a worker's tool is killed because the work is being abandoned, and no other worker failed.
        """
        failed = threading.Event()

        def run(number: Number) -> None:
            if (
                failed.is_set()
                or self._interrupts.stop_factoring
                or self._is_time_limit_exceeded()
                or self._skip_expired(number)
            ):
                return

            try:
                self._attempt(number, factor)
            except BaseException:
                failed.set()
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._config.max_threads) as executor:
            futures = [executor.submit(run, number) for number in numbers if number.active]
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)

            if any(future.exception() is not None for future in done):
                executor.shutdown(cancel_futures=True)

                # Every worker has now finished, so collect all their errors. A tool failure is preferred over the
                # Interrupted of a tool killed by an abort, however the two happened to be ordered.
                errors = [e for future in futures if not future.cancelled() and (e := future.exception()) is not None]
                error = next((e for e in errors if not isinstance(e, Interrupted)), errors[0])
                raise error

    #
    # Public Methods
    #

    def run(self, numbers: Collection[Number], limits: TimeLimits | None = None) -> EngineStatus:
        """Run factorization using the configured mode.

        Returns:
            EngineStatus: The ending status of the factorization run.

        A number that an external tool fails on is left unfactored, and the run continues with the rest.

        Raises:
            ToolError: If an external tool fails several times in a row.
        """
        self._limits = limits if limits is not None else TimeLimits()
        self._start_time = time.monotonic()

        # A number may come back with a new assignment after an earlier one expired.
        with self._expired_lock:
            self._expired.clear()

        try:
            with self._interrupts.abortable(abandon_tools):
                if self._config.factoring_mode == "yafu":
                    status = self._run_yafu(numbers)
                else:
                    status = self._run_standard(numbers)
        except Interrupted:
            logger.warning("Abandoned the factorization in progress")
            status = EngineStatus.INTERRUPTED

        return status

    def _run_yafu(self, numbers: Collection[Number]) -> EngineStatus:
        """Factor numbers using direct YAFU calls.

        Returns:
            EngineStatus: The ending status of the factorization run.
        """
        logger.info("Using direct YAFU factoring mode for {}", format_count(len(numbers), "number"))

        for number in sorted(numbers):
            # Check before each factorization, so a factorization isn't started if an interrupt has been received.
            if (status := self._stop_status()) is not None:
                return status

            if self._skip_expired(number):
                continue

            logger.info("Factoring {} using YAFU", format_number(number.n))
            self._attempt(number, Number.factor_yafu_direct)

        # Check once more for an interrupt that may have occurred during the final factorization.
        if (status := self._stop_status()) is not None:
            return status

        return EngineStatus.SUCCESS

    def _run_standard(self, numbers: Collection[Number]) -> EngineStatus:  # ruff: ignore[complex-structure, too-many-branches]
        """Factor numbers using the built-in sequence of methods.

        Returns:
            EngineStatus: The ending status of the factorization run.
        """
        # Attempt to trial factor each number.
        logger.info("Attempting trial factoring on {}", format_count(len(numbers), "number"))

        for number in numbers:
            if self._skip_expired(number):
                continue

            self._attempt(number, Number.factor_tf)

            if (status := self._stop_status()) is not None:
                return status

        # Attempt to find factors via the Rho method.
        logger.info("Attempting rho factoring on {}", format_count(len(numbers), "number"))

        self._factor_concurrently(numbers, Number.factor_rho)

        if (status := self._stop_status()) is not None:
            return status

        # Attempt to find factors via P-1.
        logger.info("Attempting P-1 factoring on {}", format_count(len(numbers), "number"))

        self._factor_concurrently(numbers, Number.factor_pm1)

        if (status := self._stop_status()) is not None:
            return status

        # Attempt to factor each number via ECM.
        for ecm_level in range(ECM_MIN_LEVEL, ECM_MAX_LEVEL + 1):
            overall_number_count = len([x for x in numbers if x.active])
            ecm_numbers = [x for x in numbers if x.ecm_needed and not self._skip_expired(x)]
            ecm_number_count = len(ecm_numbers)

            if ecm_number_count == 0:
                break

            curves, b1 = ECM_CURVES[ecm_level]

            logger.info(
                "Attempting ECM factoring on {} (of {} total remaining) at t-level {} with {} of B1 = {}",
                format_count(ecm_number_count, "number"),
                overall_number_count,
                ecm_level,
                format_count(curves, "curve"),
                b1,
            )

            for number in ecm_numbers:
                if self._skip_expired(number):
                    continue

                self._attempt(number, partial(Number.factor_ecm, level=ecm_level))

                if (status := self._stop_status("ECM")) is not None:
                    return status

        # Finish the remaining numbers with their preferred final method, grouped by method. We generate the groups
        # first, as the runs could conceivably change the statistics enough for a number's preferred final method to
        # change. If its new preferred method has already run its group, it would be silently dropped.
        final_groups: dict[str, list[Number]] = {method: [] for method in FINAL_METHOD_NAMES}

        for number in numbers:
            if number.active and not self._skip_expired(number):
                final_groups[number.final_method].append(number)

        for method, method_name in FINAL_METHOD_NAMES.items():
            final_numbers = final_groups[method]
            number_count = len(final_numbers)
            overall_number_count = len([x for x in numbers if x.active])

            if number_count == 0:
                continue

            logger.info(
                "Attempting final factoring on {} (of {} total remaining) (initially grouped under {})",
                format_count(number_count, "number"),
                overall_number_count,
                method_name,
            )

            for number in final_numbers:
                if not number.active or self._skip_expired(number):
                    continue

                self._attempt(number, Number.factor_final)

                if (status := self._stop_status("final")) is not None:
                    return status

        return EngineStatus.SUCCESS
