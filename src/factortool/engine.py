# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Factorization engine for processing numbers."""

from __future__ import annotations

import concurrent.futures
import threading
import time

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

from loguru import logger

from factortool.constants import ECM_CURVES, FINAL_METHOD_NAMES
from factortool.interrupt import Interrupted, InterruptState
from factortool.number import Number, abandon_tools

if TYPE_CHECKING:
    from factortool.config import Config


class ExitStatus(Enum):
    """Exit status codes for the factorization engine."""

    SUCCESS = 0
    INTERRUPTED = 1
    TIME_LIMIT_EXCEEDED = 2


class FactorEngine:
    """Engine for managing factorization tasks."""

    def __init__(self, config: Config, interrupts: InterruptState | None = None) -> None:
        """Initialize the factorization engine."""
        self._config = config

        # Both are set at the start of each run.
        self._time_limit: float | None = None
        self._start_time = time.monotonic()

        if interrupts is None:
            interrupts = InterruptState()
            interrupts.install()

        self._interrupts = interrupts

        self._expired: set[Number] = set()
        self._expired_lock = threading.Lock()

    def _skip_expired(self, number: Number) -> bool:
        """Check if a number's assignment is expired, logging the first time it is skipped.

        Returns:
            bool: True if the number's assignment is expired, False otherwise.
        """
        if not number.assignment_expired:
            return False

        with self._expired_lock:
            if number not in self._expired:
                self._expired.add(number)
                logger.warning("Skipping {} as its assignment is about to expire", number.n)

        return True

    def _is_time_limit_exceeded(self) -> bool:
        """Check if the current run has exceeded its time limit.

        Returns:
            bool: True if the run has a time limit and has exceeded it, False otherwise.
        """
        return self._time_limit is not None and time.monotonic() - self._start_time > self._time_limit

    def _stop_status(self, stage: str | None = None) -> ExitStatus | None:
        """Check if the run should end early, either due to an interrupt or the time limit.

        If a stage is named, an interrupt is logged as leaving that stage's remaining factorizations unfinished.

        Returns:
            ExitStatus | None: The status to end the run with, or None if it should continue.
        """
        if self._interrupts.stop_factoring:
            if stage is not None:
                logger.info("Not finishing remaining {} factorizations due to interrupt", stage)

            return ExitStatus.INTERRUPTED

        if self._is_time_limit_exceeded():
            logger.warning(
                "Time limit exceeded: runtime ({:.1f}s) exceeds ({:.1f}s)",
                time.monotonic() - self._start_time,
                self._time_limit,
            )
            return ExitStatus.TIME_LIMIT_EXCEEDED

        return None

    def _factor_concurrently(self, numbers: Collection[Number], factor: Callable[[Number], None]) -> None:
        """Apply a factoring method to each unfactored number using a pool of worker threads.

        A failure in any worker ends the stage. The numbers still queued are not started, the tools already running are
        left to finish, and the failure is then raised here as if it had happened in the calling thread.

        Raises:
            Interrupted: If the third interrupt arrives, once the work in progress has been abandoned.
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
                factor(number)
            except BaseException:
                failed.set()
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._config.max_threads) as executor:
            try:
                done, _ = concurrent.futures.wait(
                    [executor.submit(run, number) for number in numbers if not number.factored],
                    return_when=concurrent.futures.FIRST_EXCEPTION,
                )
                error = next((e for future in done if (e := future.exception()) is not None), None)

                if error is not None:
                    executor.shutdown(cancel_futures=True)
                    raise error
            except Interrupted:
                abandon_tools()
                executor.shutdown(cancel_futures=True)
                raise

    #
    # Public Methods
    #

    def run(self, numbers: Collection[Number], time_limit: float | None = None) -> ExitStatus:
        """Run factorization using the configured mode.

        Returns:
            ExitStatus: The exit status of the factorization run.

        Raises:
            ToolError: If an external tool cannot be started or exits with an error.
        """
        self._time_limit = time_limit
        self._start_time = time.monotonic()

        try:
            with self._interrupts.abortable():
                if self._config.factoring_mode == "yafu":
                    return self._run_yafu(numbers)

                return self._run_standard(numbers)
        except Interrupted:
            logger.warning("Abandoned the factorization in progress")
            return ExitStatus.INTERRUPTED

    def _run_yafu(self, numbers: Collection[Number]) -> ExitStatus:
        """Factor numbers using direct YAFU calls.

        Returns:
            ExitStatus: The exit status of the factorization run.
        """
        logger.info("Using direct YAFU factoring mode for {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        for number in sorted(numbers):
            # Check before each factorization, so a factorization isn't started if an interrupt has been received.
            if (status := self._stop_status()) is not None:
                return status

            if self._skip_expired(number):
                continue

            logger.info("Factoring {} using YAFU", number.n)
            number.factor_yafu_direct()

        # Check once more for an interrupt that may have occurred during the final factorization.
        if (status := self._stop_status()) is not None:
            return status

        return ExitStatus.SUCCESS

    def _run_standard(self, numbers: Collection[Number]) -> ExitStatus:  # ruff: ignore[complex-structure, too-many-branches]
        """Factor numbers using the built-in sequence of methods.

        Returns:
            ExitStatus: The exit status of the factorization run.
        """
        # Attempt to trial factor each number.
        logger.info("Attempting trial factoring on {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        for number in numbers:
            if self._skip_expired(number):
                continue

            number.factor_tf()

            if (status := self._stop_status()) is not None:
                return status

        # Attempt to find factors via the Rho method.
        logger.info("Attempting rho factoring on {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        self._factor_concurrently(numbers, Number.factor_rho)

        if (status := self._stop_status()) is not None:
            return status

        # Attempt to find factors via P-1.
        logger.info("Attempting P-1 factoring on {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        self._factor_concurrently(numbers, Number.factor_pm1)

        if (status := self._stop_status()) is not None:
            return status

        # Attempt to factor each number via ECM.
        minimum_ecm_level = min(ECM_CURVES.keys())
        maximum_ecm_level = max(ECM_CURVES.keys())

        for ecm_level in range(minimum_ecm_level, maximum_ecm_level + 1):
            overall_number_count = len([x for x in numbers if not x.factored])
            ecm_numbers = [x for x in numbers if x.ecm_needed and not self._skip_expired(x)]
            ecm_number_count = len(ecm_numbers)

            if ecm_number_count == 0:
                break

            curves, b1 = ECM_CURVES[ecm_level]

            logger.info(
                "Attempting ECM factoring on {} number{} (of {} total remaining)"
                " at t-level {} with {} curve{} of B1 = {}",
                ecm_number_count,
                "s" if ecm_number_count != 1 else "",
                overall_number_count,
                ecm_level,
                curves,
                "s" if curves != 1 else "",
                b1,
            )

            for number in ecm_numbers:
                if self._skip_expired(number):
                    continue

                number.factor_ecm(ecm_level)

                if (status := self._stop_status("ECM")) is not None:
                    return status

        # Finish the remaining numbers with their preferred final method, grouped by method. We generate the groups
        # first, as the runs could conceivably change the statistics enough for a number's preferred final method to
        # change. If its new preferred method has already run its group, it would be silently dropped.
        final_groups: dict[str, list[Number]] = {method: [] for method in FINAL_METHOD_NAMES}

        for number in numbers:
            if not number.factored and not self._skip_expired(number):
                final_groups[number.final_method].append(number)

        for method, method_name in FINAL_METHOD_NAMES.items():
            final_numbers = final_groups[method]
            number_count = len(final_numbers)
            overall_number_count = len([x for x in numbers if not x.factored])

            if number_count == 0:
                continue

            logger.info(
                "Attempting final factoring on {} number{} (of {} total remaining) (initially grouped under {})",
                number_count,
                "s" if number_count != 1 else "",
                overall_number_count,
                method_name,
            )

            for number in final_numbers:
                if number.factored or self._skip_expired(number):
                    continue

                number.factor_final()

                if (status := self._stop_status("final")) is not None:
                    return status

        return ExitStatus.SUCCESS
