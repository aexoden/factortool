# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Factorization engine for processing numbers."""

from __future__ import annotations

import concurrent.futures
import time

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

from loguru import logger

from factortool.constants import ECM_CURVES
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

    def __init__(
        self, config: Config, target_duration: float = 600.0, interrupts: InterruptState | None = None
    ) -> None:
        """Initialize the factorization engine."""
        self._config = config
        self._target_duration = target_duration
        self._start_time = time.monotonic()

        if interrupts is None:
            interrupts = InterruptState()
            interrupts.install()

        self._interrupts = interrupts

    def _is_time_limit_exceeded(self) -> bool:
        """Check if the current runtime exceeds twice the target duration.

        Returns:
            bool: True if runtime exceeds 2 * target_duration, False otherwise.
        """
        elapsed = time.monotonic() - self._start_time
        if elapsed > 2.0 * self._target_duration:
            logger.warning(
                "Time limit exceeded: runtime ({:.1f}s) exceeds ({:.1f}s)", elapsed, 2.0 * self._target_duration
            )
            return True

        return False

    def _factor_concurrently(self, numbers: Collection[Number], factor: Callable[[Number], None]) -> None:
        """Apply a factoring method to each unfactored number using a pool of worker threads.

        Raises:
            Interrupted: If the third interrupt arrives, once the work in progress has been abandoned.
        """

        def run(number: Number) -> None:
            if self._interrupts.stop_factoring:
                return

            factor(number)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._config.max_threads) as executor:
            try:
                futures = [executor.submit(run, number) for number in numbers if not number.factored]
                concurrent.futures.wait(futures)
            except Interrupted:
                abandon_tools()
                executor.shutdown(cancel_futures=True)
                raise

    #
    # Public Methods
    #

    def run(self, numbers: Collection[Number]) -> ExitStatus:
        """Run factorization using the configured mode.

        Returns:
            ExitStatus: The exit status of the factorization run.
        """
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
            if self._interrupts.stop_factoring:
                return ExitStatus.INTERRUPTED

            if self._is_time_limit_exceeded():
                return ExitStatus.TIME_LIMIT_EXCEEDED

            logger.info("Factoring {} using YAFU", number.n)
            number.factor_yafu_direct()

        # Check once more for an interrupt that may have occurred during the final factorization.
        if self._interrupts.stop_factoring:
            return ExitStatus.INTERRUPTED

        if self._is_time_limit_exceeded():
            return ExitStatus.TIME_LIMIT_EXCEEDED

        return ExitStatus.SUCCESS

    def _run_standard(self, numbers: Collection[Number]) -> ExitStatus:  # ruff:ignore[complex-structure, too-many-branches, too-many-return-statements]
        """Factor numbers using the built-in sequence of methods.

        Returns:
            ExitStatus: The exit status of the factorization run.
        """
        # Attempt to trial factor each number.
        logger.info("Attempting trial factoring on {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        for number in numbers:
            number.factor_tf()

            if self._interrupts.stop_factoring:
                return ExitStatus.INTERRUPTED

            if self._is_time_limit_exceeded():
                return ExitStatus.TIME_LIMIT_EXCEEDED

        # Attempt to find factors via the Rho method.
        logger.info("Attempting rho factoring on {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        self._factor_concurrently(numbers, Number.factor_rho)

        if self._interrupts.stop_factoring:
            return ExitStatus.INTERRUPTED

        if self._is_time_limit_exceeded():
            return ExitStatus.TIME_LIMIT_EXCEEDED

        # Attempt to find factors via P-1.
        logger.info("Attempting P-1 factoring on {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        self._factor_concurrently(numbers, Number.factor_pm1)

        if self._interrupts.stop_factoring:
            return ExitStatus.INTERRUPTED

        if self._is_time_limit_exceeded():
            return ExitStatus.TIME_LIMIT_EXCEEDED

        # Attempt to factor each number via ECM.
        minimum_ecm_level = min(ECM_CURVES.keys())
        maximum_ecm_level = max(ECM_CURVES.keys())

        for ecm_level in range(minimum_ecm_level, maximum_ecm_level + 1):
            overall_number_count = len([x for x in numbers if not x.factored])
            ecm_numbers = [x for x in numbers if x.ecm_needed]
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
                number.factor_ecm(ecm_level)

                if self._interrupts.stop_factoring:
                    logger.info("Not finishing remaining ECM factorizations due to interrupt")
                    return ExitStatus.INTERRUPTED

                if self._is_time_limit_exceeded():
                    return ExitStatus.TIME_LIMIT_EXCEEDED

        # Do SIQS on the remaining numbers that prefer SIQS.
        overall_number_count = len([x for x in numbers if not x.factored])
        number_count = len([x for x in numbers if not x.factored and x.prefer_siqs])

        if number_count > 0:
            logger.info(
                "Attempting SIQS factoring on {} number{} (of {} total remaining)",
                number_count,
                "s" if number_count != 1 else "",
                overall_number_count,
            )

            for number in [x for x in numbers if not x.factored and x.prefer_siqs]:
                number.factor_siqs()

                if self._interrupts.stop_factoring:
                    logger.info("Not finishing remaining SIQS factorizations due to interrupt")
                    return ExitStatus.INTERRUPTED

                if self._is_time_limit_exceeded():
                    return ExitStatus.TIME_LIMIT_EXCEEDED

        # Finish the remaining numbers with NFS.
        number_count = len([x for x in numbers if not x.factored])

        if number_count == 0:
            return ExitStatus.SUCCESS

        logger.info("Attempting NFS factoring on {} number{}", number_count, "s" if number_count != 1 else "")

        for number in [x for x in numbers if not x.factored]:
            number.factor_nfs()

            if self._interrupts.stop_factoring:
                logger.info("Not finishing remaining NFS factorizations due to interrupt")
                return ExitStatus.INTERRUPTED

            if self._is_time_limit_exceeded():
                return ExitStatus.TIME_LIMIT_EXCEEDED

        return ExitStatus.SUCCESS
