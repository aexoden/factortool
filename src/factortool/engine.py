# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import concurrent.futures
import signal
import sys

from collections.abc import Collection
from types import FrameType

from loguru import logger

from factortool.config import Config
from factortool.constants import ECM_CURVES
from factortool.number import Number


class FactorEngine:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._interrupt_level: int = 0
        signal.signal(signal.SIGINT, self._handle_sigint)

    #
    # Signal Handlers
    #

    def _handle_sigint(self, _signum: int, _frame: FrameType | None) -> None:
        self._interrupt_level += 1

        if self._interrupt_level == 1:
            logger.critical("Interrupt received. Finishing current factorization")
        else:
            logger.critical("Second interrupt received. Terminating immediately")
            sys.exit(2)

    #
    # Public Methods
    #

    def run(self, numbers: Collection[Number]) -> bool:
        """Run factorization using the configured mode."""
        if self._config.factoring_mode == "yafu":
            return self._run_yafu(numbers)

        return self._run_standard(numbers)

    def _run_yafu(self, numbers: Collection[Number]) -> bool:
        """Factor numbers using direct YAFU calls."""
        logger.info("Using direct YAFU factoring mode for {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        for number in sorted(numbers):
            if self._interrupt_level > 0:
                return True

            logger.info("Factoring {} using YAFU", number.n)
            number.factor_yafu_direct()

        return False

    def _run_standard(self, numbers: Collection[Number]) -> bool:  # noqa: PLR0911, PLR0912
        # Attempt to trial factor each number.
        logger.info("Attempting trial factoring on {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        for number in numbers:
            number.factor_tf()

            if self._interrupt_level > 0:
                return True

        # Attempt to find factors via the Rho method.
        logger.info("Attempting rho factoring on {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        def factor_rho(number: Number) -> None:
            number.factor_rho()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._config.max_threads) as executor:
            for number in [x for x in numbers if not x.factored]:
                executor.submit(factor_rho, number)

        if self._interrupt_level > 0:
            return True

        # Attempt to find factors via P-1.
        logger.info("Attempting P-1 factoring on {} number{}", len(numbers), "s" if len(numbers) != 1 else "")

        def factor_pm1(number: Number) -> None:
            number.factor_pm1()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._config.max_threads) as executor:
            for number in [x for x in numbers if not x.factored]:
                executor.submit(factor_pm1, number)

        if self._interrupt_level > 0:
            return True

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
                "Attempting ECM factoring on {} number{} (of {} total remaining) at t-level {} with {} curve{} of B1 = {}",  # noqa: E501
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

                if self._interrupt_level > 0:
                    logger.info("Not finishing remaining ECM factorizations due to interrupt")
                    return True

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

                if self._interrupt_level > 0:
                    logger.info("Not finishing remaining SIQS factorizations due to interrupt")
                    return True

        # Finish the remaining numbers with NFS.
        number_count = len([x for x in numbers if not x.factored])

        if number_count == 0:
            return False

        logger.info("Attempting NFS factoring on {} number{}", number_count, "s" if number_count != 1 else "")

        for number in [x for x in numbers if not x.factored]:
            number.factor_nfs()

            if self._interrupt_level > 0:
                logger.info("Not finishing remaining NFS factorizations due to interrupt")
                return True

        return False
