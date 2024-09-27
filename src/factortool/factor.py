# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import signal
import sys

from types import FrameType

from loguru import logger


class FactorEngine:
    def __init__(self) -> None:
        self._interrupt_level = 0
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, _signum: int, _frame: FrameType | None) -> None:
        self._interrupt_level += 1

        if self._interrupt_level == 1:
            logger.critical("Interrupt received. Finishing current factorization.")
        else:
            logger.critical("Second interrupt received. Terminating immediately.")
            sys.exit(2)

    def run(self, _numbers: list[int]) -> None:
        count = 0

        while True:
            if self._interrupt_level > 0:
                count += 1

            if count > 100_000_000:
                return
