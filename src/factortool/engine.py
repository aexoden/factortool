# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import signal
import sys

from collections.abc import Iterable
from types import FrameType

from loguru import logger

from factortool.number import Number


class FactorEngine:
    def __init__(self) -> None:
        self._interrupt_level: int = 0
        signal.signal(signal.SIGINT, self._handle_sigint)

    #
    # Signal Handlers
    #

    def _handle_sigint(self, _signum: int, _frame: FrameType | None) -> None:
        self._interrupt_level += 1

        if self._interrupt_level == 1:
            logger.critical("Interrupt received. Finishing current factorization.")
        else:
            logger.critical("Second interrupt received. Terminating immediately.")
            sys.exit(2)

    #
    # Public Methods
    #

    def run(self, numbers: Iterable[Number]) -> None:
        # Attempt to trial factor each number.
        for number in numbers:
            number.factor_tf()

            if self._interrupt_level > 0:
                return
