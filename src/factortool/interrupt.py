# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Shared interrupt state."""

from __future__ import annotations

import signal
import sys
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType

from loguru import logger

# How often an interruptible wait checks for an interrupt signal.
POLL_INTERVAL = 0.25


class Interrupted(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Exception raised when an interrupt signal is received."""


class InterruptState:
    """Tracks received interrupts and provides waits that end early when an interrupt signal is received."""

    def __init__(self) -> None:
        """Initialize the interrupt state."""
        self._level = 0

    @property
    def level(self) -> int:
        """The count of received interrupt signals."""
        return self._level

    @property
    def interrupted(self) -> bool:
        """Whether an interrupt signal has been received."""
        return self._level > 0

    def install(self) -> None:
        """Install the interrupt signal handler."""
        signal.signal(signal.SIGINT, self._handle_sigint)

    def wait(self, timeout: float) -> bool:
        """Sleep for up to timeout seconds, returning early if an interrupt arrives.

        Returns:
            bool: True if the wait ended early due to an interrupt signal. False otherwise.
        """
        deadline = time.monotonic() + timeout

        while not self.interrupted:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                return False

            time.sleep(min(remaining, POLL_INTERVAL))

        return True

    def _handle_sigint(self, _signum: int, _frame: FrameType | None) -> None:
        self._level += 1

        if self._level == 1:
            logger.critical("Interrupt received. Finishing current factorization")
        else:
            logger.critical("Second interrupt received. Terminating immediately")
            sys.exit(2)
