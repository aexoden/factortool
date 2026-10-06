# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Shared interrupt state."""

from __future__ import annotations

import signal
import time

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import FrameType

from loguru import logger

# How often an interruptible wait checks for an interrupt signal.
POLL_INTERVAL = 0.25

# One interrupt asks for the current batch to be finished and no more work to be fetched.
FINISH_BATCH = 1

# A second interrupt finishes only the current factorization.
STOP_SOON = 2

# A third abandons the current factorization and any remaining work.
ABORT = 3


class Interrupted(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Exception raised when an interrupt signal is received."""


class InterruptState:
    """Tracks received interrupts and provides waits that end early when an interrupt signal is received."""

    def __init__(self) -> None:
        """Initialize the interrupt state."""
        self._level = 0
        self._abortable = False

    @property
    def level(self) -> int:
        """The count of received interrupt signals."""
        return self._level

    @property
    def interrupted(self) -> bool:
        """Whether an interrupt signal has been received."""
        return self._level > 0

    @property
    def stop_fetching(self) -> bool:
        """Whether to stop asking the backend for work."""
        return self._level >= FINISH_BATCH

    @property
    def stop_factoring(self) -> bool:
        """Whether to abandon the numbers left in the batch."""
        return self._level >= STOP_SOON

    def install(self) -> None:
        """Install the interrupt signal handler."""
        signal.signal(signal.SIGINT, self._handle_sigint)

    @contextmanager
    def abortable(self) -> Generator[None]:
        """Let a third interrupt abandon the work running inside this block by raising Interrupted.

        Yields:
            None: Control, for the duration of the abortable work.
        """
        self._abortable = True

        try:
            yield
        finally:
            self._abortable = False

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

        if self._level == FINISH_BATCH:
            if self._abortable:
                logger.critical(
                    "Interrupt received. Finishing the current batch, then stopping. Interrupt again to stop sooner"
                )
            else:
                logger.critical("Interrupt received. Not fetching any more work")
        elif self._level == STOP_SOON:
            if self._abortable:
                logger.critical("Second interrupt received. Stopping after the current factorization")
            else:
                logger.critical("Second interrupt received. Not starting any more factorizations")
        elif self._level == ABORT:
            signal.signal(signal.SIGINT, signal.default_int_handler)

            if not self._abortable:
                logger.critical("Third interrupt received. Finishing the shutdown. Interrupt again to exit immediately")
                return

            logger.critical("Third interrupt received. Abandoning the current factorization")

            raise Interrupted
