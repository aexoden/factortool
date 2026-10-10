# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Shared interrupt state."""

from __future__ import annotations

import contextlib
import os
import queue
import signal
import socket
import threading
import time

from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
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

# The exit status of a run that is interrupted by signals.
EXIT_STATUS = 2

# How long an immediate exit waits for its message to be logged.
EXIT_GRACE = 0.5

# Signals that ask the program to end, which skip straight to ABORT.
TERMINATION_SIGNALS = frozenset(
    signal.Signals[name] for name in ("SIGBREAK", "SIGHUP", "SIGTERM") if name in signal.Signals.__members__
)


class Interrupted(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Exception raised when an interrupt signal is received."""


class InterruptState:
    """Tracks received signals and provides waits that end early when one is received.

    Interrupts escalate one level at a time, while a termination signal goes straight to the ABORT level.
    """

    def __init__(self) -> None:
        """Initialize the interrupt state."""
        self._level = 0
        self._abandon: Callable[[], None] | None = None
        self._handled: frozenset[int] = frozenset()

        self._announcements: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()

        self._installed = contextlib.ExitStack()

    @property
    def level(self) -> int:
        """How far the received signals have escalated, from zero to ABORT."""
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
        """Start responding to signals. Must be called from the main thread.

        A termination signal that is being ignored is left that way to allow for `nohup` to work correctly.
        """
        self._handled = frozenset(
            {signal.SIGINT} | {signum for signum in TERMINATION_SIGNALS if signal.getsignal(signum) != signal.SIG_IGN}
        )

        # Python writes the number of each signal it receives to one end, and the receiver thread reads the other.
        reader, writer = socket.socketpair()
        writer.settimeout(0)

        receiver = threading.Thread(target=self._receive_all, args=(reader,), name="Signal-Receiver", daemon=True)
        responder = threading.Thread(target=self._respond, name="Signal-Responder", daemon=True)

        self._installed.callback(reader.close)
        self._installed.callback(responder.join)
        self._installed.callback(receiver.join)
        self._installed.callback(writer.close)
        self._installed.callback(signal.set_wakeup_fd, signal.set_wakeup_fd(writer.fileno(), warn_on_full_buffer=False))

        for signum in self._handled:
            # Python only reports a signal that has a handler.
            self._installed.callback(signal.signal, signum, signal.signal(signum, self._ignore))

        receiver.start()
        responder.start()

    def uninstall(self) -> None:
        """Stop responding to signals, undoing the installation. Must be called from the main thread."""
        self._installed.close()

    @contextmanager
    def abortable(self, abandon: Callable[[], None]) -> Generator[None]:
        """Let a third interrupt or a termination signal abandon the work running inside this block.

        The work is abandoned by calling abandon, which should make it raise Interrupted. It is called from another
        thread, and repeatedly for as long as the block is still running.

        Yields:
            None: Control, for the duration of the abortable work.
        """
        self._abandon = abandon

        try:
            yield
        finally:
            self._abandon = None

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

    def _ignore(self, _signum: int, _frame: FrameType | None) -> None:
        """Handle a signal by doing nothing, as the receiver thread is told about it separately."""

    def _receive_all(self, wakeup: socket.socket) -> None:
        """Count each signal as it arrives, until the other end is closed. Nothing here may block."""
        while received := wakeup.recv(1):
            self._receive(received[0])

        self._announcements.put(None)

    def _receive(self, signum: int) -> None:
        """Escalate in response to a signal, leaving logging to the responder thread."""
        # Python reports every signal that has a handler, whoever installed it.
        if signum not in self._handled:
            return

        if self._level >= ABORT:
            # A closing terminal can send SIGHUP more than once.
            if signal.Signals(signum).name != "SIGHUP":
                self._exit_immediately()

            return

        abortable = self._abandon is not None
        self._level = ABORT if signum in TERMINATION_SIGNALS else self._level + 1
        self._announcements.put(partial(self._announce, signum, self._level, abortable=abortable))

    def _exit_immediately(self) -> None:
        """End the program without any cleanup, whatever its other threads are doing."""
        announced = threading.Event()

        def announce() -> None:
            logger.critical("Another signal received. Exiting immediately")
            announced.set()

        # The responder thread itself may be stuck.
        self._announcements.put(announce)
        announced.wait(EXIT_GRACE)

        os._exit(EXIT_STATUS)

    def _respond(self) -> None:
        """Log what each signal is going to do, and abandon the abortable work when requested."""
        while True:
            timeout = POLL_INTERVAL if self._level >= ABORT else None

            with contextlib.suppress(queue.Empty):
                if (announce := self._announcements.get(timeout=timeout)) is None:
                    return

                announce()

            if self._level >= ABORT and (abandon := self._abandon) is not None:
                try:
                    abandon()
                except OSError as e:
                    logger.error("Failed to abandon the current factorization: {}", e)

    @staticmethod
    def _announce(signum: int, level: int, *, abortable: bool) -> None:
        """Log what a received signal is going to do."""
        if signum in TERMINATION_SIGNALS:
            name = signal.Signals(signum).name

            if abortable:
                logger.critical("{} received. Abandoning the current factorization", name)
            else:
                logger.critical("{} received. Shutting down. Signal again to exit immediately", name)
        elif level == FINISH_BATCH:
            if abortable:
                logger.critical(
                    "Interrupt received. Finishing the current batch, then stopping. Interrupt again to stop sooner"
                )
            else:
                logger.critical("Interrupt received. Not fetching any more work")
        elif level == STOP_SOON:
            if abortable:
                logger.critical("Second interrupt received. Stopping after the current factorization")
            else:
                logger.critical("Second interrupt received. Not starting any more factorizations")
        elif abortable:
            logger.critical("Third interrupt received. Abandoning the current factorization")
        else:
            logger.critical("Third interrupt received. Finishing the shutdown. Interrupt again to exit immediately")
