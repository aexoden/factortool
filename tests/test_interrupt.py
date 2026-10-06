# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for shared interrupt state and interruptible waits."""

from __future__ import annotations

import signal
import threading
import time

import pytest
import requests

from loguru import logger

from factortool.http import HttpClient
from factortool.interrupt import ABORT, Interrupted, InterruptState

# A wait short enough to keep the suite quick, but long enough to measure.
SHORT_WAIT = 0.5

# Stands in for a long backoff, which must be abandoned rather than slept through.
LONG_WAIT = 30.0

# Generous upper bound on how long an abandoned wait may take.
PROMPT = 5.0

# A backoff short enough that sitting through it is cheap.
TINY_WAIT = 0.3


def test_wait_runs_its_full_duration_when_uninterrupted() -> None:
    """Test that an uninterrupted wait runs for its full duration."""
    interrupts = InterruptState()
    start = time.monotonic()

    assert not interrupts.wait(SHORT_WAIT)
    assert time.monotonic() - start >= SHORT_WAIT


def test_wait_returns_immediately_once_interrupted() -> None:
    """Test that a wait returns immediately once an interrupt has been received."""
    interrupts = InterruptState()
    interrupts._level = 1
    start = time.monotonic()

    assert interrupts.wait(LONG_WAIT)
    assert time.monotonic() - start < PROMPT


def test_wait_wakes_promptly_on_a_real_signal() -> None:
    """Test that a wait wakes promptly when a real SIGINT arrives."""
    interrupts = InterruptState()
    previous = signal.getsignal(signal.SIGINT)
    interrupts.install()

    try:
        timer = threading.Timer(0.2, lambda: signal.raise_signal(signal.SIGINT))
        timer.start()
        start = time.monotonic()
        interrupted = interrupts.wait(LONG_WAIT)
        timer.cancel()
    finally:
        signal.signal(signal.SIGINT, previous)

    assert interrupted
    assert interrupts.level == 1
    assert time.monotonic() - start < PROMPT


def test_a_single_interrupt_stops_fetching_but_not_factoring() -> None:
    """Test that a single interrupt stops fetching but not factoring."""
    interrupts = InterruptState()
    interrupts._level = 1

    assert interrupts.interrupted
    assert interrupts.stop_fetching
    assert not interrupts.stop_factoring


def test_a_second_interrupt_abandons_the_rest_of_the_batch() -> None:
    """Test that a second interrupt abandons the rest of the batch."""
    interrupts = InterruptState()
    interrupts._level = 2

    assert interrupts.stop_fetching
    assert interrupts.stop_factoring


def test_a_third_interrupt_abandons_abortable_work() -> None:
    """Test that a third interrupt raises Interrupted inside an abortable block instead of exiting."""
    interrupts = InterruptState()
    interrupts._level = 2
    previous = signal.getsignal(signal.SIGINT)
    interrupts.install()

    def abort() -> None:
        signal.raise_signal(signal.SIGINT)
        time.sleep(PROMPT)

    try:
        with pytest.raises(Interrupted), interrupts.abortable():
            abort()
    finally:
        signal.signal(signal.SIGINT, previous)

    assert interrupts.level == ABORT


def test_a_third_interrupt_spares_work_that_is_not_abortable() -> None:
    """Test that a third interrupt outside an abortable block hands SIGINT back to Python instead of raising."""
    interrupts = InterruptState()
    interrupts._level = 2
    previous = signal.getsignal(signal.SIGINT)
    interrupts.install()

    try:
        signal.raise_signal(signal.SIGINT)
        time.sleep(SHORT_WAIT)
        handler = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, previous)

    assert interrupts.level == ABORT
    assert handler is signal.default_int_handler


def test_interruptible_backoff_abandons_the_retry() -> None:
    """Test that an interruptible backoff raises Interrupted instead of waiting."""
    interrupts = InterruptState()
    interrupts._level = 1
    client = HttpClient("test", 1.0, "test/1", interrupts)

    with pytest.raises(Interrupted):
        client._backoff(LONG_WAIT, interruptible=True)


def test_backoff_is_not_interruptible_by_default() -> None:
    """Test that a backoff is not interruptible by default."""
    interrupts = InterruptState()
    interrupts._level = 1
    client = HttpClient("test", 1.0, "test/1", interrupts)
    start = time.monotonic()

    client._backoff(TINY_WAIT, interruptible=False)

    assert time.monotonic() - start >= TINY_WAIT


def test_interrupted_is_not_mistaken_for_a_transport_error() -> None:
    """Test that Interrupted is not a requests exception."""
    assert not issubclass(Interrupted, requests.RequestException)


@pytest.mark.parametrize(
    ("level", "abortable", "message"),
    [
        (1, True, "Finishing the current batch"),
        (1, False, "Not fetching any more work"),
        (2, True, "Stopping after the current factorization"),
        (2, False, "Not starting any more factorizations"),
    ],
)
def test_interrupt_messages_describe_what_is_actually_running(level: int, *, abortable: bool, message: str) -> None:
    """Test that an interrupt outside the engine, such as while waiting for work, does not mention a factorization."""
    interrupts = InterruptState()
    interrupts._level = level - 1
    messages: list[str] = []
    sink = logger.add(messages.append)

    try:
        if abortable:
            with interrupts.abortable():
                interrupts._handle_sigint(signal.SIGINT, None)
        else:
            interrupts._handle_sigint(signal.SIGINT, None)
    finally:
        logger.remove(sink)

    assert len(messages) == 1
    assert message in messages[0]
