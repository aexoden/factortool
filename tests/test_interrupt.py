# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for shared interrupt state and interruptible waits."""

from __future__ import annotations

import signal
import threading
import time

import pytest
import requests

from factortool.http import HttpClient
from factortool.interrupt import Interrupted, InterruptState

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
