# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for shared interrupt state and interruptible waits."""

from __future__ import annotations

import signal
import socket
import sys
import threading
import time

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import requests

from loguru import logger

from factortool.http import HttpClient
from factortool.interrupt import (
    ABORT,
    EXIT_GRACE,
    EXIT_STATUS,
    POLL_INTERVAL,
    STOP_SOON,
    TERMINATION_SIGNALS,
    Interrupted,
    InterruptState,
)

from .helpers import installed_interrupts

# A wait short enough to keep the suite quick, but long enough to measure.
SHORT_WAIT = 0.5

# Stands in for a long backoff, which must be abandoned rather than slept through.
LONG_WAIT = 30.0

# Generous upper bound on how long an abandoned wait may take.
PROMPT = 5.0

# A backoff short enough that sitting through it is cheap.
TINY_WAIT = 0.3

# Each way of asking for an abort, as the interrupts already received and the signal that then arrives.
ABORTS = [(STOP_SOON, signal.SIGINT), *((0, signum) for signum in sorted(TERMINATION_SIGNALS))]


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
    with installed_interrupts() as interrupts:
        timer = threading.Timer(0.2, lambda: signal.raise_signal(signal.SIGINT))
        timer.start()
        start = time.monotonic()
        interrupted = interrupts.wait(LONG_WAIT)
        timer.cancel()

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


def void(_value: object) -> None:
    """Discard a value, for a callback that has to return nothing."""


def raise_at_level(interrupts: InterruptState, level: int, signum: int) -> None:
    """Raise a real signal with the given number of interrupts already received, giving its handler time to run."""
    interrupts._level = level
    signal.raise_signal(signum)
    time.sleep(TINY_WAIT)


@pytest.mark.parametrize(("level", "signum"), ABORTS)
def test_an_abort_abandons_abortable_work_from_another_thread(level: int, signum: int) -> None:
    """Test that an abort calls the block's abandon function off the main thread instead of raising into the work."""
    callers: list[threading.Thread] = []

    with installed_interrupts() as interrupts, interrupts.abortable(lambda: callers.append(threading.current_thread())):
        raise_at_level(interrupts, level, signum)
        time.sleep(2 * POLL_INTERVAL)

    assert interrupts.level == ABORT
    assert len(callers) > 1
    assert threading.main_thread() not in callers


def test_an_abort_keeps_abandoning_work_that_starts_after_it() -> None:
    """Test that abortable work started once an abort has already been received is abandoned as well."""
    abandoned = threading.Event()

    with installed_interrupts() as interrupts:
        raise_at_level(interrupts, 0, signal.SIGTERM)

        with interrupts.abortable(abandoned.set):
            assert abandoned.wait(PROMPT)


def test_work_is_no_longer_abandoned_once_it_has_finished() -> None:
    """Test that the abandon function is left alone once its block has exited."""
    calls: list[float] = []

    with installed_interrupts() as interrupts:
        with interrupts.abortable(lambda: calls.append(time.monotonic())):
            raise_at_level(interrupts, 0, signal.SIGTERM)

        finished = time.monotonic()
        time.sleep(3 * POLL_INTERVAL)

    # A call already under way as the block exits may still land, but nothing after that.
    assert calls
    assert all(call < finished + POLL_INTERVAL for call in calls)


@pytest.mark.parametrize(("level", "signum"), ABORTS)
def test_another_signal_after_an_abort_exits_immediately(
    monkeypatch: pytest.MonkeyPatch, level: int, signum: int
) -> None:
    """Test that a signal arriving once everything has already been asked for ends the program without cleanup."""
    hard_exit = Mock()
    monkeypatch.setattr("factortool.interrupt.os._exit", hard_exit, raising=True)

    with installed_interrupts() as interrupts:
        raise_at_level(interrupts, level, signum)
        hard_exit.assert_not_called()
        raise_at_level(interrupts, ABORT, signal.SIGINT)

    hard_exit.assert_called_once_with(EXIT_STATUS)


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP does not exist on this platform")
def test_a_repeated_hangup_does_not_exit_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the second SIGHUP a closing terminal can send is not taken as a demand to exit without cleanup."""
    # Work around mypy not skipping parsing this test on Windows.
    if sys.platform != "win32":
        hard_exit = Mock()
        monkeypatch.setattr("factortool.interrupt.os._exit", hard_exit, raising=True)

        with installed_interrupts() as interrupts:
            raise_at_level(interrupts, 0, signal.SIGHUP)
            raise_at_level(interrupts, ABORT, signal.SIGHUP)

        assert interrupts.level == ABORT
        hard_exit.assert_not_called()


def test_a_failure_to_abandon_work_does_not_stop_signals_being_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the responder survives a failing abandon function, so that a later signal still gets a response."""
    hard_exit = Mock()
    monkeypatch.setattr("factortool.interrupt.os._exit", hard_exit, raising=True)

    with installed_interrupts() as interrupts, interrupts.abortable(Mock(side_effect=PermissionError)):
        raise_at_level(interrupts, 0, signal.SIGTERM)
        raise_at_level(interrupts, ABORT, signal.SIGINT)

    hard_exit.assert_called_once_with(EXIT_STATUS)


def test_an_immediate_exit_does_not_wait_for_work_that_is_slow_to_abandon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a signal after an abort still exits while the abandon function is blocked, as taskkill can be."""
    hard_exit = Mock()
    release = threading.Event()
    monkeypatch.setattr("factortool.interrupt.os._exit", hard_exit, raising=True)

    try:
        with installed_interrupts() as interrupts, interrupts.abortable(lambda: void(release.wait())):
            raise_at_level(interrupts, 0, signal.SIGTERM)
            signal.raise_signal(signal.SIGINT)
            time.sleep(EXIT_GRACE + TINY_WAIT)
            hard_exit.assert_called_once_with(EXIT_STATUS)
            release.set()
    finally:
        release.set()


def test_an_immediate_exit_does_not_wait_for_a_log_that_cannot_be_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a signal after an abort still exits while logging is blocked, as it is by a full pipe."""
    hard_exit = Mock()
    release = threading.Event()
    monkeypatch.setattr("factortool.interrupt.os._exit", hard_exit, raising=True)
    sink = logger.add(lambda _message: void(release.wait()))

    try:
        with installed_interrupts() as interrupts:
            raise_at_level(interrupts, 0, signal.SIGTERM)
            assert interrupts.level == ABORT
            signal.raise_signal(signal.SIGINT)
            time.sleep(EXIT_GRACE + TINY_WAIT)
            hard_exit.assert_called_once_with(EXIT_STATUS)
            release.set()
    finally:
        release.set()
        logger.remove(sink)


def test_uninstalling_releases_everything_install_set_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that uninstalling stops both threads, closes both sockets and puts the previous signal handling back."""
    signals = (signal.SIGINT, *TERMINATION_SIGNALS)
    handlers = {signum: signal.getsignal(signum) for signum in signals}
    threads = threading.active_count()
    real_socketpair = socket.socketpair
    previous, other = real_socketpair()
    previous.setblocking(False)  # ruff: ignore[boolean-positional-value-in-call]
    created: list[socket.socket] = []

    def socketpair() -> tuple[socket.socket, socket.socket]:
        created.extend(pair := real_socketpair())
        return pair

    monkeypatch.setattr("factortool.interrupt.socket.socketpair", socketpair, raising=True)
    signal.set_wakeup_fd(previous.fileno())

    try:
        with installed_interrupts() as interrupts:
            assert threading.active_count() == threads + 2
            raise_at_level(interrupts, 0, signal.SIGTERM)

        restored = signal.set_wakeup_fd(-1)
    finally:
        signal.set_wakeup_fd(-1)
        expected = previous.fileno()
        previous.close()
        other.close()

    assert restored == expected
    assert [created_socket.fileno() for created_socket in created] == [-1, -1]
    assert threading.active_count() == threads
    assert {signum: signal.getsignal(signum) for signum in signals} == handlers
    assert interrupts.level == ABORT


def test_a_signal_that_is_not_handled_is_not_counted() -> None:
    """Test that a signal reported on behalf of a handler installed elsewhere doesn't escalate anything."""
    interrupts = InterruptState()

    interrupts._receive(signal.SIGINT)

    assert interrupts.level == 0


def test_signals_are_handled_without_interrupting_the_main_thread() -> None:
    """Test that the main thread's handler does nothing, leaving the response to the responder thread."""
    messages: list[str] = []
    sink = logger.add(messages.append)

    try:
        with installed_interrupts() as interrupts:
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            handler(signal.SIGINT, None)
    finally:
        logger.remove(sink)

    assert interrupts.level == 0
    assert not messages


@pytest.mark.parametrize("signum", sorted(TERMINATION_SIGNALS))
def test_a_termination_signal_skips_straight_to_an_abort(signum: int) -> None:
    """Test that a single termination signal stops everything, and ends a wait just as an interrupt does."""
    with installed_interrupts() as interrupts:
        timer = threading.Timer(0.2, lambda: signal.raise_signal(signum))
        timer.start()
        start = time.monotonic()
        interrupted = interrupts.wait(LONG_WAIT)
        timer.cancel()

    assert interrupted
    assert interrupts.level == ABORT
    assert interrupts.stop_factoring
    assert time.monotonic() - start < PROMPT


def test_an_ignored_termination_signal_stays_ignored() -> None:
    """Test that a termination signal ignored at startup, as SIGHUP is under nohup, is not handled."""
    previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)

    try:
        with installed_interrupts() as interrupts:
            handler = signal.getsignal(signal.SIGTERM)
            signal.raise_signal(signal.SIGTERM)
            time.sleep(TINY_WAIT)
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert handler == signal.SIG_IGN
    assert interrupts.level == 0


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
    ("level", "signum", "abortable", "message"),
    [
        (1, signal.SIGINT, True, "Finishing the current batch"),
        (1, signal.SIGINT, False, "Not fetching any more work"),
        (2, signal.SIGINT, True, "Stopping after the current factorization"),
        (2, signal.SIGINT, False, "Not starting any more factorizations"),
        (3, signal.SIGINT, True, "Third interrupt received. Abandoning the current factorization"),
        (3, signal.SIGINT, False, "Finishing the shutdown"),
        (3, signal.SIGTERM, True, "SIGTERM received. Abandoning the current factorization"),
        (3, signal.SIGTERM, False, "SIGTERM received. Shutting down"),
    ],
)
def test_signal_messages_describe_what_is_actually_running(
    level: int, signum: int, *, abortable: bool, message: str
) -> None:
    """Test that a signal outside the engine, such as while waiting for work, does not mention a factorization."""
    messages: list[str] = []
    sink = logger.add(messages.append)

    try:
        with installed_interrupts() as interrupts, interrupts.abortable(Mock()) if abortable else nullcontext():
            raise_at_level(interrupts, level - 1 if signum == signal.SIGINT else 0, signum)
    finally:
        logger.remove(sink)

    assert interrupts.level == level
    assert len(messages) == 1
    assert message in messages[0]
