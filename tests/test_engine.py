# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the factorization engine."""

from __future__ import annotations

import threading
import time

from typing import TYPE_CHECKING, Literal
from unittest.mock import Mock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from factortool.engine import ExitStatus, FactorEngine
from factortool.interrupt import FINISH_BATCH, STOP_SOON, Interrupted, InterruptState
from factortool.number import run_tool

from .helpers import (
    TOOL_STOP_TIMEOUT,
    heartbeat_stopped,
    make_config,
    make_number,
    make_tool_with_helper,
    wait_for_heartbeat,
)

if TYPE_CHECKING:
    from factortool.number import Number

# Composite, so Number does not treat them as already factored.
COMPOSITES = [100, 102, 104]


def run_interrupted_on(
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["standard", "yafu"],
    count: int,
    interrupt_after: int,
    level: int = STOP_SOON,
) -> tuple[ExitStatus, int]:
    """Run the engine, raising an interrupt after a given number of factorizations.

    Returns:
        tuple[ExitStatus, int]: The status the engine reported and the number of completed factorizations.
    """
    config = make_config().model_copy(update={"factoring_mode": mode})
    interrupts = InterruptState()
    engine = FactorEngine(config, 600.0, interrupts)
    numbers = [make_number(n) for n in COMPOSITES[:count]]
    completed = 0

    def factor(self: Number) -> None:
        nonlocal completed
        completed += 1
        self.methods.append(mode)

        # The interrupt lands while this factorization is running; YAFU is in its own process group, so the run
        # always completes rather than being killed.
        if completed == interrupt_after:
            interrupts._level = level

    def factor_ecm(self: Number, level: int) -> None:
        factor(self)
        self._ecm_level = level

    for method in ("factor_yafu_direct", "factor_tf", "factor_rho", "factor_pm1", "factor_siqs", "factor_nfs"):
        monkeypatch.setattr(f"factortool.number.Number.{method}", factor, raising=True)

    monkeypatch.setattr("factortool.number.Number.factor_ecm", factor_ecm, raising=True)

    return engine.run(numbers), completed


@pytest.mark.parametrize("mode", ["yafu", "standard"])
@pytest.mark.parametrize(("count", "interrupt_after"), [(3, 1), (3, 3), (1, 1)])
def test_interrupt_is_reported_regardless_of_when_it_arrives(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"], count: int, interrupt_after: int
) -> None:
    """Test that an interrupt is reported even when it arrives during the final number."""
    status, _ = run_interrupted_on(monkeypatch, mode, count, interrupt_after)

    assert status == ExitStatus.INTERRUPTED


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_uninterrupted_run_reports_success(monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]) -> None:
    """Test that an uninterrupted run reports success."""
    status, _ = run_interrupted_on(monkeypatch, mode, 3, 0)

    assert status == ExitStatus.SUCCESS


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_a_single_interrupt_finishes_the_batch(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]
) -> None:
    """Test that a single interrupt lets the batch in hand run to completion."""
    _, expected = run_interrupted_on(monkeypatch, mode, 3, 0)
    status, completed = run_interrupted_on(monkeypatch, mode, 3, 1, level=FINISH_BATCH)

    assert status == ExitStatus.SUCCESS
    assert completed == expected


def test_interrupt_before_the_run_starts_no_yafu_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a second interrupt arriving before the run, such as during the fetch, starts no YAFU factorization."""
    interrupts = InterruptState()
    engine = FactorEngine(make_config().model_copy(update={"factoring_mode": "yafu"}), 600.0, interrupts)
    factor = Mock()
    monkeypatch.setattr("factortool.number.Number.factor_yafu_direct", factor, raising=True)
    interrupts._level = STOP_SOON

    assert engine.run([make_number(n) for n in COMPOSITES]) == ExitStatus.INTERRUPTED
    factor.assert_not_called()


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_abandoned_factorization_leaves_the_number_unfactored(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]
) -> None:
    """Test that a factorization abandoned by the third interrupt keeps its composite, so shutdown carries it over."""
    engine = FactorEngine(make_config().model_copy(update={"factoring_mode": mode}), 600.0, InterruptState())
    number = make_number(COMPOSITES[0])

    def abandon(*_args: object) -> list[int]:
        raise Interrupted

    for function in ("factor_tf", "factor_yafu_direct"):
        monkeypatch.setattr(f"factortool.number.{function}", abandon, raising=True)

    assert engine.run([number]) == ExitStatus.INTERRUPTED
    assert number.composite_factors == [COMPOSITES[0]]
    assert not number.factored


def test_third_interrupt_abandons_tools_running_in_worker_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that abandoning the rho stage kills the workers' tools and waits for the workers before returning."""
    engine = FactorEngine(make_config(), 600.0, InterruptState())
    number = make_number(COMPOSITES[0])
    heartbeat = tmp_path / "heartbeat"
    finished = threading.Event()

    def factor_yafu(n: int, *_args: object) -> list[int]:
        try:
            run_tool(make_tool_with_helper(heartbeat), tmp_path)
        finally:
            finished.set()

        return [n]

    def interrupt_once_started(*_args: object, **_kwargs: object) -> None:
        # Stands in for the third interrupt.
        wait_for_heartbeat(heartbeat)
        raise Interrupted

    monkeypatch.setattr("factortool.number.factor_tf", lambda n, _stats: [n], raising=True)
    monkeypatch.setattr("factortool.number.factor_yafu", factor_yafu, raising=True)
    monkeypatch.setattr("factortool.engine.concurrent.futures.wait", interrupt_once_started, raising=True)

    start = time.monotonic()

    assert engine.run([number]) == ExitStatus.INTERRUPTED
    assert time.monotonic() - start < TOOL_STOP_TIMEOUT
    assert finished.is_set()
    assert heartbeat_stopped(heartbeat)
    assert number.composite_factors == [COMPOSITES[0]]


@pytest.mark.parametrize("mode", ["yafu", "standard"])
def test_numbers_whose_assignment_has_expired_are_skipped(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["standard", "yafu"]
) -> None:
    """Test that no work is started on a number whose assignment has expired, while other numbers are unaffected."""
    config = make_config().model_copy(update={"factoring_mode": mode})
    engine = FactorEngine(config, 600.0, InterruptState())
    lapsed, assigned, unassigned = (make_number(n) for n in COMPOSITES)
    lapsed.expires_at = time.time()
    assigned.expires_at = time.time() + 3600.0
    attempted: set[int] = set()

    def factor(self: Number, *_: object) -> None:
        attempted.add(self.n)
        self.prime_factors = [self.n]
        self.composite_factors = []

    for method in ("factor_yafu_direct", "factor_tf"):
        monkeypatch.setattr(f"factortool.number.Number.{method}", factor, raising=True)

    assert engine.run([lapsed, assigned, unassigned]) == ExitStatus.SUCCESS
    assert attempted == {assigned.n, unassigned.n}
    assert not lapsed.factored
