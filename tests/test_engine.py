# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the factorization engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from unittest.mock import Mock

import pytest

from factortool.engine import ExitStatus, FactorEngine
from factortool.interrupt import FINISH_BATCH, STOP_SOON, Interrupted, InterruptState

from .helpers import make_config, make_number

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
