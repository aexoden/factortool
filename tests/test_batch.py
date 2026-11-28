# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the adaptive batch size controller."""

from __future__ import annotations

import json
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from factortool.batch import BATCH_MAX_GROWTH, BATCH_RESET_AGE, BatchController, BatchKey, BatchState

KEY = BatchKey(backend="factordb", factoring_mode="standard", max_threads=2, min_digits=60)

TARGET_DURATION = 600.0

# The throughput saved by save_throughput, in numbers per second.
ITEMS_PER_SECOND = 10.0

# The largest batch saved by save_throughput, large enough not to limit the batch size.
LARGEST_BATCH = 10_000


def save_throughput(state_path: Path, key: object, age: float = 0.0) -> None:
    """Save a throughput of ITEMS_PER_SECOND for the given key, measured the given number of seconds ago."""
    measurement = {
        "key": key,
        "last_update": time.time() - age,
        "numbers": ITEMS_PER_SECOND * TARGET_DURATION,
        "seconds": TARGET_DURATION,
        "largest_batch": LARGEST_BATCH,
    }
    _ = state_path.write_text(json.dumps({"measurements": [measurement]}), encoding="utf-8")


def test_the_measured_throughput_carries_over_for_the_same_selection(tmp_path: Path) -> None:
    """Test that a later run with the same selection starts from the measured throughput."""
    save_throughput(tmp_path / "batch_state.json", KEY.model_dump())

    assert (
        BatchController(TARGET_DURATION, KEY, tmp_path / "batch_state.json").batch_size
        == TARGET_DURATION * ITEMS_PER_SECOND
    )


@pytest.mark.parametrize(
    "change", [{"backend": "mersenne_ca"}, {"max_threads": 8}, {"factoring_mode": "yafu"}, {"min_digits": 61}]
)
def test_a_change_of_selection_measures_afresh(tmp_path: Path, change: dict[str, object]) -> None:
    """Test that a throughput measured with a different backend, thread count, factoring mode or digits is not used."""
    save_throughput(tmp_path / "batch_state.json", KEY.model_dump())

    key = KEY.model_copy(update=change)

    assert BatchController(TARGET_DURATION, key, tmp_path / "batch_state.json").batch_size == 1


def test_a_run_with_another_selection_keeps_the_measured_throughput(tmp_path: Path) -> None:
    """Test that a run with a different selection measures its own throughput and leaves the first one alone."""
    state_path = tmp_path / "batch_state.json"
    save_throughput(state_path, KEY.model_dump())

    other_key = KEY.model_copy(update={"max_threads": 8})
    BatchController(TARGET_DURATION, other_key, state_path).record_batch(5, 10.0)

    assert BatchController(TARGET_DURATION, KEY, state_path).batch_size == TARGET_DURATION * ITEMS_PER_SECOND
    measurements = BatchState.model_validate_json(state_path.read_text(encoding="utf-8")).measurements
    assert {m.key.max_threads for m in measurements} == {KEY.max_threads, other_key.max_threads}


def test_a_run_with_the_same_selection_updates_its_measurement(tmp_path: Path) -> None:
    """Test that recording a batch replaces the selection's measurement rather than adding another."""
    state_path = tmp_path / "batch_state.json"
    save_throughput(state_path, KEY.model_dump())

    BatchController(TARGET_DURATION, KEY, state_path).record_batch(5, 10.0)

    measurements = BatchState.model_validate_json(state_path.read_text(encoding="utf-8")).measurements
    assert [m.key for m in measurements] == [KEY]


def test_a_stale_throughput_is_discarded(tmp_path: Path) -> None:
    """Test that a throughput not updated for longer than the reset age is neither used nor kept."""
    state_path = tmp_path / "batch_state.json"
    save_throughput(state_path, KEY.model_dump(), age=BATCH_RESET_AGE + 1)

    assert BatchController(TARGET_DURATION, KEY, state_path).batch_size == 1

    other_key = KEY.model_copy(update={"min_digits": 61})
    BatchController(TARGET_DURATION, other_key, state_path).record_batch(5, 10.0)

    measurements = BatchState.model_validate_json(state_path.read_text(encoding="utf-8")).measurements
    assert [m.key for m in measurements] == [other_key]


def test_a_state_in_the_single_selection_format_is_discarded(tmp_path: Path) -> None:
    """Test that a state written when only one selection's throughput was kept is not used."""
    state = {"last_update": time.time(), "key": KEY.model_dump(), "items_per_second": ITEMS_PER_SECOND}
    _ = (tmp_path / "batch_state.json").write_text(json.dumps(state), encoding="utf-8")

    assert BatchController(TARGET_DURATION, KEY, tmp_path / "batch_state.json").batch_size == 1


def test_the_first_measurement_sets_the_throughput(tmp_path: Path) -> None:
    """Test that with nothing measured yet, the first batch's throughput is taken as it is."""
    state_path = tmp_path / "batch_state.json"
    BatchController(TARGET_DURATION, KEY, state_path).record_batch(1, 10.0)

    measurement = BatchState.model_validate_json(state_path.read_text(encoding="utf-8")).measurements[0]

    assert measurement.items_per_second == pytest.approx(0.1)


@pytest.mark.parametrize(("largest_batch", "expected"), [(1, BATCH_MAX_GROWTH), (100, 200), (1000, 600)])
def test_the_batch_size_grows_at_most_by_the_growth_limit(tmp_path: Path, largest_batch: int, expected: int) -> None:
    """Test that the batch size is the throughput's, but at most BATCH_MAX_GROWTH times the largest measured batch."""
    state_path = tmp_path / "batch_state.json"
    measurement = {"key": KEY.model_dump(), "last_update": time.time(), "numbers": 10.0, "seconds": 10.0}
    state = {"measurements": [measurement | {"largest_batch": largest_batch}]}
    _ = state_path.write_text(json.dumps(state), encoding="utf-8")

    assert BatchController(TARGET_DURATION, KEY, state_path).batch_size == expected


def test_a_fast_first_number_doubles_the_batch_rather_than_filling_the_target(tmp_path: Path) -> None:
    """Test that a quickly factored first number grows the next batch only by the growth limit."""
    state_path = tmp_path / "batch_state.json"
    BatchController(TARGET_DURATION, KEY, state_path).record_batch(1, 0.1)

    assert BatchController(TARGET_DURATION, KEY, state_path).batch_size == BATCH_MAX_GROWTH


def test_a_batch_that_factored_nothing_counts_its_time(tmp_path: Path) -> None:
    """Test that a first batch that factored nothing keeps the batch at one, and its time still counts."""
    state_path = tmp_path / "batch_state.json"
    BatchController(TARGET_DURATION, KEY, state_path).record_batch(0, 30.0)

    controller = BatchController(TARGET_DURATION, KEY, state_path)
    assert controller.batch_size == 1

    controller.record_batch(1, 10.0)

    measurement = BatchState.model_validate_json(state_path.read_text(encoding="utf-8")).measurements[0]
    assert measurement.items_per_second == pytest.approx(1 / 40, rel=1e-3)


def test_a_slow_batch_after_a_fast_first_number_stops_the_growth(tmp_path: Path) -> None:
    """Test that a batch far slower than the first number outweighs it, rather than the batch size growing anyway."""
    state_path = tmp_path / "batch_state.json"
    BatchController(TARGET_DURATION, KEY, state_path).record_batch(1, 1.0)

    controller = BatchController(TARGET_DURATION, KEY, state_path)
    assert controller.batch_size == BATCH_MAX_GROWTH

    # The two numbers take the whole target duration, so the next batch should be about the same size.
    controller.record_batch(BATCH_MAX_GROWTH, TARGET_DURATION)

    assert BatchController(TARGET_DURATION, KEY, state_path).batch_size <= BATCH_MAX_GROWTH + 1


def test_a_slow_first_number_lets_the_batch_double_while_batches_stay_short(tmp_path: Path) -> None:
    """Test that batches far faster than the first number grow the batch size by the growth limit each time."""
    state_path = tmp_path / "batch_state.json"
    BatchController(TARGET_DURATION, KEY, state_path).record_batch(1, 60.0)

    sizes: list[int] = []

    for _ in range(4):
        controller = BatchController(TARGET_DURATION, KEY, state_path)
        sizes.append(controller.batch_size)
        controller.record_batch(controller.batch_size, controller.batch_size * 1.0)

    assert sizes == [2, 4, 8, 16]
