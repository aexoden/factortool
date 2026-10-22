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

from factortool.batch import BatchController, BatchKey

KEY = BatchKey(backend="factordb", factoring_mode="standard", max_threads=2, min_digits=60)

TARGET_DURATION = 600.0

# The throughput saved by save_throughput, in numbers per second.
ITEMS_PER_SECOND = 10.0


def save_throughput(state_path: Path, key: object) -> None:
    """Save a recently measured throughput of ITEMS_PER_SECOND for the given key."""
    state = {"last_update": time.time(), "key": key, "items_per_second": ITEMS_PER_SECOND}
    state_path.write_text(json.dumps(state), encoding="utf-8")


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
def test_a_change_of_selection_resets_the_measured_throughput(tmp_path: Path, change: dict[str, object]) -> None:
    """Test that a throughput measured with a different backend, thread count, factoring mode or digits is not used."""
    save_throughput(tmp_path / "batch_state.json", KEY.model_dump())

    key = KEY.model_copy(update=change)

    assert BatchController(TARGET_DURATION, key, tmp_path / "batch_state.json").batch_size == 1


def test_a_state_without_the_thread_count_is_discarded(tmp_path: Path) -> None:
    """Test that a state written before the thread count and factoring mode were part of the key is not used."""
    save_throughput(tmp_path / "batch_state.json", {"backend": "factordb", "min_digits": 60})

    assert BatchController(TARGET_DURATION, KEY, tmp_path / "batch_state.json").batch_size == 1
