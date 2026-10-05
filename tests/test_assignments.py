# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for unfinished assignment persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from factortool.assignments import AssignmentStore, select_unfinished

from .helpers import make_number


def test_load_returns_nothing_without_a_state_file(tmp_path: Path) -> None:
    """Test that loading from a non-existent state file returns an empty list."""
    assert AssignmentStore(tmp_path / "assignments.json", "mersenne_ca").load() == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """Test that saving and then loading returns the same list of composites."""
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    store.save([103, 101])

    assert store.load() == [101, 103]


def test_load_ignores_state_from_a_different_backend(tmp_path: Path) -> None:
    """Test that loading ignores state from a different backend."""
    state_path = tmp_path / "assignments.json"
    AssignmentStore(state_path, "mersenne_ca").save([101, 103])

    assert AssignmentStore(state_path, "factordb").load() == []


def test_load_ignores_a_corrupt_state_file(tmp_path: Path) -> None:
    """Test that loading ignores a corrupt state file."""
    state_path = tmp_path / "assignments.json"
    state_path.write_text("{not json", encoding="utf-8")

    assert AssignmentStore(state_path, "mersenne_ca").load() == []


def test_clear_discards_pending_work(tmp_path: Path) -> None:
    """Test that clearing the store discards any pending work."""
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    store.save([101])
    store.clear()

    assert store.load() == []


def test_select_unfinished_splits_partial_progress_from_untouched() -> None:
    """Test that select_unfinished correctly separates partially factored numbers from untouched ones."""
    done = make_number(100)
    done.prime_factors = [2, 2, 5, 5]
    done.composite_factors = []

    partially_factored = make_number(200)
    partially_factored.prime_factors = [2, 2]
    partially_factored.composite_factors = [50]

    untouched = make_number(300)

    partial, pending = select_unfinished([done, partially_factored, untouched])

    assert [x.n for x in partial] == [200]
    assert [x.n for x in pending] == [300]
