# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for unfinished assignment persistence."""

from __future__ import annotations

import json
import time

from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from factortool.assignments import ASSIGNMENT_EXPIRY_FUDGE_FACTOR, AssignmentState, AssignmentStore, select_unfinished
from factortool.cli.main import preserve_unfinished

from .helpers import make_number

# How long mersenne.ca reserves an assignment.
LIFETIME = 3600.0


def write_state(state_path: Path, fields: dict[str, object], last_update: float | None = None) -> None:
    """Write a raw mersenne.ca assignment state file."""
    state = {"last_update": time.time() if last_update is None else last_update, "backend": "mersenne_ca", **fields}
    state_path.write_text(json.dumps(state), encoding="utf-8")


def read_expiry(state_path: Path, n: int) -> float:
    """Read back the stored expiry for one assignment.

    Returns:
        float: When the claim on the assignment expires.
    """
    state = AssignmentState.model_validate_json(state_path.read_text(encoding="utf-8"))

    return next(x.expires_at for x in state.assignments if x.n == n)


def test_load_returns_nothing_without_a_state_file(tmp_path: Path) -> None:
    """Test that loading from a non-existent state file returns an empty list."""
    assert AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", LIFETIME).load() == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """Test that saving and then loading returns the same list of composites."""
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", LIFETIME)
    store.note_assigned([101, 103])
    store.save([103, 101])

    assert store.load() == [101, 103]


def test_load_ignores_state_from_a_different_backend(tmp_path: Path) -> None:
    """Test that loading ignores state from a different backend."""
    state_path = tmp_path / "assignments.json"
    store = AssignmentStore(state_path, "mersenne_ca", LIFETIME)
    store.note_assigned([101, 103])
    store.save([101, 103])

    assert AssignmentStore(state_path, "factordb", LIFETIME).load() == []


def test_load_ignores_a_corrupt_state_file(tmp_path: Path) -> None:
    """Test that loading ignores a corrupt state file."""
    state_path = tmp_path / "assignments.json"
    state_path.write_text("{not json", encoding="utf-8")

    assert AssignmentStore(state_path, "mersenne_ca", LIFETIME).load() == []


def test_clear_discards_pending_work(tmp_path: Path) -> None:
    """Test that clearing the store discards any pending work."""
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", LIFETIME)
    store.note_assigned([101])
    store.save([101])
    store.clear()

    assert store.load() == []


def test_load_discards_a_lapsed_assignment(tmp_path: Path) -> None:
    """Test that loading discards an assignment whose claim has lapsed."""
    state_path = tmp_path / "assignments.json"
    store = AssignmentStore(state_path, "mersenne_ca", -LIFETIME)
    store.note_assigned([101])
    store.save([101])

    assert AssignmentStore(state_path, "mersenne_ca", LIFETIME).load() == []


def test_carrying_an_assignment_over_does_not_renew_it(tmp_path: Path) -> None:
    """Test that carrying an assignment over to another run does not renew its expiry."""
    state_path = tmp_path / "assignments.json"
    first = AssignmentStore(state_path, "mersenne_ca", LIFETIME)
    first.note_assigned([101])
    first.save([101])
    original = read_expiry(state_path, 101)

    second = AssignmentStore(state_path, "mersenne_ca", LIFETIME)
    second.load()
    second.save([101])

    assert read_expiry(state_path, 101) == original


def test_a_composite_never_seen_assigned_is_not_retained(tmp_path: Path) -> None:
    """Test that a composite never seen assigned is treated as freshly assigned."""
    state_path = tmp_path / "assignments.json"
    store = AssignmentStore(state_path, "mersenne_ca", LIFETIME)
    store.note_assigned([101])
    store.save([101, 103])

    assert AssignmentStore(state_path, "mersenne_ca", LIFETIME).load() == [101]


# Remaining lifetimes either side of the fudge factor.
CLOSE_TO_EXPIRY = [(ASSIGNMENT_EXPIRY_FUDGE_FACTOR / 2, False), (ASSIGNMENT_EXPIRY_FUDGE_FACTOR * 2, True)]


@pytest.mark.parametrize(("remaining", "kept"), CLOSE_TO_EXPIRY)
def test_load_discards_an_assignment_close_to_expiry(tmp_path: Path, remaining: float, *, kept: bool) -> None:
    """Test that loading discards an assignment expiring within the fudge factor, even though it has not expired."""
    state_path = tmp_path / "assignments.json"
    write_state(state_path, {"assignments": [{"n": 101, "expires_at": time.time() + remaining}]})

    assert AssignmentStore(state_path, "mersenne_ca", LIFETIME).load() == ([101] if kept else [])


@pytest.mark.parametrize(("remaining", "kept"), CLOSE_TO_EXPIRY)
def test_save_omits_an_assignment_close_to_expiry(tmp_path: Path, remaining: float, *, kept: bool) -> None:
    """Test that saving omits an assignment expiring within the fudge factor."""
    state_path = tmp_path / "assignments.json"
    store = AssignmentStore(state_path, "mersenne_ca", remaining)
    store.note_assigned([101])
    store.save([101])

    state = AssignmentState.model_validate_json(state_path.read_text(encoding="utf-8"))
    assert [x.n for x in state.assignments] == ([101] if kept else [])


def test_load_removes_expired_assignments_from_the_state(tmp_path: Path) -> None:
    """Test that loading rewrites the state without expired assignments."""
    state_path = tmp_path / "assignments.json"
    now = time.time()
    write_state(
        state_path,
        {"assignments": [{"n": 101, "expires_at": now - LIFETIME}, {"n": 103, "expires_at": now + LIFETIME}]},
    )

    AssignmentStore(state_path, "mersenne_ca", LIFETIME).load()

    state = AssignmentState.model_validate_json(state_path.read_text(encoding="utf-8"))
    assert [x.n for x in state.assignments] == [103]


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


def test_preserve_unfinished_keeps_work_left_by_a_completed_run(tmp_path: Path) -> None:
    """Test that a number a method could not finish is reported or retained, not dropped with the finished ones."""
    backend = Mock(assigns_work=True)
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", LIFETIME)
    store.note_assigned([100, 200, 300])
    store.save([100, 200, 300])

    done = make_number(100)
    done.prime_factors = [2, 2, 5, 5]
    done.composite_factors = []

    # A method can return a composite cofactor without the engine reporting anything other than success.
    partially_factored = make_number(200)
    partially_factored.prime_factors = [2]
    partially_factored.composite_factors = [100]
    partially_factored.report_partial = Mock()  # type: ignore[method-assign]

    untouched = make_number(300)

    preserve_unfinished(backend, store, [done, partially_factored, untouched])

    partially_factored.report_partial.assert_called_once_with()
    assert store.load() == [300]


def test_save_raises_when_the_state_cannot_be_written(tmp_path: Path) -> None:
    """Test that a failure to retain assignments reaches the caller instead of being swallowed."""
    store = AssignmentStore(tmp_path / "missing" / "assignments.json", "mersenne_ca", LIFETIME)
    store.note_assigned([100])

    with pytest.raises(OSError, match="missing"):
        store.save([100])


def test_load_survives_being_unable_to_remove_expired_assignments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that the remaining assignments are still resumed when the expired ones cannot be removed from the state."""
    state_path = tmp_path / "assignments.json"
    assignments = [{"n": 100, "expires_at": time.time() + LIFETIME}, {"n": 200, "expires_at": 0.0}]
    write_state(state_path, {"assignments": assignments})

    monkeypatch.setattr("factortool.assignments.safe_write", Mock(side_effect=OSError("disk full")), raising=True)

    assert AssignmentStore(state_path, "mersenne_ca", LIFETIME).load() == [100]
