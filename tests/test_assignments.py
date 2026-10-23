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

from factortool.assignments import Assignment, AssignmentState, AssignmentStore

# How long mersenne.ca reserves an assignment.
LIFETIME = 3600.0


def assigned(*composites: int, lifetime: float = LIFETIME) -> list[Assignment]:
    """Build assignments for the given composites that expire after the given lifetime.

    Returns:
        list[Assignment]: The assignments, with no factors found.
    """
    expires_at = time.time() + lifetime

    return [Assignment(n=n, expires_at=expires_at) for n in composites]


def write_state(state_path: Path, fields: dict[str, object], last_update: float | None = None) -> None:
    """Write a raw mersenne.ca assignment state file."""
    state = {"last_update": time.time() if last_update is None else last_update, "backend": "mersenne_ca", **fields}
    state_path.write_text(json.dumps(state), encoding="utf-8")


def test_load_returns_nothing_without_a_state_file(tmp_path: Path) -> None:
    """Test that loading from a non-existent state file returns an empty list."""
    assert AssignmentStore(tmp_path / "assignments.json", "mersenne_ca").load() == []


def read_state(state_path: Path) -> list[Assignment]:
    """Read the assignments in a state file as they stand.

    Returns:
        list[Assignment]: The assignments in the order they were written.
    """
    return AssignmentState.model_validate_json(state_path.read_text(encoding="utf-8")).assignments


def test_record_and_load_round_trip(tmp_path: Path) -> None:
    """Test that recording and then loading returns the same assignments, smallest composite first."""
    state_path = tmp_path / "assignments.json"
    first, second = assigned(101, 103)
    AssignmentStore(state_path, "mersenne_ca").record([second, first])

    assert AssignmentStore(state_path, "mersenne_ca").load() == [first, second]


def test_factors_found_so_far_round_trip(tmp_path: Path) -> None:
    """Test that the factors found for an unfinished composite are carried over with its assignment."""
    state_path = tmp_path / "assignments.json"
    assignment = Assignment(n=15015, expires_at=time.time() + LIFETIME, factors=(3, 5, 1001))
    AssignmentStore(state_path, "mersenne_ca").record([assignment])

    assert AssignmentStore(state_path, "mersenne_ca").load() == [assignment]


def test_every_change_is_written_at_once(tmp_path: Path) -> None:
    """Test that the state follows each recorded assignment, update and release without waiting for the run to end."""
    state_path = tmp_path / "assignments.json"
    store = AssignmentStore(state_path, "mersenne_ca")
    first, second = assigned(101, 15015)

    store.record([first, second])
    assert read_state(state_path) == [first, second]

    progress = second.model_copy(update={"factors": (3, 5, 1001)})
    store.record([progress])
    assert read_state(state_path) == [first, progress]

    store.release([101])
    assert read_state(state_path) == [progress]


def test_an_unchanged_assignment_is_not_written_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that recording what is already recorded, or releasing what is not tracked, leaves the state file alone."""
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    assignments = assigned(101, 103)
    store.record(assignments)

    safe_write = Mock()
    monkeypatch.setattr("factortool.assignments.safe_write", safe_write, raising=True)

    store.record(assignments)
    store.release([107])

    safe_write.assert_not_called()


def test_loading_tracks_the_assignments_it_resumes(tmp_path: Path) -> None:
    """Test that a resumed assignment stays in the state as other assignments are recorded."""
    state_path = tmp_path / "assignments.json"
    resumed, fetched = assigned(101, 103)
    AssignmentStore(state_path, "mersenne_ca").record([resumed])

    store = AssignmentStore(state_path, "mersenne_ca")
    store.load()
    store.record([fetched])

    assert read_state(state_path) == [resumed, fetched]


def test_a_failure_to_record_is_logged_once_and_the_run_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that the state file failing mid-run neither raises nor floods the log, and the final compaction raises."""
    error = Mock()
    monkeypatch.setattr("factortool.assignments.logger.error", error, raising=True)
    store = AssignmentStore(tmp_path / "missing" / "assignments.json", "mersenne_ca")

    store.record(assigned(101))
    store.record(assigned(103))
    store.release([101])

    assert error.call_count == 1

    with pytest.raises(OSError, match="missing"):
        store.compact()


def test_compact_drops_expired_assignments(tmp_path: Path) -> None:
    """Test that compacting the state at the end of a run leaves only the assignments still held."""
    state_path = tmp_path / "assignments.json"
    store = AssignmentStore(state_path, "mersenne_ca")
    (expired,) = assigned(101, lifetime=-1.0)
    (held,) = assigned(103)
    store.record([expired, held])

    store.compact()

    assert read_state(state_path) == [held]


def test_load_accepts_an_assignment_saved_without_factors(tmp_path: Path) -> None:
    """Test that an assignment saved before factors were retained is still resumed."""
    state_path = tmp_path / "assignments.json"
    write_state(state_path, {"assignments": [{"n": 101, "expires_at": time.time() + LIFETIME}]})

    (assignment,) = AssignmentStore(state_path, "mersenne_ca").load()

    assert (assignment.n, assignment.factors) == (101, ())


def test_load_ignores_state_from_a_different_backend(tmp_path: Path) -> None:
    """Test that loading ignores state from a different backend."""
    state_path = tmp_path / "assignments.json"
    AssignmentStore(state_path, "mersenne_ca").record(assigned(101, 103))

    assert AssignmentStore(state_path, "factordb").load() == []


def test_load_ignores_a_corrupt_state_file(tmp_path: Path) -> None:
    """Test that loading ignores a corrupt state file."""
    state_path = tmp_path / "assignments.json"
    state_path.write_text("{not json", encoding="utf-8")

    assert AssignmentStore(state_path, "mersenne_ca").load() == []


def test_load_ignores_a_state_file_with_a_number_that_cannot_be_factored(tmp_path: Path) -> None:
    """Test that a state file holding a number below 2 is treated as corrupt."""
    state_path = tmp_path / "assignments.json"
    write_state(state_path, {"assignments": [{"n": 1, "expires_at": time.time() + LIFETIME}]})

    assert AssignmentStore(state_path, "mersenne_ca").load() == []


# Remaining lifetimes either side of the expiry. Nothing is held back for being merely close to it.
AROUND_EXPIRY = [(-1.0, False), (60.0, True)]


@pytest.mark.parametrize(("remaining", "kept"), AROUND_EXPIRY)
def test_load_discards_only_an_expired_assignment(tmp_path: Path, remaining: float, *, kept: bool) -> None:
    """Test that loading discards an assignment once it has expired, and not merely for being close to it."""
    state_path = tmp_path / "assignments.json"
    AssignmentStore(state_path, "mersenne_ca").record(assigned(101, lifetime=remaining))

    assert [x.n for x in AssignmentStore(state_path, "mersenne_ca").load()] == ([101] if kept else [])


def test_load_removes_expired_assignments_from_the_state(tmp_path: Path) -> None:
    """Test that loading rewrites the state without expired assignments."""
    state_path = tmp_path / "assignments.json"
    now = time.time()
    write_state(
        state_path,
        {"assignments": [{"n": 101, "expires_at": now - LIFETIME}, {"n": 103, "expires_at": now + LIFETIME}]},
    )

    AssignmentStore(state_path, "mersenne_ca").load()

    assert [x.n for x in read_state(state_path)] == [103]


def test_load_survives_being_unable_to_remove_expired_assignments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that the remaining assignments are still resumed when the expired ones cannot be removed from the state."""
    state_path = tmp_path / "assignments.json"
    assignments = [{"n": 100, "expires_at": time.time() + LIFETIME}, {"n": 200, "expires_at": 0.0}]
    write_state(state_path, {"assignments": assignments})

    monkeypatch.setattr("factortool.assignments.safe_write", Mock(side_effect=OSError("disk full")), raising=True)

    assert [x.n for x in AssignmentStore(state_path, "mersenne_ca").load()] == [100]
