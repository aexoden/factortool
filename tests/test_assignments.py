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
from factortool.cli.main import preserve_unfinished

if TYPE_CHECKING:
    from factortool.number import Number

from .helpers import make_number

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


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """Test that saving and then loading returns the same assignments, smallest composite first."""
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    first, second = assigned(101, 103)
    store.save([second, first])

    assert store.load() == [first, second]


def test_factors_found_so_far_round_trip(tmp_path: Path) -> None:
    """Test that the factors found for an unfinished composite are carried over with its assignment."""
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    assignment = Assignment(n=15015, expires_at=time.time() + LIFETIME, factors=[3, 5, 1001])
    store.save([assignment])

    assert store.load() == [assignment]


def test_load_accepts_an_assignment_saved_without_factors(tmp_path: Path) -> None:
    """Test that an assignment saved before factors were retained is still resumed."""
    state_path = tmp_path / "assignments.json"
    write_state(state_path, {"assignments": [{"n": 101, "expires_at": time.time() + LIFETIME}]})

    (assignment,) = AssignmentStore(state_path, "mersenne_ca").load()

    assert (assignment.n, assignment.factors) == (101, [])


def test_load_ignores_state_from_a_different_backend(tmp_path: Path) -> None:
    """Test that loading ignores state from a different backend."""
    state_path = tmp_path / "assignments.json"
    AssignmentStore(state_path, "mersenne_ca").save(assigned(101, 103))

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


def test_clear_discards_pending_work(tmp_path: Path) -> None:
    """Test that clearing the store discards any pending work."""
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    store.save(assigned(101))
    store.clear()

    assert store.load() == []


# Remaining lifetimes either side of the expiry. Nothing is held back for being merely close to it.
AROUND_EXPIRY = [(-1.0, False), (60.0, True)]


@pytest.mark.parametrize(("remaining", "kept"), AROUND_EXPIRY)
def test_load_discards_only_an_expired_assignment(tmp_path: Path, remaining: float, *, kept: bool) -> None:
    """Test that loading discards an assignment once it has expired, and not merely for being close to it."""
    state_path = tmp_path / "assignments.json"
    AssignmentStore(state_path, "mersenne_ca").save(assigned(101, lifetime=remaining))

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

    state = AssignmentState.model_validate_json(state_path.read_text(encoding="utf-8"))
    assert [x.n for x in state.assignments] == [103]


def make_unfinished(expires_at: float | None) -> tuple[list[Number], Number]:
    """Build a finished, a partially factored and an untouched number sharing an assignment expiry.

    Returns:
        tuple[list[Number], Number]: All three numbers, and the partially factored one by itself.
    """
    done = make_number(100)
    done.prime_factors = [2, 2, 5, 5]
    done.composite_factors = []

    # A method can return a composite cofactor without the engine reporting anything other than success.
    partially_factored = make_number(200)
    partially_factored.prime_factors = [2]
    partially_factored.composite_factors = [100]
    partially_factored.report_partial = Mock()  # type: ignore[method-assign]

    numbers = [done, partially_factored, make_number(300)]

    for number in numbers:
        number.expires_at = expires_at

    return numbers, partially_factored


@pytest.mark.parametrize("accepts_partial_results", [True, False])
def test_preserve_unfinished_retains_work_whose_assignment_is_held(
    tmp_path: Path, *, accepts_partial_results: bool
) -> None:
    """Test that unfinished numbers are retained with their factors, not reported, while their assignment is held."""
    backend = Mock(assigns_work=True, accepts_partial_results=accepts_partial_results)
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    expires_at = time.time() + LIFETIME
    numbers, partially_factored = make_unfinished(expires_at)

    preserve_unfinished(backend, store, numbers)

    partially_factored.report_partial.assert_not_called()  # type: ignore[attr-defined]
    assert store.load() == [
        Assignment(n=200, expires_at=expires_at, factors=[2, 100]),
        Assignment(n=300, expires_at=expires_at, factors=[300]),
    ]


def test_preserve_unfinished_discards_a_partial_result_the_backend_does_not_want(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a partial factorization is neither reported nor retained once its assignment has expired."""
    warning = Mock()
    monkeypatch.setattr("factortool.cli.main.logger.warning", warning, raising=True)
    backend = Mock(assigns_work=True, accepts_partial_results=False)
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    numbers, partially_factored = make_unfinished(time.time() - 1.0)

    preserve_unfinished(backend, store, numbers)

    partially_factored.report_partial.assert_not_called()  # type: ignore[attr-defined]
    assert store.load() == []
    assert [call.args[1] for call in warning.call_args_list] == ["200 <3>"]


def test_preserve_unfinished_reports_a_partial_result_whose_assignment_expired(tmp_path: Path) -> None:
    """Test that a backend accepting partial factorizations is sent one that can no longer be completed."""
    backend = Mock(assigns_work=True, accepts_partial_results=True)
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca")
    numbers, partially_factored = make_unfinished(time.time() - 1.0)

    preserve_unfinished(backend, store, numbers)

    partially_factored.report_partial.assert_called_once_with()  # type: ignore[attr-defined]
    assert store.load() == []


def test_save_raises_when_the_state_cannot_be_written(tmp_path: Path) -> None:
    """Test that a failure to retain assignments reaches the caller instead of being swallowed."""
    store = AssignmentStore(tmp_path / "missing" / "assignments.json", "mersenne_ca")

    with pytest.raises(OSError, match="missing"):
        store.save(assigned(100))


def test_load_survives_being_unable_to_remove_expired_assignments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that the remaining assignments are still resumed when the expired ones cannot be removed from the state."""
    state_path = tmp_path / "assignments.json"
    assignments = [{"n": 100, "expires_at": time.time() + LIFETIME}, {"n": 200, "expires_at": 0.0}]
    write_state(state_path, {"assignments": assignments})

    monkeypatch.setattr("factortool.assignments.safe_write", Mock(side_effect=OSError("disk full")), raising=True)

    assert [x.n for x in AssignmentStore(state_path, "mersenne_ca").load()] == [100]
