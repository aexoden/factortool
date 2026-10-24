# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Persistence for composites assigned but not yet finished."""

from __future__ import annotations

import threading
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from factortool.constants import MIN_FACTORABLE
from factortool.util import format_count, safe_write


def assignment_expired(expires_at: float) -> bool:
    """Whether an assignment has expired.

    Returns:
        bool: True if the assignment has expired.
    """
    return expires_at <= time.time()


class Assignment(BaseModel):
    """An assignment of a composite number, with any factors found so far."""

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=MIN_FACTORABLE)
    expires_at: float
    factors: tuple[int, ...] = ()


class AssignmentState(BaseModel):
    """Composites carried over from a previous run."""

    last_update: float
    backend: str
    assignments: list[Assignment] = []


class AssignmentStore:
    """Tracks composites assigned to this client that are not yet finished.

    Every change is written to the state file as it is made.
    """

    def __init__(self, state_path: Path, backend_name: str) -> None:
        """Initialize the assignment store."""
        self._state_path = state_path
        self._backend_name = backend_name
        self._lock = threading.Lock()
        self._assignments: dict[int, Assignment] = {}
        self._write_failed = False

    def load(self) -> list[Assignment]:
        """Read the assignments retained by a previous run, discarding any that have expired, and track the rest.

        Returns:
            list[Assignment]: The assignments retained by a previous run, smallest composite first.
        """
        pending = self._read()

        with self._lock:
            self._assignments = {x.n: x for x in pending}

        return pending

    def record(self, assignments: Iterable[Assignment]) -> None:
        """Track new assignments or the latest progress on existing ones, writing the state if anything changed."""
        with self._lock:
            changed = False

            for assignment in assignments:
                if self._assignments.get(assignment.n) != assignment:
                    self._assignments[assignment.n] = assignment
                    changed = True

            if changed:
                self._write_logged()

    def release(self, composites: Iterable[int]) -> None:
        """Stop tracking the given composites, writing the state if any were tracked."""
        with self._lock:
            released = [self._assignments.pop(n) for n in set(composites) if n in self._assignments]

            if released:
                self._write_logged()

    def compact(self) -> None:
        """Rewrite the state with only the assignments that have not yet expired.

        Raises:
            OSError: If the state file cannot be written.
        """
        with self._lock:
            self._assignments = {n: x for n, x in self._assignments.items() if not assignment_expired(x.expires_at)}
            self._write()
            pending = len(self._assignments)

        if pending:
            logger.info("Saved {} for the next run", format_count(pending, "unfinished composite"))

    def _read(self) -> list[Assignment]:
        """Read the unexpired assignments from the state file, rewriting it without any that have expired.

        Returns:
            list[Assignment]: The unexpired assignments, smallest composite first.
        """
        if not self._state_path.exists():
            return []

        try:
            state = AssignmentState.model_validate_json(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as e:
            logger.warning("Ignoring unreadable assignment state at {}: {}", self._state_path, e)
            return []

        if state.backend != self._backend_name:
            return []

        assignments = state.assignments
        pending = sorted((x for x in assignments if not assignment_expired(x.expires_at)), key=lambda x: x.n)
        expired = len(assignments) - len(pending)

        if expired > 0:
            logger.warning("Discarding {} with an expired assignment", format_count(expired, "composite"))

            try:
                self._write_state(pending)
            except OSError as e:
                logger.warning("Failed to update assignment state at {}: {}", self._state_path, e)

        return pending

    def _write_logged(self) -> None:
        """Write the tracked assignments, logging the first failure instead of raising it. Requires the lock."""
        try:
            self._write()
        except OSError as e:
            if not self._write_failed:
                logger.error("Failed to record assignments at {}: {}", self._state_path, e)
                self._write_failed = True

    def _write(self) -> None:
        """Write the tracked assignments to the state file. Requires the lock."""
        self._write_state(sorted(self._assignments.values(), key=lambda x: x.n))

    def _write_state(self, assignments: list[Assignment]) -> None:
        """Write the given assignments to the state file."""
        state = AssignmentState(last_update=time.time(), backend=self._backend_name, assignments=assignments)
        safe_write(self._state_path, state.model_dump_json().encode("utf-8"))
