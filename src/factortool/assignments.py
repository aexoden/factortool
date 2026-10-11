# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Persistence for composites assigned but not yet finished."""

from __future__ import annotations

import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from factortool.constants import MIN_FACTORABLE
from factortool.util import safe_write


def assignment_expired(expires_at: float) -> bool:
    """Whether an assignment has expired.

    Returns:
        bool: True if the assignment has expired.
    """
    return expires_at <= time.time()


class Assignment(BaseModel):
    """An assignment of a composite number, with any factors found so far."""

    n: int = Field(ge=MIN_FACTORABLE)
    expires_at: float
    factors: list[int] = []


class AssignmentState(BaseModel):
    """Composites carried over from a previous run."""

    last_update: float
    backend: str
    assignments: list[Assignment] = []


class AssignmentStore:
    """Stores assigned composites from a previous run that were not yet finished."""

    def __init__(self, state_path: Path, backend_name: str) -> None:
        """Initialize the assignment store."""
        self._state_path = state_path
        self._backend_name = backend_name

    def load(self) -> list[Assignment]:
        """Read the assignments retained from a previous run, discarding any that have expired.

        Returns:
            list[Assignment]: The list of assignments retained from a previous run.
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
            logger.warning("Discarding {} composite{} with an expired assignment", expired, "" if expired == 1 else "s")

            try:
                self._write(pending)
            except OSError as e:
                logger.warning("Failed to update assignment state at {}: {}", self._state_path, e)

        if pending:
            logger.info("Resuming {} composite{} from a previous run", len(pending), "" if len(pending) == 1 else "s")

        return pending

    def save(self, assignments: Iterable[Assignment]) -> None:
        """Record the assignments to be retained for the next run.

        Raises:
            OSError: If the state file cannot be written.
        """
        pending = sorted(assignments, key=lambda x: x.n)
        self._write(pending)

        if pending:
            logger.info(
                "Saved {} unfinished composite{} for the next run", len(pending), "" if len(pending) == 1 else "s"
            )

    def _write(self, assignments: list[Assignment]) -> None:
        """Write the given assignments to the state file."""
        state = AssignmentState(last_update=time.time(), backend=self._backend_name, assignments=assignments)
        safe_write(self._state_path, state.model_dump_json().encode("utf-8"))

    def clear(self) -> None:
        """Clear the assignment state."""
        self.save([])
