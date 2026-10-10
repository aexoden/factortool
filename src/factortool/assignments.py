# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Persistence for composites assigned but not yet finished."""

from __future__ import annotations

import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ValidationError

from factortool.util import safe_write

if TYPE_CHECKING:
    from factortool.number import Number

# Assignments take time to complete, and they may be reassigned immediately after expiry. If we don't have a reasonable
# chance of finishing the assignment before expiry, we should just drop it.
ASSIGNMENT_EXPIRY_FUDGE_FACTOR = 600.0


def assignment_expired(expires_at: float) -> bool:
    """Whether an assignment is too close to its expiry to be worth starting.

    Returns:
        bool: True if the assignment expires within the fudge factor, or has already expired.
    """
    return expires_at - ASSIGNMENT_EXPIRY_FUDGE_FACTOR <= time.time()


class Assignment(BaseModel):
    """An assignment of a composite number."""

    n: int
    expires_at: float


class AssignmentState(BaseModel):
    """Composites carried over from a previous run."""

    last_update: float
    backend: str
    assignments: list[Assignment] = []


class AssignmentStore:
    """Stores assigned composites from a previous run that were not yet finished."""

    def __init__(self, state_path: Path, backend_name: str, lifetime: float) -> None:
        """Initialize the assignment store."""
        self._state_path = state_path
        self._backend_name = backend_name
        self._lifetime = lifetime
        self._expiry: dict[int, float] = {}

    def load(self) -> list[int]:
        """Read the composites retained from a previous run, discarding any whose assignment has expired.

        Assignments are considered expired shortly before their designated expiry time to minimize the risk of working
        on something that has been reassigned.

        Returns:
            list[int]: The list of composites retained from a previous run.
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
        pending = [x for x in assignments if not assignment_expired(x.expires_at)]
        expired = len(assignments) - len(pending)

        if expired > 0:
            logger.warning("Discarding {} composite{} with an expired assignment", expired, "" if expired == 1 else "s")

            try:
                self._write(pending)
            except OSError as e:
                logger.warning("Failed to update assignment state at {}: {}", self._state_path, e)

        if pending:
            logger.info("Resuming {} composite{} from a previous run", len(pending), "" if len(pending) == 1 else "s")

        self._expiry |= {x.n: x.expires_at for x in pending}

        return sorted(x.n for x in pending)

    def note_assigned(self, composites: Iterable[int]) -> None:
        """Record that the given composites have been assigned."""
        expires_at = time.time() + self._lifetime

        for n in composites:
            self._expiry[n] = expires_at

    def expires_at(self, n: int) -> float | None:
        """Get the expiry time for the given composite, if known.

        Returns:
            float | None: The expiry time for the given composite, or None if unknown.
        """
        return self._expiry.get(n)

    def save(self, composites: Iterable[int]) -> None:
        """Record the composites to be retained for the next run, omitting any expired assignments.

        Raises:
            OSError: If the state file cannot be written.
        """
        pending: list[Assignment] = []
        unknown: list[int] = []
        expired = 0

        for n in sorted(composites):
            if (expires_at := self._expiry.get(n)) is None:
                unknown.append(n)
            elif assignment_expired(expires_at):
                expired += 1
            else:
                pending.append(Assignment(n=n, expires_at=expires_at))

        if unknown:
            logger.error("Discarding composites with no known expiry: {}", ", ".join(str(n) for n in unknown))

        if expired > 0:
            logger.info(
                "Discarding {} composite{} with expired or near expired assignments",
                expired,
                "" if expired == 1 else "s",
            )

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
        self._expiry.clear()
        self.save([])


def select_unfinished(numbers: Collection[Number]) -> tuple[list[Number], list[Number]]:
    """Split unfactored numbers into those that are partially finished and those that are untouched.

    Returns:
        tuple[list[Number], list[Number]]: A tuple containing two lists:
            - The first list contains partially finished numbers.
            - The second list contains untouched numbers.
    """
    unfinished = [x for x in numbers if not x.factored]
    partial = [x for x in unfinished if len(x.prime_factors) > 0]
    reported = {x.n for x in partial}
    untouched = [x for x in unfinished if x.n not in reported]

    return partial, untouched
