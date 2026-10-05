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


class AssignmentState(BaseModel):
    """Composites carried over from a previous run."""

    last_update: float
    backend: str
    composites: list[int]


class AssignmentStore:
    """Stores assigned composites from a previous run that were not yet finished."""

    def __init__(self, state_path: Path, backend_name: str) -> None:
        """Initialize the assignment store."""
        self._state_path = state_path
        self._backend_name = backend_name

    def load(self) -> list[int]:
        """Read the composites retained from a previous run.

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

        if state.composites:
            logger.info(
                "Resuming {} composite{} from a previous run",
                len(state.composites),
                "" if len(state.composites) == 1 else "s",
            )

        return state.composites

    def save(self, composites: Iterable[int]) -> None:
        """Record the composites to be retained for the next run."""
        pending = sorted(composites)
        state = AssignmentState(last_update=time.time(), backend=self._backend_name, composites=pending)

        try:
            safe_write(self._state_path, state.model_dump_json().encode("utf-8"))
        except OSError as e:
            logger.error("Failed to save assignment state to {}: {}", self._state_path, e)
            return

        if pending:
            logger.info(
                "Saved {} unfinished composite{} for the next run", len(pending), "" if len(pending) == 1 else "s"
            )

    def clear(self) -> None:
        """Clear the assignment state."""
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
