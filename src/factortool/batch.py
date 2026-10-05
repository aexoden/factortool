# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Adaptive batch size controller for factorization tasks."""

from __future__ import annotations

import math
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ValidationError

from factortool.util import safe_write

BATCH_DECAY_RATE = 2 / 1800
BATCH_RESET_AGE = 7200


class BatchKey(BaseModel):
    """The work selection a measured throughput applies to."""

    backend: str = "factordb"
    min_digits: int
    max_digits: int | None = None
    skip_count: int = 0


class BatchState(BaseModel):
    """State information for batch size controller."""

    last_update: float
    key: BatchKey
    items_per_second: float


class BatchController:
    """Controller for adaptive batch sizing."""

    def __init__(self, target_duration: float, key: BatchKey, state_path: Path) -> None:
        """Initialize the batch controller."""
        self._target_duration = target_duration
        self._state_path = state_path

        self._reset_state(key)
        self._load_state()

        if self._state.key != key:
            logger.info("Resetting batch size due to configuration change")
            self._reset_state(key)

        time_since_update = time.time() - self._state.last_update

        if time_since_update > BATCH_RESET_AGE:
            logger.info("Resetting batch size due to inactivity")
            self._reset_state(key)

    def _reset_state(self, key: BatchKey) -> None:
        self._state = BatchState(last_update=time.time(), key=key, items_per_second=0)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return

        try:
            self._state = BatchState.model_validate_json(self._state_path.read_text(encoding="utf-8"))
        except ValidationError:
            # A state file written before the key was introduced carries no usable throughput for this selection.
            logger.info("Discarding batch state written in an older format")

    def _save_state(self) -> None:
        safe_write(self._state_path, self._state.model_dump_json().encode("utf-8"))

    @property
    def batch_size(self) -> int:
        """Current batch size."""
        return max(1, int(self._state.items_per_second * self._target_duration))

    def record_batch(self, batch_size: int, duration: float) -> None:
        """Record the processing of a batch."""
        items_per_second = batch_size / duration if duration > 0 else 0

        batch_size_factor = 1 - math.exp(-0.15 * batch_size)
        items_per_second = items_per_second * batch_size_factor + self._state.items_per_second * (1 - batch_size_factor)

        time_since_update = time.time() - self._state.last_update
        factor = pow(1 - BATCH_DECAY_RATE, time_since_update)
        new_items_per_second = self._state.items_per_second * factor + items_per_second * (1 - factor)

        logger.info(
            "Updating average items per second from {:.3f} to {:.3f} (Actual: {:.3f})",
            self._state.items_per_second,
            new_items_per_second,
            items_per_second,
        )

        self._state.last_update = time.time()
        self._state.items_per_second = new_items_per_second
        self._save_state()
