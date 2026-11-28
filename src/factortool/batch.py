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

# Seconds after which a throughput that has not been updated is discarded.
BATCH_RESET_AGE = 7200


class BatchKey(BaseModel):
    """The work selection a measured throughput applies to."""

    backend: str = "factordb"
    factoring_mode: str
    max_threads: int
    min_digits: int
    max_digits: int | None = None
    skip_count: int = 0


class BatchMeasurement(BaseModel):
    """The throughput measured for one work selection."""

    key: BatchKey
    last_update: float
    items_per_second: float


class BatchState(BaseModel):
    """The throughputs measured for each recently used work selection."""

    measurements: list[BatchMeasurement]


class BatchController:
    """Controller for adaptive batch sizing.

    A throughput is kept for each work selection, so a run with a different selection leaves the others' throughputs
    alone. One that has not been updated for BATCH_RESET_AGE seconds is discarded.
    """

    _measurement: BatchMeasurement
    _others: list[BatchMeasurement]

    def __init__(self, target_duration: float, key: BatchKey, state_path: Path) -> None:
        """Initialize the batch controller."""
        self._target_duration: float = target_duration
        self._state_path: Path = state_path

        now = time.time()
        measurements = [m for m in self._load_measurements() if now - m.last_update <= BATCH_RESET_AGE]
        self._others = [m for m in measurements if m.key != key]
        matching = [m for m in measurements if m.key == key]

        if matching:
            self._measurement = matching[0]
        else:
            logger.info("No recent batch measurement for this selection, so the batch size is measured afresh")
            self._measurement = BatchMeasurement(key=key, last_update=now, items_per_second=0)

    def _load_measurements(self) -> list[BatchMeasurement]:
        if not self._state_path.exists():
            return []

        try:
            return BatchState.model_validate_json(self._state_path.read_text(encoding="utf-8")).measurements
        except ValidationError:
            # A state file written in an earlier format carries no usable throughput.
            logger.info("Discarding batch state written in an older format")
            return []

    def _save_state(self) -> None:
        now = time.time()
        others = [m for m in self._others if now - m.last_update <= BATCH_RESET_AGE]
        state = BatchState(measurements=[self._measurement, *others])
        safe_write(self._state_path, state.model_dump_json().encode("utf-8"))

    @property
    def batch_size(self) -> int:
        """Current batch size."""
        return max(1, int(self._measurement.items_per_second * self._target_duration))

    def record_batch(self, batch_size: int, duration: float) -> None:
        """Record the processing of a batch."""
        measurement = self._measurement
        items_per_second = batch_size / duration if duration > 0 else 0

        batch_size_factor = 1 - math.exp(-0.15 * batch_size)
        items_per_second = items_per_second * batch_size_factor + measurement.items_per_second * (1 - batch_size_factor)

        time_since_update = time.time() - measurement.last_update
        factor = math.pow(1 - BATCH_DECAY_RATE, time_since_update)
        new_items_per_second = measurement.items_per_second * factor + items_per_second * (1 - factor)

        logger.debug(
            "Updating average items per second from {:.3f} to {:.3f} (Actual: {:.3f})",
            measurement.items_per_second,
            new_items_per_second,
            items_per_second,
        )

        measurement.last_update = time.time()
        measurement.items_per_second = new_items_per_second
        self._save_state()
