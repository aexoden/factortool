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

# The fraction of the measured numbers and seconds forgotten per second, so the throughput follows changes in the work.
BATCH_DECAY_RATE = 2 / 1800

# Seconds after which a throughput that has not been updated is discarded.
BATCH_RESET_AGE = 7200

# How many times the largest batch measured so far (in numbers factored) the next batch may be. A first measurement
# can be far off (a number with a small factor finishes much faster than the rest), and too large a batch costs much
# more than too small a one, so the batch size grows step by step while the measurement improves.
BATCH_MAX_GROWTH = 2


class BatchKey(BaseModel):
    """The work selection a measured throughput applies to."""

    backend: str = "factordb"
    factoring_mode: str
    max_threads: int
    min_digits: int
    max_digits: int | None = None
    skip_count: int = 0


class BatchMeasurement(BaseModel):
    """The throughput measured for one work selection.

    The throughput is the ratio of the numbers factored to the seconds spent, both decayed by BATCH_DECAY_RATE, so every
    number factored counts the same, whichever batch it was in.
    """

    key: BatchKey
    last_update: float
    numbers: float
    seconds: float
    largest_batch: int

    @property
    def items_per_second(self) -> float:
        """Numbers factored per second."""
        return self.numbers / self.seconds if self.seconds > 0 else 0.0


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
            self._measurement = BatchMeasurement(key=key, last_update=now, numbers=0, seconds=0, largest_batch=0)

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
        """Current batch size: the number expected to take the target duration, limited by BATCH_MAX_GROWTH."""
        measurement = self._measurement
        expected = int(measurement.items_per_second * self._target_duration)

        return max(1, min(expected, BATCH_MAX_GROWTH * measurement.largest_batch))

    def record_batch(self, batch_size: int, duration: float) -> None:
        """Record the processing of a batch.

        Args:
            batch_size: How many numbers were factored.
            duration: How long the batch took, in seconds.
        """
        measurement = self._measurement
        previous = measurement.items_per_second

        now = time.time()
        factor = math.pow(1 - BATCH_DECAY_RATE, max(0.0, now - measurement.last_update))

        measurement.numbers = measurement.numbers * factor + batch_size
        measurement.seconds = measurement.seconds * factor + max(0.0, duration)
        measurement.largest_batch = max(measurement.largest_batch, batch_size)
        measurement.last_update = now

        logger.debug(
            "Updating average items per second from {:.3f} to {:.3f} (Actual: {:.3f})",
            previous,
            measurement.items_per_second,
            batch_size / duration if duration > 0 else 0,
        )

        self._save_state()
