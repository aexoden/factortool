# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import math
import time

from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from factortool.util import safe_write

BATCH_DECAY_RATE = 2 / 1800
BATCH_RESET_AGE = 7200
MAX_BATCH_SIZE = 200


class BatchState(BaseModel):
    last_update: float
    min_digits: int
    skip_count: int
    items_per_second: float


class BatchController:
    def __init__(self, target_duration: float, min_digits: int, skip_count: int, state_path: Path) -> None:
        self._target_duration = target_duration
        self._state_path = state_path

        self._reset_state(min_digits, skip_count)
        self._load_state()

        if self._state.min_digits != min_digits or self._state.skip_count != skip_count:
            logger.info("Resetting batch size due to configuration change")
            self._reset_state(min_digits, skip_count)

        time_since_update = time.time() - self._state.last_update

        if time_since_update > BATCH_RESET_AGE:
            logger.info("Resetting batch size due to inactivity")
            self._reset_state(min_digits, skip_count)

    def _reset_state(self, min_digits: int, skip_count: int) -> None:
        self._state = BatchState(
            last_update=time.time(), min_digits=min_digits, skip_count=skip_count, items_per_second=0
        )

    def _load_state(self) -> None:
        if self._state_path.exists():
            self._state = BatchState.model_validate_json(self._state_path.read_text(encoding="utf-8"))

    def _save_state(self) -> None:
        safe_write(self._state_path, self._state.model_dump_json().encode("utf-8"))

    @property
    def batch_size(self) -> int:
        return min(MAX_BATCH_SIZE, max(1, int(self._state.items_per_second * self._target_duration)))

    def record_batch(self, batch_size: int, duration: float) -> None:
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
