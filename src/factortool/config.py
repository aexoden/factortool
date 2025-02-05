# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import sys

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ValidationError


@dataclass
class Config(BaseModel):
    batch_state_path: Path
    cado_nfs_path: Path
    factordb_cooldown_period: float
    factordb_response_path: Path
    factordb_session_path: Path
    factordb_username: str
    factordb_password: str
    max_siqs_digits: int
    max_threads: int
    result_output_path: Path
    stats_path: Path
    yafu_path: Path


def read_config(path: Path) -> Config:
    try:
        with path.open("r", encoding="utf-8") as f:
            config = Config.model_validate_json(f.read())
    except ValidationError as e:
        logger.error(f"Error while processing configuration file: {e}")
        sys.exit(1)

    return config
