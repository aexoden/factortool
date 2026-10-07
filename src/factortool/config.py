# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Configuration for factorization tool."""

from __future__ import annotations

import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from loguru import logger
from pydantic import BaseModel, ValidationError, field_validator


class YafuPaths(NamedTuple):
    """Paths needed for a yafu invocation."""

    binary: Path
    work: Path
    ini: Path


ASCII_RANGE = range(0x20, 0x7F)


@dataclass
class Config(BaseModel):
    """Configuration for factorization tool."""

    assignment_state_path: Path
    backend: Literal["factordb", "mersenne_ca"]
    batch_state_path: Path
    cado_nfs_path: Path
    factordb_api_token: str
    factordb_cooldown_period: float
    factoring_mode: Literal["standard", "yafu"]
    gimps_login: str
    max_siqs_digits: int
    max_threads: int
    mersenne_ca_cooldown_period: float
    result_output_path: Path
    stats_path: Path
    user_agent: str
    work_path: Path
    yafu_path: Path
    yafu_ini_path: Path | None

    @field_validator("user_agent")
    @classmethod
    def validate_user_agent(cls, value: str) -> str:
        """Require an empty override or printable ASCII header.

        Returns:
            str: The validated User-Agent string.

        Raises:
            ValueError: If the User-Agent is not empty and contains non-printable ASCII characters or leading/trailing
                whitespace.
        """
        if value and (value != value.strip() or any(ord(char) not in ASCII_RANGE for char in value)):
            msg = "user_agent must contain printable ASCII only, with no leading or trailing whitespace"
            raise ValueError(msg)

        return value

    @property
    def yafu_paths(self) -> YafuPaths:
        """Paths needed for a yafu invocation, with yafu.ini defaulting to the one beside the binary."""
        binary = self.yafu_path.absolute()
        ini = self.yafu_ini_path.absolute() if self.yafu_ini_path is not None else (binary.parent / "yafu.ini")
        return YafuPaths(binary=binary, work=self.work_path, ini=ini)


def read_config(path: Path) -> Config:
    """Read configuration from a JSON file.

    Returns:
        Config: The configuration object.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            config = Config.model_validate_json(f.read())
    except ValidationError as e:
        logger.error(f"Error while processing configuration file: {e}")
        sys.exit(1)

    return config
