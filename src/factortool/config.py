# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Configuration for factorization tool."""

from __future__ import annotations

import os
import shutil
import sys

from pathlib import Path
from typing import Annotated, Literal, NamedTuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, field_validator, model_validator

from factortool.constants import FINAL_METHOD_NAMES, NFS_CADO_MIN_DIGITS, NFS_YAFU_MIN_DIGITS


class YafuPaths(NamedTuple):
    """Paths needed for a yafu invocation."""

    binary: Path
    work: Path
    ini: Path


class FinalMethods(NamedTuple):
    """Settings that decide which final factoring methods (SIQS or NFS) may be used."""

    max_siqs_digits: int
    use_nfs_cado: bool
    use_nfs_yafu: bool

    def for_digits(self, digits: int) -> tuple[str, ...]:
        """Determine the final factoring methods eligible for a composite with the given digit count.

        SIQS is used as a last resort if no method is otherwise eligible.

        Returns:
            tuple[str, ...]: The statistics keys of the eligible methods.
        """
        eligible = {
            "siqs": digits <= self.max_siqs_digits,
            "nfs_cado": self.use_nfs_cado and digits >= NFS_CADO_MIN_DIGITS,
            "nfs_yafu": self.use_nfs_yafu and digits >= NFS_YAFU_MIN_DIGITS,
        }

        return tuple(method for method in FINAL_METHOD_NAMES if eligible[method]) or ("siqs",)


ASCII_RANGE = range(0x20, 0x7F)

CooldownPeriod = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def _is_executable_file(path: Path) -> bool:
    """Check if the given path is an executable file.

    Returns:
        bool: True if the path is an executable file, or on Windows, becomes one once an omitted extension is added.
    """
    return (path.is_file() and os.access(path, os.X_OK)) or shutil.which(path) is not None


class Config(BaseModel):
    """Configuration for factorization tool."""

    model_config = ConfigDict(frozen=True)

    # Required settings
    backend: Literal["factordb", "mersenne_ca"]
    max_threads: PositiveInt
    yafu_path: Path

    # Optional settings with defaults
    assignment_state_path: Path = Path("assignment_state.json")
    batch_state_path: Path = Path("batch_state.json")
    cado_nfs_path: Path | None = None
    factordb_api_token: str = ""
    factordb_cooldown_period: CooldownPeriod = 1.0
    factoring_mode: Literal["standard", "yafu"] = "standard"
    gimps_login: str = ""
    max_siqs_digits: PositiveInt = 100
    mersenne_ca_cooldown_period: CooldownPeriod = 1.0
    pending_submissions_path: Path = Path("pending_submissions.jsonl")
    result_output_path: Path = Path("results")
    stats_path: Path = Path("stats.json")
    use_nfs_cado: bool = False
    use_nfs_yafu: bool = False
    user_agent: str = ""
    work_path: Path = Path("work")
    yafu_ini_path: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def warn_unknown_settings(cls, data: object) -> object:
        """Warn about unrecognized settings.

        Returns:
            object: The unmodified input data.
        """
        if isinstance(data, dict):
            for key in sorted(data.keys() - cls.model_fields.keys()):
                logger.warning(f"Ignoring unrecognized configuration setting: {key}")

        return data

    @model_validator(mode="after")
    def validate_backend_credentials(self) -> Config:
        """Require the credentials the selected backend require.

        Returns:
            Config: The validated configuration object.

        Raises:
            ValueError: If the mersenne.ca backend is selected without a GIMPS login.
        """
        if self.backend == "mersenne_ca" and not self.gimps_login:
            msg = "the mersenne_ca backend requires gimps_login"
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_cado_nfs_path(self) -> Config:
        """Require the path to CADO-NFS when it is enabled.

        Returns:
            Config: The validated configuration object.

        Raises:
            ValueError: If CADO-NFS is enabled without a path to it.
        """
        if self.use_nfs_cado and self.cado_nfs_path is None:
            msg = "use_nfs_cado requires cado_nfs_path"
            raise ValueError(msg)

        return self

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

    @property
    def final_methods(self) -> FinalMethods:
        """Settings that decide which final factoring methods may be used."""
        return FinalMethods(self.max_siqs_digits, self.use_nfs_cado, self.use_nfs_yafu)

    def find_tool_problems(self) -> list[str]:
        """Look for the external tools and files this configuration needs.

        Returns:
            list[str]: A description of each one that is missing or unusable.
        """
        problems: list[str] = []
        tools = {"yafu_path": self.yafu_path}

        if self.use_nfs_cado and self.cado_nfs_path is not None:
            tools["cado_nfs_path"] = self.cado_nfs_path

        for setting, path in sorted(tools.items()):
            if not _is_executable_file(path.absolute()):
                problems.append(f"{setting} ({path}) is not an executable file")

        if self.yafu_ini_path is not None and not self.yafu_ini_path.is_file():
            problems.append(f"yafu_ini_path ({self.yafu_ini_path}) is not a file")

        return problems


def read_config(path: Path) -> Config:
    """Read configuration from a JSON file.

    Returns:
        Config: The configuration object.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            config = Config.model_validate_json(f.read())
    except ValidationError as e:
        missing = [str(error["loc"][0]) for error in e.errors() if error["type"] == "missing" and error["loc"]]

        if missing:
            logger.error(f"Configuration file {path} is missing required setting(s): {', '.join(missing)}")

        if len(missing) < e.error_count():
            logger.error(f"Error while processing configuration file: {e}")

        sys.exit(1)

    return config
