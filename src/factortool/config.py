# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import sys

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError


@dataclass
class Config(BaseModel):
    factordb_username: str
    factordb_password: str


def read_config(path: Path) -> Config:
    try:
        with path.open("r", encoding="utf-8") as f:
            config = Config.model_validate_json(f.read())
    except ValidationError as e:
        print(e)
        sys.exit(1)

    return config
