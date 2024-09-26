# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import sys

from pathlib import Path

from loguru import logger

from factortool.config import read_config


def main() -> None:
    try:
        _config = read_config(Path("config.json"))
    except FileNotFoundError:
        logger.error("Configuration file not found")
        sys.exit(1)
