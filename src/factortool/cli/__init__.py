# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import sys

from pathlib import Path

from loguru import logger

from factortool.config import read_config


def setup_logger() -> None:
    logger.remove(0)

    logger_format = " | ".join([
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>",
        "<magenta>{elapsed}</magenta>",
        "<level>{level: <8}</level>",
        "<level>{message}</level>",
    ])

    logger.add(sys.stdout, format=logger_format)


def main() -> None:
    setup_logger()

    try:
        _config = read_config(Path("config.json"))
    except FileNotFoundError:
        logger.error("Configuration file not found")
        sys.exit(1)
