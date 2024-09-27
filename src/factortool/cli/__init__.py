# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import sys

from pathlib import Path

from loguru import logger
from tap import Tap

from factortool.config import read_config
from factortool.factor import FactorEngine


class Arguments(Tap):
    """Utility for factoring numbers using trial factoring, ECM and NFS methods"""
    config_path: Path = Path("config.json")  # Path to the JSON-formatted configuration file
    min_digits: int = 1  # Minimum number of digits fetched composite numbers should have
    batch_size: int = 50  # Number of composite numbers to work on at a time
    skip_count: int = 0  # Skip this many numbers when fetching from FactorDB (to hopefully avoid conflict)


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

    args = Arguments().parse_args()

    try:
        _config = read_config(args.config_path)
    except FileNotFoundError:
        logger.error("Configuration file not found")
        sys.exit(1)

    engine = FactorEngine()
    engine.run([])
