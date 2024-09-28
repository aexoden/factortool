# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import sys

from pathlib import Path

from loguru import logger
from tap import Tap

from factortool.config import read_config
from factortool.engine import FactorEngine
from factortool.number import Number
from factortool.stats import FactoringStats


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
        config = read_config(args.config_path)
    except FileNotFoundError:
        logger.error("Configuration file not found")
        sys.exit(1)

    # Test with some random numbers for now.
    import random  # noqa: PLC0415
    test_numbers: list[Number] = []
    stats = FactoringStats(config.stats_path)

    for digits in range(60, 65):
        test_numbers.extend([Number(random.randint(10 ** (digits - 1), 10 ** digits), config, stats) for _ in range(2)])  # noqa: S311

    engine = FactorEngine(config)
    engine.run(test_numbers)

    for number in test_numbers:
        print(number.methods, number.composite_factors, number.prime_factors)
