# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import datetime
import sys

from pathlib import Path

from loguru import logger
from tap import Tap

from factortool.config import read_config
from factortool.engine import FactorEngine
from factortool.factordb import FactorDB
from factortool.number import Number, format_results
from factortool.stats import FactoringStats
from factortool.util import setup_logger


class Arguments(Tap):
    """Utility for factoring numbers using trial factoring, ECM and NFS methods"""

    config_path: Path = Path("config.json")  # Path to the JSON-formatted configuration file
    min_digits: int = 1  # Minimum number of digits fetched composite numbers should have
    batch_size: int = 50  # Number of composite numbers to work on at a time
    skip_count: int = 0  # Skip this many numbers when fetching from FactorDB (to hopefully avoid conflict)


def main() -> None:
    setup_logger()

    args = Arguments().parse_args()

    try:
        config = read_config(args.config_path)
    except FileNotFoundError:
        logger.error("Configuration file not found")
        sys.exit(1)

    stats = FactoringStats(config.stats_path)
    factordb = FactorDB(config, stats)
    engine = FactorEngine(config)

    numbers = factordb.fetch(args.min_digits, args.batch_size, args.skip_count)

    engine.run(sorted(numbers))

    factordb.submit(numbers)

    method_counts: dict[str, int] = {}
    failed_numbers: set[Number] = set()

    for number in numbers:
        for method in set(number.methods):
            if method not in method_counts:
                method_counts[method] = 0

            method_counts[method] += 1

        if not number.factored:
            failed_numbers.add(number)

    if len(failed_numbers) > 0:
        logger.warning(
            "{} numbers failed to factor: {}",
            len(failed_numbers),
            ", ".join(str(x.n) for x in sorted(failed_numbers)),
        )

    logger.info(
        "Factored {} numbers and there were {} failures",
        len(numbers) - len(failed_numbers),
        len(failed_numbers),
    )

    logger.info(
        "The following methods were used: {}",
        ", ".join(f"{method} ({count})" for method, count in method_counts.items()),
    )

    if not config.result_output_path.exists():
        config.result_output_path.mkdir(parents=True)

    output_filename = f"{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d-%H%M%S')}.txt"
    output_path = config.result_output_path.joinpath(output_filename)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(format_results(numbers) + "\n")
