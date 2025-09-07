# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import sys

from pathlib import Path

from loguru import logger
from tap import Tap

from factortool.config import read_config
from factortool.constants import ECM_CURVES
from factortool.stats import FactoringStats
from factortool.util import setup_logger


class Arguments(Tap):
    """Utility for analyzing statistics gathered by factortool"""

    config_path: Path = Path("config.json")  # Path to the JSON-formatted configuration file
    digits: int  # Digits to analyze


def main() -> None:  # noqa: PLR0914
    setup_logger()

    args = Arguments().parse_args()

    try:
        config = read_config(args.config_path)
    except FileNotFoundError:
        logger.error("Configuration file not found")
        sys.exit(1)

    stats = FactoringStats(config.stats_path, read_only=True)

    min_ecm_level = min(ECM_CURVES.keys())
    max_ecm_level = max(ECM_CURVES.keys())

    tf_count, tf_time, tf_p_factor = stats.get_probability_stats(args.digits, "tf", 1)

    if tf_count == 0:
        print(f"No trial factoring data present for {args.digits} digits.")
    else:
        assert tf_p_factor is not None  # noqa: S101
        print(
            f"Average time for {tf_count} trial factoring runs with a {tf_p_factor * 100:0.3f}% success rate is {tf_time:0.3f}s"  # noqa: E501
        )

    rho_count, rho_time, rho_p_factor = stats.get_probability_stats(args.digits, "rho", 1)

    if rho_count == 0:
        print(f"No rho data present for {args.digits} digits.")
    else:
        assert rho_p_factor is not None  # noqa: S101
        print(
            f"Average time for {rho_count} rho runs with a {rho_p_factor * 100:0.3f}% success rate is {rho_time:0.3f}s"
        )

    pm1_count, pm1_time, pm1_p_factor = stats.get_probability_stats(args.digits, "pm1", 1)

    if pm1_count == 0:
        print(f"No P-1 data present for {args.digits} digits.")
    else:
        assert pm1_p_factor is not None  # noqa: S101
        print(
            f"Average time for {pm1_count} P-1 runs with a {pm1_p_factor * 100:0.3f}% success rate is {pm1_time:0.3f}s"
        )

    print()

    siqs_count, siqs_time = stats.get_siqs_stats(args.digits, config.max_threads)
    nfs_count, nfs_time = stats.get_nfs_stats(args.digits, config.max_threads)
    yafu_count, yafu_time = stats.get_yafu_stats(args.digits, config.max_threads)

    if siqs_count == 0 and nfs_count == 0:
        logger.error("No SIQS or NFS data present for this digit count.")
        sys.exit(2)

    print(f"ECM Crossover Analysis for {args.digits} digits:")
    print()

    if siqs_time is not None:
        print(f"Average time for SIQS is {siqs_time:0.3f}s")
        print()

    if nfs_time is not None:
        print(f"Average time for NFS is {nfs_time:0.3f}s")
        print()

    if yafu_time is not None:
        print(f"Average time for YAFU (direct) is {yafu_time:0.3f}s")
        print()

    print("Stopping ECM after doing the given level averages:")

    for ecm_level in range(min_ecm_level, max_ecm_level + 1):
        ecm_count, ecm_time, ecm_p_factor = stats.get_ecm_stats(args.digits, ecm_level, config.max_threads)

        # If there is no ECM data for this level, we've reached the end and can just exit.
        if ecm_count == 0:
            sys.exit(0)

        assert ecm_p_factor is not None  # noqa: S101

        _, average_time = stats.get_average_time(args.digits, ecm_level, config.max_threads)

        print(f"  {ecm_level:3}  {ecm_count:8}  {ecm_time:7.3f}s  {ecm_p_factor * 100:7.3f}%  {average_time:7.3f}s")
