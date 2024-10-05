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


def get_average_time(
        stats: FactoringStats,
        threads: int,
        digits: int,
        nfs_time: float,
        next_ecm_level: int,
        maximum_ecm_level: int
        ) -> float:
    if next_ecm_level > maximum_ecm_level:
        return nfs_time

    ecm_threads = min(threads, ECM_CURVES[next_ecm_level][0])
    ecm_count, ecm_time, ecm_p_factor = stats.get_ecm_stats(digits, next_ecm_level, ecm_threads)

    # It is a bug for any of the following to be violated, and it helps the static type checker.
    assert ecm_count > 0  # noqa: S101
    assert ecm_time is not None  # noqa: S101
    assert ecm_p_factor is not None  # noqa: S101

    # This level of ECM is done unconditionally.
    average_time = ecm_time

    # If no factor is found, we must recursively check the next level.
    average_time += (1 - ecm_p_factor) * get_average_time(stats, threads, digits, nfs_time, next_ecm_level + 1, maximum_ecm_level)

    return average_time


def main() -> None:
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

    nfs_count, nfs_time = stats.get_nfs_stats(args.digits, config.max_threads)

    if nfs_count == 0:
        logger.error("No NFS data present for this digit count.")
        sys.exit(2)

    # It would be a bug for nfs_time to be None if nfs_count is greater than zero.
    assert nfs_time is not None  # noqa: S101

    print(f"ECM Crossover Analysis for {args.digits} digits:")
    print()
    print(f"Average time for NFS is {nfs_time:0.3f}s")
    print()
    print("Stopping ECM after doing the given level averages:")

    for ecm_level in range(min_ecm_level, max_ecm_level + 1):
        ecm_threads = min(config.max_threads, ECM_CURVES[ecm_level][0])

        ecm_count, ecm_time, ecm_p_factor = stats.get_ecm_stats(args.digits, ecm_level, ecm_threads)

        # If there is no ECM data for this level, we've reached the end and can just exit.
        if ecm_count == 0:
            sys.exit(0)

        assert ecm_p_factor is not None  # noqa: S101

        average_time = get_average_time(stats, config.max_threads, args.digits, nfs_time, min_ecm_level, ecm_level)

        print(f'  {ecm_level:3}  {ecm_count:8}  {ecm_time:7.3f}s  {ecm_p_factor * 100:7.3f}%  {average_time:7.3f}s')
