# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024-2026 Jason Lynch <jason@aexoden.com>
# ruff:file-ignore[print] (print() used for output)
"""Utility for analyzing factorization statistics."""

from __future__ import annotations

import sys

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from loguru import logger

from factortool.config import Config, read_config
from factortool.constants import (
    ECM_MIN_LEVEL,
    FINAL_METHOD_NAMES,
    NFS_CADO_MIN_DIGITS,
    NFS_YAFU_MIN_DIGITS,
)
from factortool.stats import ECMCutoffs, FactoringStats, InvalidStatsError
from factortool.util import ArgumentParser, setup_logger


class Arguments(ArgumentParser):
    """Utility for analyzing statistics gathered by factortool."""

    config_path: Path = Path("config.json")  # Path to the JSON-formatted configuration file
    digits: int  # Digits to analyze


def _ineligible_reason(method: str, config: Config) -> str:
    """Describe why an ineligible final factoring method won't be used.

    Returns:
        str: A short description of the reason.
    """
    if method == "siqs":
        return f"above max_siqs_digits of {config.max_siqs_digits}"

    enabled, setting, min_digits = {
        "nfs_cado": (config.use_nfs_cado, "use_nfs_cado", NFS_CADO_MIN_DIGITS),
        "nfs_yafu": (config.use_nfs_yafu, "use_nfs_yafu", NFS_YAFU_MIN_DIGITS),
    }[method]

    if not enabled:
        return f"{setting} is disabled"

    return f"below minimum of {min_digits} digits"


def _format_time(seconds: float | None) -> str:
    """Format a time for a table column, or as unavailable if there is none.

    Returns:
        str: The formatted time.
    """
    return f"{seconds:7.3f}s" if seconds is not None else f"{'N/A':>8}"


def _print_ecm_table(
    stats: FactoringStats, digits: int, threads: int, final_methods: Sequence[str], cutoffs: ECMCutoffs
) -> None:
    """Print the estimated average times from stopping ECM after each level with data."""
    levels = stats.get_ecm_levels(digits, threads)

    # A level of zero is doing no ECM at all. The cutoffs are shown even without data, as they may be beyond it.
    shown_levels = {0, *levels, cutoffs.target}

    if cutoffs.optimal is not None:
        shown_levels.add(cutoffs.optimal)

    print("Stopping ECM after doing the given level averages:")
    print(f"  {'Lvl':>4}  {'Runs':>8}  {'ECM time':>8}  {'P(fact)':>8}  {'From ECM':>8}  {'Overall':>8}")

    previous_level = 0

    for ecm_level in sorted(shown_levels):
        # There can be gaps, as cofactors start ECM at whichever level found them.
        if ecm_level > max(previous_level + 1, ECM_MIN_LEVEL):
            print(f"  {'...':>4}")

        previous_level = ecm_level

        ecm_count, ecm_time, ecm_p_factor = stats.get_ecm_stats(digits, ecm_level, threads)
        _, ecm_average_time = stats.get_ecm_average_time(digits, ecm_level, threads, final_methods)
        _, average_time = stats.get_average_time(digits, ecm_level, threads, final_methods)

        markers = []

        if ecm_level == cutoffs.optimal:
            markers.append("optimal")

        if ecm_level == cutoffs.target:
            markers.append("current")

        marker_str = f"  <- {', '.join(markers)}" if markers else ""

        if ecm_level == 0:
            ecm_str = f"{'none':>4}  {'-':>8}  {'-':>8}  {'-':>8}"
        elif ecm_p_factor is None:
            ecm_str = f"{ecm_level:4}  {ecm_count:8}  {'-':>8}  {'-':>8}"
        else:
            ecm_str = f"{ecm_level:4}  {ecm_count:8}  {_format_time(ecm_time)}  {ecm_p_factor * 100:7.3f}%"

        print(f"  {ecm_str}  {_format_time(ecm_average_time)}  {_format_time(average_time)}{marker_str}")

    if levels and levels[0] > ECM_MIN_LEVEL:
        print()
        print(f"There is no ECM data below level {levels[0]}. The From ECM times assume the levels before it are done.")
        print("The Overall times can't be estimated without data for those levels.")


def main() -> None:  # ruff:ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements]
    """Analyze factorization statistics."""
    setup_logger()

    args = Arguments().parse_args()

    if args.digits < 1:
        logger.error("--digits ({}) must be at least 1", args.digits)
        sys.exit(1)

    try:
        config = read_config(args.config_path)
    except FileNotFoundError:
        logger.error("Configuration file not found")
        sys.exit(1)

    try:
        stats = FactoringStats(config.stats_path, read_only=True)
    except InvalidStatsError as e:
        logger.error("{}", e)
        sys.exit(1)

    tf_count, tf_time, tf_p_factor = stats.get_probability_stats(args.digits, "tf", 1)

    if tf_count == 0:
        print(f"No trial factoring data present for {args.digits} digits.")
    else:
        assert tf_p_factor is not None  # ruff:ignore[assert]
        print(
            f"Average time for {tf_count} trial factoring runs with a"
            f" {tf_p_factor * 100:0.3f}% success rate is {tf_time:0.3f}s"
        )

    rho_count, rho_time, rho_p_factor = stats.get_probability_stats(args.digits, "rho", 1)

    if rho_count == 0:
        print(f"No rho data present for {args.digits} digits.")
    else:
        assert rho_p_factor is not None  # ruff:ignore[assert]
        print(
            f"Average time for {rho_count} rho runs with a {rho_p_factor * 100:0.3f}% success rate is {rho_time:0.3f}s"
        )

    pm1_count, pm1_time, pm1_p_factor = stats.get_probability_stats(args.digits, "pm1", 1)

    if pm1_count == 0:
        print(f"No P-1 data present for {args.digits} digits.")
    else:
        assert pm1_p_factor is not None  # ruff:ignore[assert]
        print(
            f"Average time for {pm1_count} P-1 runs with a {pm1_p_factor * 100:0.3f}% success rate is {pm1_time:0.3f}s"
        )

    print()

    final_methods = config.final_methods.for_digits(args.digits)
    final_stats = {
        method: stats.get_final_stats(method, args.digits, config.max_threads) for method in FINAL_METHOD_NAMES
    }
    yafu_count, yafu_time = stats.get_yafu_stats(args.digits, config.max_threads)

    if yafu_count == 0 and all(count == 0 for count, _ in final_stats.values()):
        logger.error("No YAFU, SIQS or NFS data present for this digit count.")
        sys.exit(2)

    print(f"ECM Crossover Analysis for {args.digits} digits:")
    print()

    selected_method = stats.get_final_method(args.digits, config.max_threads, final_methods)

    print("Final factoring methods:")
    print(f"  {'Method':<8}  {'Runs':>8}  {'Avg time':>9}  Status")

    for method, method_name in FINAL_METHOD_NAMES.items():
        final_count, final_time = final_stats[method]
        final_time_str = f"{final_time:8.3f}s" if final_time is not None else f"{'N/A':>9}"

        if method == selected_method:
            status = "selected" if final_time is not None else "selected (to collect data)"
        elif method in final_methods:
            status = "eligible"
        else:
            status = f"not used ({_ineligible_reason(method, config)})"

        print(f"  {method_name:<8}  {final_count:8}  {final_time_str}  {status}")

    print()

    if yafu_time is not None:
        print(f"Average time for YAFU (direct) is {yafu_time:0.3f}s")
        print()

    cutoffs = stats.get_ecm_cutoffs(args.digits, config.max_threads, final_methods)
    optimal_ecm_level, current_ecm_level = cutoffs

    _print_ecm_table(stats, args.digits, config.max_threads, final_methods, cutoffs)

    print()

    if optimal_ecm_level is None:
        print("Optimal ECM cutoff: N/A (insufficient data)")
    else:
        print(f"Optimal ECM cutoff: {optimal_ecm_level or 'none'}")

    if optimal_ecm_level is None:
        extra_text = " (initial estimate; insufficient data)"
    elif current_ecm_level > optimal_ecm_level:
        extra_text = " (extended to gather data)"
    else:
        extra_text = ""

    print(f"Current ECM cutoff: {current_ecm_level or 'none'}{extra_text}")
