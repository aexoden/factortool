# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the factoring statistics."""

from __future__ import annotations

import sys
import threading

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

import pytest

from factortool.constants import ECM_MIN_LEVEL
from factortool.stats import ECMCutoffs, FactoringStats

THREADS = 8
UPDATES = 100

# The digit count of the entries every thread shares.
DIGITS = 50

ECM_LEVEL = ECM_MIN_LEVEL

# Enough samples of an ECM level that no further levels are done just to collect data.
ECM_SETTLED_SAMPLES = 1024


def run_threads(workers: Sequence[Callable[[], None]], background: Sequence[Callable[[], None]] = ()) -> None:
    """Run each worker in its own thread, calling each background function repeatedly alongside them until they finish.

    Any exception raised in a thread fails the calling test.
    """
    errors: list[BaseException] = []
    done = threading.Event()

    def guard(function: Callable[[], None], *, repeat: bool) -> Callable[[], None]:
        def run() -> None:
            try:
                function()

                while repeat and not done.is_set():
                    function()
            except BaseException as e:  # ruff: ignore[blind-except]
                errors.append(e)

        return run

    worker_threads = [threading.Thread(target=guard(worker, repeat=False)) for worker in workers]
    background_threads = [threading.Thread(target=guard(function, repeat=True)) for function in background]

    # Switch threads as often as possible, so an unguarded update is all but certain to be interrupted.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    try:
        for thread in (*worker_threads, *background_threads):
            thread.start()

        for thread in worker_threads:
            thread.join()

        done.set()

        for thread in background_threads:
            thread.join()
    finally:
        done.set()
        sys.setswitchinterval(interval)

    assert errors == []


def make_ecm_stats(
    tmp_path: Path, final_time: float | None, levels: Sequence[tuple[float, int, int]]
) -> FactoringStats:
    """Build statistics for DIGITS from a final method time and each ECM level's time, run count and success count.

    Returns:
        FactoringStats: The statistics, with the ECM levels starting from the first.
    """
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)

    if final_time is not None:
        stats.update_final("siqs", DIGITS, 1, final_time)

    for level, (ecm_time, runs, successes) in enumerate(levels, ECM_MIN_LEVEL):
        for run in range(runs):
            stats.update_ecm(DIGITS, level, 1, ecm_time, success=run < successes)

    return stats


def test_ecm_cutoffs_use_an_initial_estimate_without_data(tmp_path: Path) -> None:
    """Test that the cutoff falls back to a third of the digit count until there is both ECM and final method data."""
    estimate = ECMCutoffs(None, DIGITS // 3)

    assert make_ecm_stats(tmp_path, None, []).get_ecm_cutoffs(DIGITS, 1, ["siqs"]) == estimate
    assert make_ecm_stats(tmp_path, 10.0, []).get_ecm_cutoffs(DIGITS, 1, ["siqs"]) == estimate
    assert make_ecm_stats(tmp_path, None, [(1.0, 4, 2)]).get_ecm_cutoffs(DIGITS, 1, ["siqs"]) == estimate


@pytest.mark.parametrize(("digits", "target"), [(2, 0), (5, 0), (6, 2), (8, 2), (9, 3)])
def test_initial_ecm_estimate_below_the_first_level_is_no_ecm(tmp_path: Path, digits: int, target: int) -> None:
    """Test that an initial estimate too low to reach the first ECM level is reported as doing no ECM."""
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)

    assert stats.get_ecm_cutoffs(digits, 1, ["siqs"]) == ECMCutoffs(None, target)


def test_ecm_cutoffs_choose_the_level_with_the_lowest_average_time(tmp_path: Path) -> None:
    """Test that ECM stops at the last level that lowers the estimated average time, once the data has settled."""
    # The first two levels each save more final factoring than they cost, and the third costs more than it saves.
    levels = [
        (1.0, ECM_SETTLED_SAMPLES, ECM_SETTLED_SAMPLES // 2),
        (1.0, ECM_SETTLED_SAMPLES, ECM_SETTLED_SAMPLES // 2),
        (9.0, ECM_SETTLED_SAMPLES, ECM_SETTLED_SAMPLES // 2),
    ]
    stats = make_ecm_stats(tmp_path, 10.0, levels)

    assert stats.get_ecm_cutoffs(DIGITS, 1, ["siqs"]) == ECMCutoffs(ECM_MIN_LEVEL + 1, ECM_MIN_LEVEL + 1)


def test_ecm_cutoffs_choose_no_ecm_when_none_is_worthwhile(tmp_path: Path) -> None:
    """Test that no ECM at all is done when even the first level costs more than the final factoring it saves."""
    stats = make_ecm_stats(tmp_path, 1.0, [(0.9, ECM_SETTLED_SAMPLES, ECM_SETTLED_SAMPLES // 2)])

    assert stats.get_ecm_cutoffs(DIGITS, 1, ["siqs"]) == ECMCutoffs(0, 0)
    assert stats.get_ecm_average_time(DIGITS, 0, 1, ["siqs"]) == (0, 1.0)


def test_ecm_cutoffs_keep_collecting_data_when_no_ecm_is_optimal(tmp_path: Path) -> None:
    """Test that ECM continues past an optimum of none for as long as the first level has few samples."""
    stats = make_ecm_stats(tmp_path, 1.0, [(0.9, 4, 0)])

    # Four samples allow eight extra levels beyond doing no ECM.
    assert stats.get_ecm_cutoffs(DIGITS, 1, ["siqs"]) == ECMCutoffs(0, ECM_MIN_LEVEL + 7)


def test_ecm_cutoffs_stop_before_a_level_slower_than_the_final_method(tmp_path: Path) -> None:
    """Test that a level slower than the final method is not run to collect data, even as the very first level."""
    stats = make_ecm_stats(tmp_path, 1.0, [(2.0, 4, 2)])
    assert stats.get_ecm_cutoffs(DIGITS, 1, ["siqs"]) == ECMCutoffs(0, 0)

    stats = make_ecm_stats(tmp_path, 10.0, [(1.0, 4, 2), (20.0, 4, 2)])
    assert stats.get_ecm_cutoffs(DIGITS, 1, ["siqs"]) == ECMCutoffs(ECM_MIN_LEVEL, ECM_MIN_LEVEL)


def test_concurrent_updates_are_all_recorded_and_saved(tmp_path: Path) -> None:
    """Test that updates from several threads are neither lost nor able to break a save happening alongside them."""
    stats = FactoringStats(tmp_path / "stats.json", min_write_interval=0.0)

    def update(thread: int) -> None:
        for i in range(UPDATES):
            # A shared entry exposes lost updates, and new entries change the data while it is being saved.
            stats.update_probability(DIGITS, "rho", 1, 1.0, success=True)
            stats.update_probability(1000 + thread * UPDATES + i, "pm1", 1, 1.0, success=False)

    run_threads([partial(update, thread) for thread in range(THREADS)])

    assert stats.get_probability_stats(DIGITS, "rho", 1) == (THREADS * UPDATES, 1.0, 1.0)

    reloaded = FactoringStats(tmp_path / "stats.json", read_only=True)
    assert reloaded.get_probability_stats(DIGITS, "rho", 1) == (THREADS * UPDATES, 1.0, 1.0)


def test_reads_and_explicit_saves_are_safe_alongside_updates(tmp_path: Path) -> None:
    """Test that reads and explicit saves racing updates of every kind see consistent data and lose none of it."""
    # Only the explicit saves write the file, as the interval never elapses after the first update.
    stats = FactoringStats(tmp_path / "stats.json", min_write_interval=3600.0)
    total = THREADS * UPDATES

    def update(thread: int) -> None:
        for i in range(UPDATES):
            unique = 1000 + thread * UPDATES + i

            # Shared entries expose lost updates, and new entries change the data while it is being read and saved.
            stats.update_probability(DIGITS, "rho", 1, 1.0, success=True)
            stats.update_final("siqs", DIGITS, 1, 1.0)
            stats.update_ecm(DIGITS, ECM_LEVEL, 1, 1.0, success=True)
            stats.update_final("siqs", unique, 1, 1.0)
            stats.update_ecm(unique, ECM_LEVEL, 1, 1.0, success=False)

    def read() -> None:
        # Every update adds one run taking one second, so an average is only ever anything else mid-update.
        previous = (0, 0, 0)

        while previous != (total, total, total):
            rho_count, rho_time, rho_p_factor = stats.get_probability_stats(DIGITS, "rho", 1)
            final_count, final_time = stats.get_final_stats("siqs", DIGITS, 1)
            ecm_count, ecm_time, ecm_p_factor = stats.get_ecm_stats(DIGITS, ECM_LEVEL, 1)
            counts = (rho_count, final_count, ecm_count)

            assert {rho_time, rho_p_factor, final_time, ecm_time, ecm_p_factor} <= {None, 1.0}
            assert all(count >= earlier for count, earlier in zip(counts, previous, strict=True))
            previous = counts

            # These combine several reads, and so take the lock more than once.
            assert stats.get_final_method(DIGITS, 1, ["siqs"]) == "siqs"
            stats.get_ecm_cutoffs(DIGITS, 1, ["siqs"])

    run_threads([partial(update, thread) for thread in range(THREADS)], background=[read, read, stats.save_data])

    stats.save_data()

    for source in (stats, FactoringStats(tmp_path / "stats.json", read_only=True)):
        assert source.get_probability_stats(DIGITS, "rho", 1) == (total, 1.0, 1.0)
        assert source.get_final_stats("siqs", DIGITS, 1) == (total, 1.0)
        assert source.get_ecm_stats(DIGITS, ECM_LEVEL, 1) == (total, 1.0, 1.0)
        assert source.get_final_stats("siqs", 1000 + total - 1, 1) == (1, 1.0)
        assert source.get_ecm_stats(1000 + total - 1, ECM_LEVEL, 1) == (1, 1.0, 0.0)
