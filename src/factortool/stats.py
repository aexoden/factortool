# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Factoring statistics data models."""

from __future__ import annotations

import json
import math
import threading
import time

from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from factortool.constants import ECM_MIN_LEVEL, ECM_P_FACTOR_DECAY, ECM_P_FACTOR_DEFAULT
from factortool.util import safe_write


class FinalRunData(BaseModel):
    """Time and run count data for final factorization runs (SIQS, CADO-NFS, YAFU NFS)."""

    total_time: float = Field(default=0.0, description="Total time spent on final factorization of this type")
    run_count: int = Field(default=0, description="Number of final factorization runs of this type")


class FinalDigitData(BaseModel):
    """Data for final factorization runs (SIQS, CADO-NFS, YAFU NFS) for a given digit count, grouped by thread count."""

    thread_data: dict[int, FinalRunData] = Field(
        default_factory=dict[int, FinalRunData], description="Final data for each thread count"
    )


class ProbabilityRunData(BaseModel):
    """Time, run count, and success count data for probabilistic factorization runs (TF, Rho, P-1, ECM)."""

    total_time: float = Field(default=0.0, description="Total time spent on factorization")
    run_count: int = Field(default=0, description="Number of factorization runs")
    success_count: int = Field(default=0, description="Number of successful factorizations")


class ECMLevelData(BaseModel):
    """Data for ECM runs at a given level, grouped by thread count."""

    thread_data: dict[int, ProbabilityRunData] = Field(
        default_factory=dict[int, ProbabilityRunData], description="ECM data for each thread count"
    )


class ECMDigitData(BaseModel):
    """Data for ECM runs for a given digit count, grouped by ECM level."""

    level_data: dict[int, ECMLevelData] = Field(
        default_factory=dict[int, ECMLevelData], description="ECM data for each level"
    )


class ProbabilityDigitData(BaseModel):
    """Data for probabilistic factorization runs (TF, Rho, P-1) for a given digit count, grouped by thread count."""

    thread_data: dict[int, ProbabilityRunData] = Field(
        default_factory=dict[int, ProbabilityRunData], description="Factoring data for each thread count"
    )


class InvalidStatsError(Exception):
    """Exception raised for invalid or incompatible statistics data."""


class ECMCutoffs(NamedTuple):
    """The last ECM levels worth doing for a digit count, where zero means no ECM is worthwhile."""

    # Level with the lowest estimated average time, or None if there is insufficient data.
    optimal: int | None

    # Level to actually stop at, which may be higher to collect additional data.
    target: int


class FactoringData(BaseModel):
    """All factoring statistics data."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]

    tf: dict[int, ProbabilityDigitData] = Field(
        default_factory=dict[int, ProbabilityDigitData], description="Trial factoring data for each digit count"
    )
    rho: dict[int, ProbabilityDigitData] = Field(
        default_factory=dict[int, ProbabilityDigitData], description="Rho data for each digit count"
    )
    pm1: dict[int, ProbabilityDigitData] = Field(
        default_factory=dict[int, ProbabilityDigitData], description="P-1 data for each digit count"
    )
    yafu: dict[int, ProbabilityDigitData] = Field(
        default_factory=dict[int, ProbabilityDigitData], description="YAFU data for each digit count"
    )
    ecm: dict[int, ECMDigitData] = Field(
        default_factory=dict[int, ECMDigitData], description="ECM data for each digit count"
    )
    siqs: dict[int, FinalDigitData] = Field(
        default_factory=dict[int, FinalDigitData], description="SIQS data for each digit count"
    )
    nfs_cado: dict[int, FinalDigitData] = Field(
        default_factory=dict[int, FinalDigitData], description="CADO-NFS data for each digit count"
    )
    nfs_yafu: dict[int, FinalDigitData] = Field(
        default_factory=dict[int, FinalDigitData], description="YAFU NFS data for each digit count"
    )


class FactoringStats:
    """Class for managing factoring statistics data."""

    _path: Path
    _data: FactoringData
    _min_write_interval: float
    _last_write_time: int
    _data_changed: bool
    _read_only: bool
    _lock: threading.Lock

    def __init__(self, path: Path, *, min_write_interval: float = 5.0, read_only: bool = False) -> None:
        """Initialize the factoring statistics manager."""
        self._path: Path = path
        self._min_write_interval = int(min_write_interval * 1_000_000_000)
        self._last_write_time = 0
        self._data_changed = False
        self._read_only = read_only
        self._lock = threading.Lock()

        self._load_data()

    def _load_data(self) -> None:
        if not self._path.exists():
            self._data = FactoringData(schema_version=1)
            return

        try:
            self._data = FactoringData.model_validate_json(self._path.read_text(encoding="utf-8"))
        except ValidationError as e:
            msg = (
                f"Statistics cache {self._path} is invalid or incompatible. "
                "Rename or remove it to start with fresh statistics. "
                "The existing file has not been changed."
            )
            raise InvalidStatsError(msg) from e

    def _save_data(self, *, force: bool = False) -> None:
        if self._read_only:
            return

        current_time = time.monotonic_ns()

        if not force and (current_time - self._last_write_time < self._min_write_interval):
            return

        if not force and not self._data_changed:
            return

        # Work around pydantic not offering a way to sort keys.
        # data = self._data.model_dump_json(indent=2, so)
        model_dict = self._data.model_dump()
        data = json.dumps(model_dict, sort_keys=True, indent=2).encode("utf-8")
        safe_write(self._path, data)

        self._last_write_time = current_time
        self._data_changed = False

    def save_data(self) -> None:
        """Force saving the factoring statistics data."""
        with self._lock:
            self._save_data(force=True)

    def update_probability(
        self, digits: int, method: str, threads: int, execution_time: float, *, success: bool
    ) -> None:
        """Update probabilistic factorization data."""
        with self._lock:
            data = getattr(self._data, method)

            if digits not in data:
                data[digits] = ProbabilityDigitData()

            if threads not in data[digits].thread_data:
                data[digits].thread_data[threads] = ProbabilityRunData()

            run_data = data[digits].thread_data[threads]

            run_data.total_time += execution_time
            run_data.run_count += 1

            if success:
                run_data.success_count += 1

            self._data_changed = True
            self._save_data()

    def update_final(self, method: str, digits: int, threads: int, execution_time: float) -> None:
        """Update final factorization data for the method with the given statistics key."""
        with self._lock:
            data = getattr(self._data, method)

            if digits not in data:
                data[digits] = FinalDigitData()

            if threads not in data[digits].thread_data:
                data[digits].thread_data[threads] = FinalRunData()

            run_data = data[digits].thread_data[threads]

            run_data.total_time += execution_time
            run_data.run_count += 1

            self._data_changed = True
            self._save_data()

    def update_ecm(self, digits: int, ecm_level: int, threads: int, execution_time: float, *, success: bool) -> None:
        """Update ECM factorization data."""
        with self._lock:
            if digits not in self._data.ecm:
                self._data.ecm[digits] = ECMDigitData()

            if ecm_level not in self._data.ecm[digits].level_data:
                self._data.ecm[digits].level_data[ecm_level] = ECMLevelData()

            if threads not in self._data.ecm[digits].level_data[ecm_level].thread_data:
                self._data.ecm[digits].level_data[ecm_level].thread_data[threads] = ProbabilityRunData()

            run_data = self._data.ecm[digits].level_data[ecm_level].thread_data[threads]

            run_data.total_time += execution_time
            run_data.run_count += 1

            if success:
                run_data.success_count += 1

            self._data_changed = True
            self._save_data()

    def get_final_stats(self, method: str, digits: int, threads: int) -> tuple[int, float | None]:
        """Get final factorization statistics for the method with the given statistics key.

        Returns:
            A tuple containing the number of runs and the average time per run, or None if no data is available.
        """
        with self._lock:
            data = getattr(self._data, method)

            if digits in data and threads in data[digits].thread_data:
                run_data = data[digits].thread_data[threads]

                if run_data.run_count > 0:
                    return (
                        run_data.run_count,
                        run_data.total_time / run_data.run_count,
                    )

            return (0, None)

    def get_final_method(self, digits: int, threads: int, methods: Sequence[str]) -> str:
        """Choose the final factoring method for a composite with the given digit count.

        A method with no data is chosen first, so that data is collected for it. Otherwise, the fastest is chosen.

        Returns:
            str: The statistics key of the chosen method.
        """
        times: dict[str, float] = {}

        for method in methods:
            _, final_time = self.get_final_stats(method, digits, threads)

            if final_time is None:
                return method

            times[method] = final_time

        return min(times, key=lambda method: times[method])

    def get_final_time(self, digits: int, threads: int, methods: Sequence[str]) -> float | None:
        """Get the average time of the fastest final factoring method with data.

        Returns:
            float | None: The average time of the fastest method, or None if none of the methods have data.
        """
        times = [self.get_final_stats(method, digits, threads)[1] for method in methods]

        return min((x for x in times if x is not None), default=None)

    def get_yafu_stats(self, digits: int, threads: int) -> tuple[int, float | None]:
        """Get YAFU factorization statistics.

        Returns:
            A tuple containing the number of YAFU runs and the average time per run, or None if no data is available.
        """
        with self._lock:
            if digits in self._data.yafu and threads in self._data.yafu[digits].thread_data:
                run_data = self._data.yafu[digits].thread_data[threads]

                if run_data.run_count > 0:
                    return (
                        run_data.run_count,
                        run_data.total_time / run_data.run_count,
                    )

            return (0, None)

    def get_probability_stats(self, digits: int, method: str, threads: int) -> tuple[int, float | None, float | None]:
        """Get probabilistic factorization statistics.

        Returns:
            A tuple containing the number of runs, the average time per run, and the success probability,
            or None values if no data is available.
        """
        with self._lock:
            data = getattr(self._data, method)

            if digits not in data:
                return (0, None, None)

            if threads not in data[digits].thread_data:
                return (0, None, None)

            run_data = data[digits].thread_data[threads]

            if run_data.run_count > 0:
                return (
                    run_data.run_count,
                    run_data.total_time / run_data.run_count,
                    run_data.success_count / run_data.run_count,
                )

            return (0, None, None)

    def get_ecm_stats(self, digits: int, ecm_level: int, threads: int) -> tuple[int, float | None, float | None]:
        """Get ECM factorization statistics.

        Returns:
            A tuple containing the number of ECM runs, the average time per run, and the success probability,
            or None values if no data is available.
        """
        with self._lock:
            if digits not in self._data.ecm:
                return (0, None, None)

            if ecm_level not in self._data.ecm[digits].level_data:
                return (0, None, None)

            if threads not in self._data.ecm[digits].level_data[ecm_level].thread_data:
                return (0, None, None)

            run_data = self._data.ecm[digits].level_data[ecm_level].thread_data[threads]

            if run_data.run_count > 0:
                return (
                    run_data.run_count,
                    run_data.total_time / run_data.run_count,
                    run_data.success_count / run_data.run_count,
                )

            return (0, None, None)

    def get_ecm_levels(self, digits: int, threads: int) -> list[int]:
        """Get the ECM levels with data for the given digit count, which need not be consecutive.

        A number that is a cofactor of a larger one starts ECM at whichever level found it, so there may be no data for
        the levels before that.

        Returns:
            list[int]: The levels with at least one run, in ascending order.
        """
        with self._lock:
            if digits not in self._data.ecm:
                return []

            return sorted(
                level
                for level, level_data in self._data.ecm[digits].level_data.items()
                if threads in level_data.thread_data and level_data.thread_data[threads].run_count > 0
            )

    def get_average_time(
        self, digits: int, maximum_ecm_level: int, threads: int, methods: Sequence[str]
    ) -> tuple[int, float | None]:
        """Estimate the average time to factor a number with the given digit count, starting from trial factoring.

        Returns:
            A tuple containing the estimated number of ECM runs and the average time to factor the number,
            or None if insufficient data is available.
        """
        # A number starting from trial factoring does every ECM level from the first, so the estimate needs them all.
        if maximum_ecm_level >= ECM_MIN_LEVEL and self.get_ecm_levels(digits, threads)[:1] != [ECM_MIN_LEVEL]:
            return (0, None)

        tf_count, tf_time, tf_p_factor = self.get_probability_stats(digits, "tf", 1)

        if tf_count == 0:
            return (0, None)

        assert tf_time is not None  # ruff:ignore[assert]
        assert tf_p_factor is not None  # ruff:ignore[assert]

        rho_count, rho_time, rho_p_factor = self.get_probability_stats(digits, "rho", 1)

        if rho_count == 0:
            return (0, None)

        assert rho_time is not None  # ruff:ignore[assert]
        assert rho_p_factor is not None  # ruff:ignore[assert]

        pm1_count, pm1_time, pm1_p_factor = self.get_probability_stats(digits, "pm1", 1)

        if pm1_count == 0:
            return (0, None)

        assert pm1_time is not None  # ruff:ignore[assert]
        assert pm1_p_factor is not None  # ruff:ignore[assert]

        ecm_count, ecm_final_time = self.get_ecm_average_time(digits, maximum_ecm_level, threads, methods)

        if ecm_final_time is None:
            return (0, None)

        total_time = tf_time
        total_time += rho_time * (1 - tf_p_factor)
        total_time += pm1_time * (1 - tf_p_factor) * (1 - rho_p_factor)
        total_time += ecm_final_time * (1 - tf_p_factor) * (1 - rho_p_factor) * (1 - pm1_p_factor)

        return (ecm_count, total_time)

    def get_ecm_average_time(
        self, digits: int, maximum_ecm_level: int, threads: int, methods: Sequence[str]
    ) -> tuple[int, float | None]:
        """Estimate the average time to factor a number with the given digit count, starting from ECM.

        The estimate starts from the first ECM level with data, as if the levels before it were already done.

        Returns:
            tuple[int, float | None]: A tuple containing the estimated number of ECM runs and the average time to factor
                the number, or None if insufficient data is available.
        """
        final_time = self.get_final_time(digits, threads, methods)

        if final_time is None:
            return (0, None)

        first_ecm_level = next(iter(self.get_ecm_levels(digits, threads)), ECM_MIN_LEVEL)

        if maximum_ecm_level < first_ecm_level:
            return (0, final_time)

        return self._get_average_time_internal(digits, threads, final_time, first_ecm_level, maximum_ecm_level)

    def get_ecm_cutoffs(self, digits: int, threads: int, methods: Sequence[str]) -> ECMCutoffs:
        """Determine the ECM levels at which to stop doing ECM factoring.

        Returns:
            ECMCutoffs: The optimal ECM level and the target ECM level to stop at.
        """
        # Establish a semi-arbitrary limit on our maximum ECM level. The smallest factors should never have more than
        # about half the digits of the number, so do a few levels beyond that (as ECM may miss factors).
        maximum_ecm_level = digits // 2 + 10

        # Without enough data to estimate anything, limit the ECM work to one-third the digit count. This is just an
        # initial target, and after the first run we can adjust based on actual data.
        initial_cutoffs = ECMCutoffs(None, _normalize_ecm_level(digits // 3))
        final_time = self.get_final_time(digits, threads, methods)

        if final_time is None:
            return initial_cutoffs

        # Start from the first level with data. The levels before it cost the same whichever later level ECM stops at,
        # so they don't affect which of those is fastest. Stopping at the level before it is doing no further ECM or
        # none at all if it is the first level.
        first_ecm_level = next(iter(self.get_ecm_levels(digits, threads)), ECM_MIN_LEVEL)
        no_ecm_level = first_ecm_level - 1

        # Collect data on the statistics based on stopping ECM at a given level, starting from no further ECM, which
        # leaves only the final method. Once we've reached the next level with no data, simply abort. Along the way,
        # we'll note which ECM level was fastest on average.
        ecm_counts: dict[int, int] = {}
        best_maximum_ecm_level = no_ecm_level
        best_maximum_ecm_level_time = final_time

        for ecm_level in range(first_ecm_level, maximum_ecm_level + 1):
            ecm_count, average_time = self._get_average_time_internal(
                digits, threads, final_time, first_ecm_level, ecm_level
            )

            if ecm_count == 0:
                break

            assert average_time is not None  # ruff:ignore[assert]

            if average_time < best_maximum_ecm_level_time:
                best_maximum_ecm_level = ecm_level
                best_maximum_ecm_level_time = average_time

            ecm_counts[ecm_level] = ecm_count

        if not ecm_counts:
            return initial_cutoffs

        # Cap the maximum ECM level based on the final method statistics. A level that takes longer than the fastest
        # final method can never be worthwhile, no matter how often it finds a factor, so there's no point collecting
        # more data. We do apply a fudge factor in case of measurement inaccuracy.
        for ecm_level in range(ECM_MIN_LEVEL, maximum_ecm_level + 1):
            test_ecm_time = self.get_ecm_stats(digits, ecm_level, threads)[1]

            if test_ecm_time is not None and final_time * 1.25 < test_ecm_time:
                maximum_ecm_level = ecm_level - 1
                break

        # Otherwise, we'll balance collecting more data with taking advantage of what we already know, based on how many
        # samples have been collected. The function as defined here will do an extra number of levels based on the
        # number of samples for the level following the one with the minimum time. The parameters as chosen here will do
        # eight extra levels when there are 4 samples, decaying to zero extra levels when 1024 samples are reached.
        # These numbers are, of course, arbitrary, but should work reasonably well enough. We need to check each level
        # from the minimum up to the test level because in certain situations, the numbers won't be monotonically
        # decreasing.
        test_ecm_level = best_maximum_ecm_level + 1
        test_ecm_levels = range(first_ecm_level, test_ecm_level + 1)
        lowest_ecm_count = min(ecm_counts.get(ecm_level, 0) for ecm_level in test_ecm_levels)
        lowest_ecm_count = max(lowest_ecm_count, 1)

        extra_ecm_levels = max(0, math.ceil(-math.log2(lowest_ecm_count) + 10))

        return ECMCutoffs(
            _normalize_ecm_level(best_maximum_ecm_level),
            _normalize_ecm_level(min(best_maximum_ecm_level + extra_ecm_levels, maximum_ecm_level)),
        )

    def _get_average_time_internal(
        self, digits: int, threads: int, final_time: float, next_ecm_level: int, maximum_ecm_level: int
    ) -> tuple[int, float | None]:
        ecm_count, ecm_time, ecm_p_factor = self.get_ecm_stats(digits, next_ecm_level, threads)

        # If there is no ECM data at this level, we're stuck.
        if ecm_count == 0:
            return (0, None)

        # It is a bug for any of the following to be violated, and it helps the static type checker.
        assert ecm_time is not None  # ruff:ignore[assert]
        assert ecm_p_factor is not None  # ruff:ignore[assert]

        # Adjust the probability of finding a factor by applying an exponentially decaying weighted average. This
        # minimizes the impact of only having a few samples.
        ecm_p_factor_decay = pow(ECM_P_FACTOR_DECAY, ecm_count)
        ecm_p_factor = ecm_p_factor_decay * ECM_P_FACTOR_DEFAULT + (1 - ecm_p_factor_decay) * ecm_p_factor

        # This level of ECM is done unconditionally.
        average_time = ecm_time

        # If no factor is found, we must recursively check the next level.
        if next_ecm_level == maximum_ecm_level:
            final_ecm_count = ecm_count
            average_time += (1 - ecm_p_factor) * final_time
        else:
            final_ecm_count, extra_time = self._get_average_time_internal(
                digits, threads, final_time, next_ecm_level + 1, maximum_ecm_level
            )

            if extra_time is None:
                return (0, None)

            average_time += (1 - ecm_p_factor) * extra_time

        return (final_ecm_count, average_time)


def _normalize_ecm_level(ecm_level: int) -> int:
    """Convert any ECM cutoff below the first level to zero, as they all mean doing no ECM at all.

    Returns:
        int: The ECM cutoff, or zero if it is below the first level.
    """
    return ecm_level if ecm_level >= ECM_MIN_LEVEL else 0
