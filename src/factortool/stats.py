# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Factoring statistics data models."""

from __future__ import annotations

import json
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import BaseModel, Field

from factortool.constants import ECM_CURVES, ECM_P_FACTOR_DECAY, ECM_P_FACTOR_DEFAULT
from factortool.util import safe_write


class FinalRunData(BaseModel):
    """Time and run count data for final factorization runs (SIQS, NFS)."""

    total_time: float = Field(default=0.0, description="Total time spent on final factorization of this type")
    run_count: int = Field(default=0, description="Number of final factorization runs of this type")


class FinalDigitData(BaseModel):
    """Data for final factorization runs (SIQS, NFS) for a given digit count, grouped by thread count."""

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


class FactoringData(BaseModel):
    """All factoring statistics data."""

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
    nfs: dict[int, FinalDigitData] = Field(
        default_factory=dict[int, FinalDigitData], description="NFS data for each digit count"
    )


class FactoringStats:
    """Class for managing factoring statistics data."""

    _path: Path
    _data: FactoringData
    _min_write_interval: float
    _last_write_time: int
    _data_changed: bool
    _read_only: bool

    def __init__(self, path: Path, *, min_write_interval: float = 5.0, read_only: bool = False) -> None:
        """Initialize the factoring statistics manager."""
        self._path: Path = path
        self._min_write_interval = int(min_write_interval * 1_000_000_000)
        self._last_write_time = 0
        self._data_changed = False
        self._read_only = read_only

        self._load_data()

    def _load_data(self) -> None:
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                self._data = FactoringData.model_validate_json(f.read())
        else:
            self._data = FactoringData()

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
        self._save_data(force=True)

    def update_probability(
        self, digits: int, method: str, threads: int, execution_time: float, *, success: bool
    ) -> None:
        """Update probabilistic factorization data."""
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

    def update_siqs(self, digits: int, threads: int, execution_time: float) -> None:
        """Update SIQS factorization data."""
        if digits not in self._data.siqs:
            self._data.siqs[digits] = FinalDigitData()

        if threads not in self._data.siqs[digits].thread_data:
            self._data.siqs[digits].thread_data[threads] = FinalRunData()

        run_data = self._data.siqs[digits].thread_data[threads]

        run_data.total_time += execution_time
        run_data.run_count += 1

        self._data_changed = True
        self._save_data()

    def update_nfs(self, digits: int, threads: int, execution_time: float) -> None:
        """Update NFS factorization data."""
        if digits not in self._data.nfs:
            self._data.nfs[digits] = FinalDigitData()

        if threads not in self._data.nfs[digits].thread_data:
            self._data.nfs[digits].thread_data[threads] = FinalRunData()

        run_data = self._data.nfs[digits].thread_data[threads]

        run_data.total_time += execution_time
        run_data.run_count += 1

        self._data_changed = True
        self._save_data()

    def update_ecm(self, digits: int, ecm_level: int, threads: int, execution_time: float, *, success: bool) -> None:
        """Update ECM factorization data."""
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

    def get_siqs_stats(self, digits: int, threads: int) -> tuple[int, float | None]:
        """Get SIQS factorization statistics.

        Returns:
            A tuple containing the number of SIQS runs and the average time per run, or None if no data is available.
        """
        if digits in self._data.siqs and threads in self._data.siqs[digits].thread_data:
            run_data = self._data.siqs[digits].thread_data[threads]

            if run_data.run_count > 0:
                return (
                    run_data.run_count,
                    run_data.total_time / run_data.run_count,
                )

        return (0, None)

    def get_nfs_stats(self, digits: int, threads: int) -> tuple[int, float | None]:
        """Get NFS factorization statistics.

        Returns:
            A tuple containing the number of NFS runs and the average time per run, or None if no data is available.
        """
        if digits in self._data.nfs and threads in self._data.nfs[digits].thread_data:
            run_data = self._data.nfs[digits].thread_data[threads]

            if run_data.run_count > 0:
                return (
                    run_data.run_count,
                    run_data.total_time / run_data.run_count,
                )

        return (0, None)

    def get_yafu_stats(self, digits: int, threads: int) -> tuple[int, float | None]:
        """Get YAFU factorization statistics.

        Returns:
            A tuple containing the number of YAFU runs and the average time per run, or None if no data is available.
        """
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

    def get_average_time(self, digits: int, maximum_ecm_level: int, threads: int) -> tuple[int, float | None]:  # noqa: PLR0914
        """Estimate the average time to factor a number with the given digit count.

        Returns:
            A tuple containing the estimated number of ECM runs and the average time to factor the number,
            or None if insufficient data is available.
        """
        tf_count, tf_time, tf_p_factor = self.get_probability_stats(digits, "tf", 1)

        if tf_count == 0:
            return (0, None)

        assert tf_time is not None  # noqa: S101
        assert tf_p_factor is not None  # noqa: S101

        rho_count, rho_time, rho_p_factor = self.get_probability_stats(digits, "rho", 1)

        if rho_count == 0:
            return (0, None)

        assert rho_time is not None  # noqa: S101
        assert rho_p_factor is not None  # noqa: S101

        pm1_count, pm1_time, pm1_p_factor = self.get_probability_stats(digits, "pm1", 1)

        if pm1_count == 0:
            return (0, None)

        assert pm1_time is not None  # noqa: S101
        assert pm1_p_factor is not None  # noqa: S101

        _, siqs_time = self.get_siqs_stats(digits, threads)
        _, nfs_time = self.get_nfs_stats(digits, threads)

        if siqs_time is not None or nfs_time is not None:
            if siqs_time is None:
                assert nfs_time is not None  # noqa: S101
                final_time = nfs_time
            elif nfs_time is None:
                final_time = siqs_time
            else:
                final_time = min(siqs_time, nfs_time)
        else:
            return (0, None)

        ecm_count, ecm_nfs_time = self._get_average_time_internal(
            digits, threads, final_time, min(ECM_CURVES.keys()), maximum_ecm_level
        )

        if ecm_nfs_time is None:
            return (0, None)

        total_time = tf_time
        total_time += rho_time * (1 - tf_p_factor)
        total_time += pm1_time * (1 - tf_p_factor) * (1 - rho_p_factor)
        total_time += ecm_nfs_time * (1 - tf_p_factor) * (1 - rho_p_factor) * (1 - pm1_p_factor)

        return (ecm_count, total_time)

    def _get_average_time_internal(
        self, digits: int, threads: int, final_time: float, next_ecm_level: int, maximum_ecm_level: int
    ) -> tuple[int, float | None]:
        ecm_count, ecm_time, ecm_p_factor = self.get_ecm_stats(digits, next_ecm_level, threads)

        # If there is no ECM data at this level, we're stuck.
        if ecm_count == 0:
            return (0, None)

        # It is a bug for any of the following to be violated, and it helps the static type checker.
        assert ecm_time is not None  # noqa: S101
        assert ecm_p_factor is not None  # noqa: S101

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
