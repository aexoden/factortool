# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import json
import os
import tempfile
import time

from pathlib import Path

from pydantic import BaseModel, Field

from factortool.constants import ECM_CURVES, ECM_P_FACTOR_DECAY, ECM_P_FACTOR_DEFAULT


class NFSRunData(BaseModel):
    total_time: float = Field(default=0.0, description="Total time spent on NFS factorization")
    run_count: int = Field(default=0, description="Number of NFS factorization runs")


class NFSDigitData(BaseModel):
    thread_data: dict[int, NFSRunData] = Field(default_factory=dict, description="NFS data for each thread count")


class ECMRunData(BaseModel):
    total_time: float = Field(default=0.0, description="Total time spent on ECM factorization")
    run_count: int = Field(default=0, description="Number of ECM factorization runs")
    success_count: int = Field(default=0, description="Number of successful ECM factorizations")


class ECMLevelData(BaseModel):
    thread_data: dict[int, ECMRunData] = Field(default_factory=dict, description="ECM data for each thread count")


class ECMDigitData(BaseModel):
    level_data: dict[int, ECMLevelData] = Field(default_factory=dict, description="ECM data for each level")


class FactoringData(BaseModel):
    nfs: dict[int, NFSDigitData] = Field(default_factory=dict, description="NFS data for each digit count")
    ecm: dict[int, ECMDigitData] = Field(default_factory=dict, description="ECM data for each digit count")


class FactoringStats:
    _path: Path
    _data: FactoringData
    _min_write_interval: float
    _last_write_time: int
    _data_changed: bool
    _read_only: bool

    def __init__(self, path: Path, *, min_write_interval: float = 5.0, read_only: bool = False) -> None:
        self._path: Path = path
        self._min_write_interval = int(min_write_interval * 1_000_000_000)
        self._last_write_time = 0
        self._data_changed = False
        self._read_only = read_only

        self._load_data()

    def __del__(self) -> None:
        self._save_data(force=True)

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

        target_path = self._path.parent

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=target_path, suffix=".tmp") as f:
            # Work around pydantic not offering a way to sort keys.
            # f.write(self._data.model_dump_json(indent=2, so))
            model_dict = self._data.model_dump()
            f.write(json.dumps(model_dict, sort_keys=True, indent=2))

            temp_path = Path(f.name)
            f.flush()
            os.fsync(f.fileno())

        temp_path.rename(self._path)

        self._last_write_time = current_time
        self._data_changed = False

    def update_nfs(self, digits: int, threads: int, execution_time: float) -> None:
        if digits not in self._data.nfs:
            self._data.nfs[digits] = NFSDigitData()

        if threads not in self._data.nfs[digits].thread_data:
            self._data.nfs[digits].thread_data[threads] = NFSRunData()

        run_data = self._data.nfs[digits].thread_data[threads]

        run_data.total_time += execution_time
        run_data.run_count += 1

        self._data_changed = True
        self._save_data()

    def update_ecm(self, digits: int, ecm_level: int, threads: int, execution_time: float, *, success: bool) -> None:
        if digits not in self._data.ecm:
            self._data.ecm[digits] = ECMDigitData()

        if ecm_level not in self._data.ecm[digits].level_data:
            self._data.ecm[digits].level_data[ecm_level] = ECMLevelData()

        if threads not in self._data.ecm[digits].level_data[ecm_level].thread_data:
            self._data.ecm[digits].level_data[ecm_level].thread_data[threads] = ECMRunData()

        run_data = self._data.ecm[digits].level_data[ecm_level].thread_data[threads]

        run_data.total_time += execution_time
        run_data.run_count += 1

        if success:
            run_data.success_count += 1

        self._data_changed = True
        self._save_data()

    def get_nfs_stats(self, digits: int, threads: int) -> tuple[int, float | None]:
        if digits in self._data.nfs and threads in self._data.nfs[digits].thread_data:
            run_data = self._data.nfs[digits].thread_data[threads]

            if run_data.run_count > 0:
                return (
                    run_data.run_count,
                    run_data.total_time / run_data.run_count,
                )

        return (0, None)

    def get_ecm_stats(self, digits: int, ecm_level: int, threads: int) -> tuple[int, float | None, float | None]:
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

    def get_average_time(self, digits: int, maximum_ecm_level: int, threads: int) -> tuple[int, float | None]:
        nfs_count, nfs_time = self.get_nfs_stats(digits, threads)

        if nfs_count == 0:
            return (0, None)

        assert nfs_time is not None  # noqa: S101

        return self._get_average_time_internal(digits, threads, nfs_time, min(ECM_CURVES.keys()), maximum_ecm_level)

    def _get_average_time_internal(
        self, digits: int, threads: int, nfs_time: float, next_ecm_level: int, maximum_ecm_level: int
    ) -> tuple[int, float | None]:
        ecm_threads = min(threads, ECM_CURVES[next_ecm_level][0])
        ecm_count, ecm_time, ecm_p_factor = self.get_ecm_stats(digits, next_ecm_level, ecm_threads)

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
            average_time += (1 - ecm_p_factor) * nfs_time
        else:
            final_ecm_count, extra_time = self._get_average_time_internal(
                digits, threads, nfs_time, next_ecm_level + 1, maximum_ecm_level
            )

            if extra_time is None:
                return (0, None)

            average_time += (1 - ecm_p_factor) * extra_time

        return (final_ecm_count, average_time)
