# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the CLI of factortool."""

from __future__ import annotations

import json
import signal
import sys
import time

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from collections.abc import Collection

import pytest

from factortool.assignments import AssignmentStore
from factortool.backend import FetchCriteria
from factortool.cli.main import (
    Arguments,
    acquire_numbers,
    get_time_limit,
    main,
    preserve_unfinished,
    validate_arguments,
    warn_if_assignments_may_expire,
)
from factortool.engine import ExitStatus, FactorEngine
from factortool.interrupt import EXIT_STATUS, InterruptState
from factortool.number import CadoNfsError, Number, ToolError, YafuError
from factortool.stats import FactoringStats

from .helpers import make_config

CRITERIA = FetchCriteria(count=3, min_digits=1, max_digits=100)

# Composite, so Number does not treat them as already factored.
CARRIED_OVER = [100, 102]
FETCHED = [104, 106, 108]


class FakeBackend:
    """A backend that records requests, assigning work unless told otherwise."""

    assignment_lifetime = 3600.0

    def __init__(self, *, assigns_work: bool = True) -> None:
        """Initialize the fake backend."""
        self.assigns_work = assigns_work
        self.requested: list[int] = []
        self.submitted: list[int] = []

    def fetch(self, criteria: FetchCriteria) -> set[Number]:
        """Record the request and hand back the canned composites.

        Returns:
            set[Number]: The composites this backend always returns.
        """
        self.requested.append(criteria.count)
        config = make_config()
        stats = FactoringStats(Path("stats.json"), read_only=True)

        return {Number(n, config, stats, None) for n in FETCHED[: criteria.count]}

    def submit(self, numbers: Collection[Number]) -> None:
        """Record a factorization instead of reporting it anywhere."""
        self.submitted.extend(x.n for x in numbers)

    def get_successful_submission_count(self) -> int:
        """Report how many factorizations were handed over.

        Returns:
            int: The number of submissions recorded.
        """
        return len(self.submitted)

    def close(self) -> None:
        """Close the fake backend, doing nothing."""


def test_a_normal_run_tops_the_batch_back_up(tmp_path: Path) -> None:
    """Test that retained work counts toward the batch and the rest is fetched."""
    backend = FakeBackend()
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", backend.assignment_lifetime)
    store.note_assigned(CARRIED_OVER)
    store.save(CARRIED_OVER)

    numbers = acquire_numbers(backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA)

    assert backend.requested == [1]
    assert sorted(x.n for x in numbers) == sorted([*CARRIED_OVER, FETCHED[0]])


def test_acquired_numbers_carry_their_assignment_expiry(tmp_path: Path) -> None:
    """Test that both retained and fetched numbers know when their assignment expires."""
    backend = FakeBackend()
    state_path = tmp_path / "assignments.json"
    first = AssignmentStore(state_path, "mersenne_ca", backend.assignment_lifetime)
    first.note_assigned(CARRIED_OVER)
    first.save(CARRIED_OVER)

    store = AssignmentStore(state_path, "mersenne_ca", backend.assignment_lifetime)
    numbers = acquire_numbers(backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA)

    assert all(x.expires_at is not None and x.expires_at == store.expires_at(x.n) for x in numbers)


def test_no_new_work_works_only_what_is_already_assigned(tmp_path: Path) -> None:
    """Test that the no new work option only uses already assigned work."""
    backend = FakeBackend()
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", backend.assignment_lifetime)
    store.note_assigned(CARRIED_OVER)
    store.save(CARRIED_OVER)

    numbers = acquire_numbers(
        backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA, fetch=False
    )

    assert backend.requested == []
    assert sorted(x.n for x in numbers) == CARRIED_OVER


def test_no_new_work_with_nothing_carried_over_finds_no_work(tmp_path: Path) -> None:
    """Test that the no new work option finds no work when nothing is retained."""
    backend = FakeBackend()
    store = AssignmentStore(tmp_path / "assignments.json", "mersenne_ca", backend.assignment_lifetime)

    numbers = acquire_numbers(
        backend, store, make_config(), FactoringStats(tmp_path / "stats.json"), CRITERIA, fetch=False
    )

    assert backend.requested == []
    assert numbers == set()


def test_an_early_exit_reports_partial_progress_without_an_assignment(tmp_path: Path) -> None:
    """Test that an early exit reports partial progress for a backend that does not assign work."""
    backend = FakeBackend(assigns_work=False)
    store = AssignmentStore(tmp_path / "assignments.json", "factordb", backend.assignment_lifetime)
    stats = FactoringStats(tmp_path / "stats.json", read_only=True)

    partially_factored = Number(200, make_config(), stats, backend)
    partially_factored.prime_factors = [2, 2]
    partially_factored.composite_factors = [50]
    untouched = Number(300, make_config(), stats, backend)

    preserve_unfinished(backend, store, [partially_factored, untouched])

    assert backend.submitted == [200]
    assert not (tmp_path / "assignments.json").exists()


def test_no_new_work_is_rejected_by_a_backend_that_assigns_nothing() -> None:
    """Test that the no new work option is refused on FactorDB, where it could only ever find no work."""
    args = Arguments().parse_args(["--no_new_work"])

    with pytest.raises(SystemExit) as error:
        validate_arguments(args, "factordb")

    assert error.value.code == 1


@pytest.mark.parametrize(("error", "exit_status"), [(YafuError("YAFU failed"), 5), (CadoNfsError("CADO failed"), 4)])
def test_a_tool_failure_exits_with_the_tool_status_after_cleaning_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: ToolError, exit_status: int
) -> None:
    """Test that a tool failure during the run still saves state before exiting with that tool's status."""
    config = {"backend": "factordb", "cado_nfs_path": "cado-nfs.py", "max_threads": 1, "yafu_path": "yafu"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", "--batch_size", "3"])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState.install", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(return_value=FakeBackend(assigns_work=False)))
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", Mock(side_effect=error), raising=True)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == exit_status
    assert (tmp_path / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1


def test_a_termination_signal_still_saves_state_before_exiting_as_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a run ended by SIGTERM saves its state and results, and exits with the interrupted status."""
    config = {"backend": "factordb", "cado_nfs_path": "cado-nfs.py", "max_threads": 1, "yafu_path": "yafu"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    backend = FakeBackend(assigns_work=False)
    backend.close = Mock()  # type: ignore[method-assign]
    interrupts = InterruptState()

    def run(_self: FactorEngine, _numbers: Collection[Number], _time_limit: float | None) -> ExitStatus:
        signal.raise_signal(signal.SIGTERM)
        time.sleep(0.5)

        return ExitStatus.SUCCESS

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", "--batch_size", "3"])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(return_value=backend))
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", run, raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState", lambda: interrupts, raising=True)

    try:
        with pytest.raises(SystemExit) as raised:
            main()
    finally:
        interrupts.uninstall()

    assert raised.value.code == EXIT_STATUS
    assert (tmp_path / "stats.json").exists()
    assert len(list((tmp_path / "results").iterdir())) == 1
    backend.close.assert_called_once()


def test_an_automatic_batch_is_limited_to_twice_its_target_duration() -> None:
    """Test that a run with an automatic batch size may take up to twice the target duration."""
    assert get_time_limit(Arguments().parse_args(["--target_duration", "300"])) == pytest.approx(600.0)


def test_an_explicit_batch_size_has_no_time_limit() -> None:
    """Test that the target duration imposes no time limit when the batch size was chosen by the user."""
    assert get_time_limit(Arguments().parse_args(["--batch_size", "3", "--target_duration", "300"])) is None


@pytest.mark.parametrize(("time_limit", "warned"), [(3600.0, True), (1200.0, False), (None, False)])
def test_a_time_limit_that_may_outlast_assignments_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, time_limit: float | None, *, warned: bool
) -> None:
    """Test that the expiry warning depends on the time limit, and is skipped when there is none."""
    warning = Mock()
    monkeypatch.setattr("factortool.cli.main.logger.warning", warning, raising=True)

    warn_if_assignments_may_expire(FakeBackend(), time_limit)

    assert warning.called == warned


@pytest.mark.parametrize(("arguments", "time_limit"), [([], 1200.0), (["--batch_size", "3"], None)])
def test_the_run_is_given_the_time_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arguments: list[str], time_limit: float | None
) -> None:
    """Test that the time limit reaches the engine with the run rather than when the engine is created."""
    config = {"backend": "factordb", "cado_nfs_path": "cado-nfs.py", "max_threads": 1, "yafu_path": "yafu"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    run = Mock(return_value=ExitStatus.SUCCESS)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["factortool", *arguments])
    monkeypatch.setattr("factortool.cli.main.setup_logger", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.InterruptState.install", Mock(), raising=True)
    monkeypatch.setattr("factortool.cli.main.create_backend", Mock(return_value=FakeBackend(assigns_work=False)))
    monkeypatch.setattr("factortool.cli.main.FactorEngine.run", run, raising=True)

    main()

    assert run.call_args.args[1] == time_limit
