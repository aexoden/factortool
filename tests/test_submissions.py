# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the journal of pending submissions."""

from __future__ import annotations

import time

from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Self
from unittest.mock import Mock

if TYPE_CHECKING:
    from collections.abc import Iterable

import pytest

from factortool.submissions import Submission, SubmissionJournal
from factortool.util import safe_write

FIFTEEN = Submission(n=15, prime_factors=(3, 5), composite_factors=(), expires_at=1.0)
PARTIAL = Submission(n=15015, prime_factors=(3, 5), composite_factors=(1001,), expires_at=2.0)
LARGE = Submission(n=10**150 + 7, prime_factors=(7,), composite_factors=((10**150 + 7) // 7,), expires_at=3.0)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    """Locate a journal that does not exist yet.

    Returns:
        Path: The path of the journal.
    """
    return tmp_path / "pending_submissions.jsonl"


def test_a_missing_journal_has_nothing_pending(path: Path) -> None:
    """Test that a journal that was never written is empty, and stays unwritten."""
    journal = SubmissionJournal(path, "factordb")

    assert journal.load() == []
    assert journal.pending_count == 0
    assert journal.rate_limited_until == pytest.approx(0.0)

    journal.compact()

    assert not path.exists()


def test_unresolved_submissions_survive_without_compaction(path: Path) -> None:
    """Test that a run that ends without any cleanup still leaves its unresolved submissions to the next."""
    journal = SubmissionJournal(path, "factordb")
    journal.add([FIFTEEN, PARTIAL, LARGE])
    journal.resolve([PARTIAL])

    assert journal.pending_count == 2  # ruff: ignore[magic-value-comparison]
    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN, LARGE]


def test_loading_compacts_the_journal(path: Path) -> None:
    """Test that loading rewrites the journal without its resolved submissions."""
    journal = SubmissionJournal(path, "factordb")
    journal.add([FIFTEEN, PARTIAL])
    journal.resolve([FIFTEEN])

    SubmissionJournal(path, "factordb").load()

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_a_journal_with_nothing_pending_is_removed(path: Path) -> None:
    """Test that compacting a journal whose submissions are all resolved removes the file."""
    journal = SubmissionJournal(path, "factordb")
    journal.add([FIFTEEN])
    journal.resolve([FIFTEEN])
    journal.compact()

    assert not path.exists()


def test_a_later_submission_replaces_an_earlier_one(path: Path) -> None:
    """Test that queueing a composite again keeps only its latest factorization, at the back of the queue."""
    complete = Submission(n=15015, prime_factors=(3, 5, 7, 11, 13), composite_factors=(), expires_at=4.0)
    journal = SubmissionJournal(path, "factordb")
    journal.add([PARTIAL, FIFTEEN, complete])

    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN, complete]


def test_resolving_an_earlier_submission_keeps_a_later_one(path: Path) -> None:
    """Test that an older factorization being resolved does not discard a newer one for the same composite."""
    complete = Submission(n=15015, prime_factors=(3, 5, 7, 11, 13), composite_factors=(), expires_at=4.0)
    journal = SubmissionJournal(path, "factordb")
    journal.add([PARTIAL])
    journal.add([complete])
    journal.resolve([PARTIAL])

    assert journal.pending_count == 1
    assert SubmissionJournal(path, "factordb").load() == [complete]

    journal.resolve([complete])

    assert journal.pending_count == 0
    assert SubmissionJournal(path, "factordb").load() == []


def test_other_backends_submissions_are_kept_but_not_returned(path: Path) -> None:
    """Test that switching backends neither submits nor discards what the other backend left unsent."""
    SubmissionJournal(path, "mersenne_ca").add([PARTIAL])

    journal = SubmissionJournal(path, "factordb")
    assert journal.load() == []
    assert journal.pending_count == 0

    journal.add([FIFTEEN])
    journal.resolve([PARTIAL])
    journal.compact()

    assert SubmissionJournal(path, "mersenne_ca").load() == [PARTIAL]
    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN]


def test_a_partially_written_line_is_ignored(path: Path) -> None:
    """Test that a run killed while writing costs only the line it was writing."""
    SubmissionJournal(path, "factordb").add([FIFTEEN, PARTIAL])

    with path.open("a", encoding="utf-8") as f:
        f.write('{"backend":"factordb","submission":{"n":21,"prime_f')

    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN, PARTIAL]


def test_a_rate_limit_is_kept_until_it_ends(path: Path) -> None:
    """Test that a rate limit still in effect reaches the next run, and one that has ended is dropped."""
    rate_limited_until = time.time() + 600.0
    journal = SubmissionJournal(path, "factordb")
    journal.note_rate_limit(time.time() + 60.0)
    journal.note_rate_limit(rate_limited_until)

    assert journal.rate_limited_until == rate_limited_until

    reloaded = SubmissionJournal(path, "factordb")
    reloaded.load()

    assert reloaded.rate_limited_until == rate_limited_until
    assert SubmissionJournal(path, "mersenne_ca").rate_limited_until == pytest.approx(0.0)

    reloaded.note_rate_limit(time.time() - 1.0)
    reloaded.compact()

    assert not path.exists()


def test_a_failure_to_record_does_not_lose_the_submission(path: Path) -> None:
    """Test that a submission that could not be appended is still written when the journal is compacted."""
    path.mkdir()
    journal = SubmissionJournal(path, "factordb")

    assert not journal.add([FIFTEEN])
    assert journal.pending_count == 1

    path.rmdir()
    journal.compact()

    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN]


def test_adding_reports_that_the_submission_was_recorded(path: Path) -> None:
    """Test that a submission appended to the journal is reported as recorded."""
    journal = SubmissionJournal(path, "factordb")

    assert journal.add([FIFTEEN])
    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN]


class PartiallyFailingFile:
    """A journal opened for appending that writes only part of its first line before failing."""

    def __init__(self, f: IO[str]) -> None:
        """Wrap the opened journal."""
        self._f = f

    def __enter__(self) -> Self:
        """Enter the context of the opened journal.

        Returns:
            Self: The journal.
        """
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Close the journal."""
        self._f.close()

    def writelines(self, lines: Iterable[str]) -> None:
        """Write part of the first line, then fail.

        Raises:
            OSError: Always, as if the disk filled up partway through.
        """
        self._f.write(next(iter(lines))[:10])
        self._f.flush()

        message = "disk full"
        raise OSError(message)


def fail_appends_partway(monkeypatch: pytest.MonkeyPatch, path: Path, count: int = 1) -> None:
    """Make the next appends to the journal write only part of their first line before failing."""
    remaining = [count]
    real_open = Path.open

    def open_failing_partway(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> IO[Any] | PartiallyFailingFile:  # ruff: ignore[any-type] (Mirrors Path.open)
        f = real_open(self, mode, *args, **kwargs)

        if self == path and mode == "a" and remaining[0] > 0:
            remaining[0] -= 1
            return PartiallyFailingFile(f)

        return f

    monkeypatch.setattr(Path, "open", open_failing_partway)


def test_a_partially_written_append_is_repaired_by_the_next_write(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Test that a write after a partially written append rewrites the journal rather than appending to the fragment."""
    journal = SubmissionJournal(path, "factordb")
    fail_appends_partway(monkeypatch, path)

    assert not journal.add([FIFTEEN])
    assert journal.add([PARTIAL])
    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN, PARTIAL]


def test_writes_are_not_reported_as_recorded_until_the_journal_is_repaired(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    """Test that every write fails while the journal cannot be rewritten after a partially written append."""
    journal = SubmissionJournal(path, "factordb")
    fail_appends_partway(monkeypatch, path)
    monkeypatch.setattr("factortool.submissions.safe_write", Mock(side_effect=OSError("disk full")))

    assert not journal.add([FIFTEEN])
    assert not journal.add([PARTIAL])

    monkeypatch.setattr("factortool.submissions.safe_write", safe_write)

    assert journal.add([LARGE])
    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN, PARTIAL, LARGE]


def test_a_partial_line_left_by_a_killed_run_is_repaired_by_the_next_write(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    """Test that a partially written line that loading could not compact away is never appended to."""
    line = FIFTEEN.model_dump_json()
    path.write_text(f'{{"backend":"factordb","submission":{line}}}\n{{"backend":"fa', encoding="utf-8")
    journal = SubmissionJournal(path, "factordb")
    monkeypatch.setattr("factortool.submissions.safe_write", Mock(side_effect=OSError("disk full")))

    assert journal.load() == [FIFTEEN]

    monkeypatch.setattr("factortool.submissions.safe_write", safe_write)

    assert journal.add([PARTIAL])
    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN, PARTIAL]
