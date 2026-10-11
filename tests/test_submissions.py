# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the journal of pending submissions."""

from __future__ import annotations

import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from factortool.submissions import Submission, SubmissionJournal

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

    journal.add([FIFTEEN])

    assert journal.pending_count == 1

    path.rmdir()
    journal.compact()

    assert SubmissionJournal(path, "factordb").load() == [FIFTEEN]
