# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Persistence for factorizations that have not yet been submitted."""

from __future__ import annotations

import threading
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from factortool.util import safe_write


class Submission(BaseModel):
    """A factorization, complete or partial, ready for submission."""

    model_config = ConfigDict(frozen=True)

    n: int
    prime_factors: tuple[int, ...]
    composite_factors: tuple[int, ...]
    expires_at: float


class JournalRecord(BaseModel):
    """A single line of the journal, which queues a submission, resolves one or notes a rate limit."""

    backend: str
    submission: Submission | None = None
    resolved: Submission | None = None
    rate_limited_until: float | None = None


class SubmissionJournal:
    """An append-only journal of submissions queued for each backend.

    Submissions are recorded as they are queued. The journal is rewritten without resolved submissions when it is
    loaded and when it is compacted.
    """

    def __init__(self, path: Path, backend_name: str) -> None:
        """Initialize the journal for the given backend. Other backends' submissions are kept but never returned."""
        self._path = path
        self._backend_name = backend_name
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, int], Submission] = {}
        self._rate_limited_until: dict[str, float] = {}
        self._append_failed = False

    @property
    def pending_count(self) -> int:
        """The number of pending submissions for this backend."""
        with self._lock:
            return sum(1 for backend, _ in self._pending if backend == self._backend_name)

    @property
    def rate_limited_until(self) -> float:
        """When the backend's most recently recorded rate limit ends, as a timestamp."""
        with self._lock:
            return self._rate_limited_until.get(self._backend_name, 0.0)

    def load(self) -> list[Submission]:
        """Read the journal left by previous runs.

        Returns:
            list[Submission]: The pending submissions for this backend, in the order they were queued.
        """
        with self._lock:
            try:
                lines = self._path.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                return []
            except OSError as e:
                logger.warning("Ignoring unreadable pending submissions at {}: {}", self._path, e)
                return []

            unreadable = 0

            for line in lines:
                try:
                    self._apply(JournalRecord.model_validate_json(line))
                except ValidationError:
                    # A run that was killed while writing can leave a partially written line.
                    unreadable += 1

            if unreadable > 0:
                logger.warning(
                    "Ignoring {} unreadable line{} in {}", unreadable, "" if unreadable == 1 else "s", self._path
                )

            try:
                self._rewrite()
            except OSError as e:
                logger.warning("Failed to compact pending submissions at {}: {}", self._path, e)

            return [x for (backend, _), x in self._pending.items() if backend == self._backend_name]

    def add(self, submissions: Iterable[Submission]) -> None:
        """Record submissions that have been queued."""
        self._record(JournalRecord(backend=self._backend_name, submission=x) for x in submissions)

    def resolve(self, submissions: Iterable[Submission]) -> None:
        """Record that the given submissions are resolved."""
        self._record(JournalRecord(backend=self._backend_name, resolved=x) for x in submissions)

    def note_rate_limit(self, rate_limited_until: float) -> None:
        """Record when a rate limit reported by the backend ends."""
        self._record([JournalRecord(backend=self._backend_name, rate_limited_until=rate_limited_until)])

    def compact(self) -> None:
        """Rewrite the journal with only its pending submissions and any rate limit still in effect.

        Raises:
            OSError: If the journal cannot be written.
        """
        with self._lock:
            self._rewrite()

    def _apply(self, record: JournalRecord) -> None:
        """Update the state for a single record."""
        if record.submission is not None:
            self._pending.pop((record.backend, record.submission.n), None)
            self._pending[record.backend, record.submission.n] = record.submission

        if record.resolved is not None and self._pending.get((record.backend, record.resolved.n)) == record.resolved:
            del self._pending[record.backend, record.resolved.n]

        if record.rate_limited_until is not None:
            self._rate_limited_until[record.backend] = record.rate_limited_until

    def _record(self, records: Iterable[JournalRecord]) -> None:
        """Apply records and append them to the journal, logging a failure to write instead of raising an exception."""
        with self._lock:
            lines: list[str] = []

            for record in records:
                self._apply(record)
                lines.append(record.model_dump_json(exclude_none=True) + "\n")

            if not lines:
                return

            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.writelines(lines)
            except OSError as e:
                # The current run continues without the journal, but the next compaction will attempt to rewrite it.
                if not self._append_failed:
                    logger.error("Failed to record pending submissions at {}: {}", self._path, e)
                    self._append_failed = True

    def _rewrite(self) -> None:
        """Replace the journal with its live records, removing it if there are none."""
        now = time.time()
        records = [JournalRecord(backend=backend, submission=x) for (backend, _), x in self._pending.items()]
        records.extend(
            JournalRecord(backend=backend, rate_limited_until=rate_limited_until)
            for backend, rate_limited_until in self._rate_limited_until.items()
            if rate_limited_until > now
        )

        if not records:
            self._path.unlink(missing_ok=True)
            return

        data = "".join(record.model_dump_json(exclude_none=True) + "\n" for record in records)
        safe_write(self._path, data.encode("utf-8"))
