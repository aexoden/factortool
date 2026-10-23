# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Protocol describing a source of composite numbers and a destination for factorizations.

Also provides BaseBackend, which implements the fetch and submission behavior shared by the concrete backends.
"""

from __future__ import annotations

import contextlib
import enum
import queue
import threading
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

import requests

from loguru import logger

from factortool.assignments import Assignment, AssignmentStore
from factortool.constants import MIN_FACTORABLE
from factortool.http import MAX_DELAY, RETRY_DELAY, HttpClient, PermanentHttpError, build_user_agent
from factortool.interrupt import ABORT, POLL_INTERVAL, Interrupted, InterruptState
from factortool.number import Number
from factortool.submissions import Submission, SubmissionJournal
from factortool.util import format_number, is_prime

if TYPE_CHECKING:
    from factortool.config import Config
    from factortool.stats import FactoringStats

# The longest a partial batch of submissions is held before submission.
SUBMIT_MAX_HOLD = 30.0

# How many times to attempt a submission during a run before leaving it for the next run.
SUBMIT_ATTEMPTS = 5

# How long a backend that is shutting down keeps retrying submissions before giving up.
FLUSH_TIMEOUT = 30.0

# How long to wait when a service returns no composites within the requested range. Matches the delay used by the
# reference mersenne.ca aliquot.php client.
NO_WORK_DELAY = 65.0


@dataclass(frozen=True)
class FetchCriteria:
    """Criteria for fetching composite numbers.

    Backends must reject unsupported constraints rather than silently ignoring them.
    """

    count: int
    min_digits: int
    max_digits: int | None = None
    skip_count: int = 0

    def __post_init__(self) -> None:
        """Validate fetch constraints.

        Raises:
            ValueError: If counts or digit bounds are invalid.
        """
        if self.count < 0:
            message = "count must be nonnegative"
            raise ValueError(message)
        if self.skip_count < 0:
            message = "skip_count must be nonnegative"
            raise ValueError(message)
        if self.min_digits < 1:
            message = "min_digits must be at least 1"
            raise ValueError(message)
        if self.max_digits is not None and self.max_digits < self.min_digits:
            message = "max_digits must be at least min_digits"
            raise ValueError(message)


class SubmitOutcome(enum.Enum):
    """Outcome of a submission to a remote service."""

    # The service accepted the submission.
    ACCEPTED = enum.auto()

    # The service will never accept the submission.
    REJECTED = enum.auto()

    # The service temporarily rejected the submission.
    FAILED = enum.auto()

    # The service refused the request as a whole, not just this submission in particular.
    DEFERRED = enum.auto()


@dataclass
class QueuedSubmission:
    """A submission waiting for the submission worker."""

    submission: Submission
    attempts: int = 0

    # When the submission should be sent even if its batch isn't full, as a time.monotonic() value.
    hold_until: float = 0.0


class Backend(Protocol):
    """A remote service that hands out composite numbers and accepts their factorizations."""

    @property
    def assigns_work(self) -> bool:
        """Whether the service reserves fetched composites for this client."""
        ...

    @property
    def assignment_lifetime(self) -> float:
        """The lifetime of the assignment in seconds."""
        ...

    @property
    def accepts_partial_results(self) -> bool:
        """Whether the service accepts factorizations that include composite factors."""
        ...

    def resume(self) -> set[Number]:
        """Resume the numbers whose assignments a previous run left unfinished, with the factors it found."""
        ...

    def fetch(self, criteria: FetchCriteria) -> set[Number]:
        """Fetch up to criteria.count matching composites, possibly returning an empty set.

        A service that limits how many composites a single request returns is asked for no more than that limit.

        Raises:
            PermanentHttpError: If a permanent client error is encountered and should not be retried.
            ValueError: If criteria are invalid or contain unsupported constraints.
        """
        ...

    def submit(self, numbers: Collection[Number]) -> None:
        """Queue factored numbers for submission, releasing their assignments."""
        ...

    def checkpoint(self, number: Number) -> None:
        """Record the progress made on an assigned number, so a later run can resume it."""
        ...

    def get_successful_submission_count(self) -> int:
        """Get the number of accepted factorizations."""
        ...

    def close(self) -> None:
        """Flush pending submissions and release resources, retaining unsent and unfinished work for the next run."""
        ...


def snapshot_assignment(number: Number) -> Assignment | None:
    """Capture a number's assignment with the factors found so far.

    Returns:
        Assignment | None: The assignment, or None if the number is not assigned.
    """
    if number.expires_at is None:
        return None

    return Assignment(n=number.n, expires_at=number.expires_at, factors=tuple(number.known_factors))


def parse_composites(text: str) -> list[int]:
    """Parse a fetch response consisting of one decimal integer per line, ignoring blank lines.

    Returns:
        list[int]: A list of composite numbers parsed from the response, empty if the response is blank.

    Raises:
        ValueError: If the response contains a non-empty line that is not a decimal integer.
    """
    return [int(line) for line in (raw.strip() for raw in text.splitlines()) if line]


class BaseBackend(ABC):
    """Shared implementation of the Backend protocol for remote services.

    Subclasses only describe how to talk to their service. This class implements the common logic for managing fetch and
    submission behavior. A subclass must call _start_submitting once it is ready to submit.
    """

    # Name of the service, used for logging.
    name: ClassVar[str]

    # Whether the service reserves fetched composites for this client.
    assigns_work: bool

    # How long, in seconds, the service holds a fetched composite for this client. Only meaningful if assigns_work.
    assignment_lifetime: float

    # Whether the service accepts factorizations that include composite factors.
    accepts_partial_results: bool

    # The most composites the service returns for a single request, or None if it has no limit.
    max_fetch_count: ClassVar[int | None] = None

    # The largest number of composites to submit in a single batch.
    submit_batch_size: ClassVar[int] = 1

    # How long, in seconds, a submission that keeps failing stays available for sending, unless its assignment expires
    # sooner. A factorization never goes stale by itself, so this only bounds the attempts.
    submission_lifetime: ClassVar[float] = 30 * 86400.0

    def __init__(
        self,
        config: Config,
        stats: FactoringStats,
        request_interval: float,
        identity: str,
        interrupts: InterruptState | None = None,
    ) -> None:
        """Initialize the backend and queue the submissions a previous run didn't send.

        Args:
            config (Config): Application configuration.
            stats (FactoringStats): Factoring statistics.
            request_interval (float): The minimum time between the starts of consecutive requests, in seconds.
            identity (str): Account name used with the service, included in the User-Agent. May be empty.
            interrupts (InterruptState | None): Interrupt state for interrupt signals, or None for a private one.
        """
        self._config = config
        self._stats = stats
        self._interrupts = interrupts if interrupts is not None else InterruptState()
        self._journal = SubmissionJournal(config.pending_submissions_path, config.backend)
        self._submit_queue: queue.Queue[QueuedSubmission] = queue.Queue()

        self._assignments = AssignmentStore(config.assignment_state_path, config.backend) if self.assigns_work else None
        self._assigned: dict[int, Number] = {}
        self._assigned_lock = threading.Lock()

        # Composites whose results are queued but the journal failed to record.
        self._unjournaled: set[int] = set()

        carried_over = self._journal.load()
        self._resubmitting = {x.n for x in carried_over}

        self._carried_assignments: list[Assignment] = []

        if self._assignments is not None:
            self._carried_assignments = [x for x in self._assignments.load() if x.n not in self._resubmitting]
            self._assignments.release(self._resubmitting)

        for submission in carried_over:
            self._submit_queue.put_nowait(QueuedSubmission(submission))

        if carried_over:
            logger.info(
                "Resubmitting {} result{} from a previous run", len(carried_over), "" if len(carried_over) == 1 else "s"
            )

        self._http_client = HttpClient(
            self.name,
            request_interval,
            build_user_agent(identity, config.user_agent),
            self._journal.rate_limited_until,
            self._journal.note_rate_limit,
        )

        # Set once the backend is closing, after which the worker stops when the queue is empty.
        self._closing = threading.Event()
        self._closing_since = 0.0

        # Set to make the worker stop prematurely, even if the queue is not empty.
        self._give_up = threading.Event()

        self._last_progress = time.monotonic()
        self._last_queued = time.monotonic()
        self._successful_submissions = 0
        self._submission_lock = threading.Lock()

        self._submit_thread = threading.Thread(
            target=self._submit_worker, name=f"{type(self).__name__}-Submission-Worker", daemon=True
        )
        self._submitting = False

    def _start_submitting(self) -> None:
        """Start the submission worker."""
        self._submitting = True
        self._submit_thread.start()

    def _discard_non_composites(self, numbers: Sequence[int]) -> list[int]:
        """Discard anything the service returned that is not a composite, as there is nothing to factor.

        Returns:
            list[int]: The numbers that are composite.
        """
        composites = [n for n in numbers if n >= MIN_FACTORABLE and not is_prime(n)]

        if len(composites) < len(numbers):
            discarded = len(numbers) - len(composites)
            logger.warning(
                "Discarding {} number{} from {} that {} not composite",
                discarded,
                "" if discarded == 1 else "s",
                self.name,
                "is" if discarded == 1 else "are",
            )

        return composites

    def fetch(self, criteria: FetchCriteria) -> set[Number]:  # ruff: ignore[complex-structure]
        """Fetch up to criteria.count matching composites, possibly returning an empty set.

        A request for more than max_fetch_count composites is reduced to max_fetch_count. Composites above
        criteria.max_digits are discarded even if the service would otherwise return them. Retries continue until
        composites are found, unless an interrupt arrives, in which case an empty set is returned.

        Returns:
            set[Number]: A set of composite numbers matching the criteria.

        Raises:
            PermanentHttpError: If a permanent client error is encountered and should not be retried.
            ValueError: If the service does not support the requested criteria.
        """
        self._validate_criteria(criteria)

        if criteria.count == 0:
            return set()

        if self.max_fetch_count is not None and criteria.count > self.max_fetch_count:
            logger.info(
                "Fetching {} composite numbers from {} (reduced from the {} requested, due to the service maximum)",
                self.max_fetch_count,
                self.name,
                criteria.count,
            )
            criteria = replace(criteria, count=self.max_fetch_count)
        else:
            logger.info(
                "Fetching {} composite number{} from {}", criteria.count, "" if criteria.count == 1 else "s", self.name
            )

        delay = RETRY_DELAY

        while not self._interrupts.stop_fetching:
            try:
                composites = self._request_composites(criteria)
            except PermanentHttpError:
                raise
            except Interrupted:
                break
            except ValueError as e:
                logger.error("Failed to parse response from {}: {}. Retrying in {} seconds...", self.name, e, delay)
            except requests.RequestException as e:
                logger.error("Failed to fetch numbers from {}: {}. Retrying in {} seconds...", self.name, e, delay)
            else:
                numbers = {
                    Number(n, self._config, self._stats, self)
                    for n in self._discard_non_composites(composites)
                    if criteria.max_digits is None or len(str(n)) <= criteria.max_digits
                }

                if numbers:
                    logger.info("Fetched {} numbers from {}", len(numbers), self.name)
                    return self._assign(numbers)

                logger.info(
                    "No work available from {} between {} and {} digits. Retrying in {} seconds...",
                    self.name,
                    criteria.min_digits,
                    criteria.max_digits if criteria.max_digits is not None else "unlimited",
                    NO_WORK_DELAY,
                )

                if self._interrupts.wait(NO_WORK_DELAY):
                    break

                continue

            if self._interrupts.wait(delay):
                break

            delay = min(MAX_DELAY, delay * 2)

        logger.warning("Abandoning fetch from {} due to an interrupt", self.name)
        return set()

    def resume(self) -> set[Number]:
        """Resume the numbers whose assignments a previous run left unfinished, with the factors it found.

        A number whose result is already waiting to be resubmitted is not resumed, as the previous run must have been
        stopped between submitting it and releasing its assignment. A number that is already fully factored, because the
        previous run could not record its result in the journal, is queued for submission instead of being resumed.

        Returns:
            set[Number]: The resumed numbers, which is empty if the service does not assign work.
        """
        if self._assignments is None:
            return set()

        resumed: list[Number] = []

        for assignment in self._carried_assignments:
            number = Number(assignment.n, self._config, self._stats, self, assignment.factors)
            number.expires_at = assignment.expires_at
            resumed.append(number)

        self._carried_assignments = []

        with self._assigned_lock:
            self._assigned.update((x.n, x) for x in resumed)

        self._record_assignments(resumed)

        if finished := [x for x in resumed if x.factored]:
            logger.info(
                "Resubmitting {} result{} a previous run could not record",
                len(finished),
                "" if len(finished) == 1 else "s",
            )
            self.submit(finished)

        unfinished = {x for x in resumed if not x.factored}

        if unfinished:
            logger.info(
                "Resuming {} composite{} from a previous run", len(unfinished), "" if len(unfinished) == 1 else "s"
            )

        return unfinished

    def _assign(self, numbers: set[Number]) -> set[Number]:
        """Take on the assignments of freshly fetched numbers, recording them before any work is done.

        Returns:
            set[Number]: The assigned numbers.
        """
        if self._assignments is None:
            return numbers

        expires_at = time.time() + self.assignment_lifetime
        assigned: set[Number] = set()

        with self._assigned_lock:
            for fetched in numbers:
                number = self._assigned.setdefault(fetched.n, fetched)
                number.expires_at = expires_at
                assigned.add(number)

        self._record_assignments(assigned)

        return assigned

    def _record_assignments(self, numbers: Iterable[Number]) -> None:
        """Record the assignments of the given numbers along with their progress."""
        if self._assignments is not None:
            self._assignments.record(x for x in map(snapshot_assignment, numbers) if x is not None)

    def checkpoint(self, number: Number) -> None:
        """Record the progress made on a number, if it is held under an assignment."""
        if self._assignments is None or (assignment := snapshot_assignment(number)) is None:
            return

        with self._assigned_lock:
            if self._assigned.get(number.n) is not number:
                return

            self._assignments.record([assignment])

    def submit(self, numbers: Collection[Number]) -> None:
        """Add factored numbers to the submission queue.

        Each is held for up to SUBMIT_MAX_HOLD in the hope of filling a batch, unless the previous result arrived at
        least that long ago. A submission is never held within SUBMIT_MAX_HOLD of its expiration.
        """
        now = time.time()
        submitted = [number for number in numbers if len(number.prime_factors) > 0]
        submissions = [
            Submission(
                n=number.n,
                prime_factors=tuple(number.prime_factors),
                composite_factors=tuple(number.composite_factors),
                expires_at=number.expires_at if number.expires_at is not None else now + self.submission_lifetime,
            )
            for number in submitted
        ]

        if not submissions:
            return

        journaled = self._journal.add(submissions)

        if self._assignments is not None:
            with self._assigned_lock:
                if journaled:
                    # The journal now keeps each result, so the assignment no longer needs to be retained.
                    for submission in submissions:
                        self._assigned.pop(submission.n, None)

                    self._assignments.release(x.n for x in submissions)
                else:
                    # Otherwise, the assignment keeps the result so a later run can submit it.
                    held = [x for x in submitted if self._assigned.get(x.n) is x]
                    self._unjournaled.update(x.n for x in held)
                    self._assignments.record(x for x in map(snapshot_assignment, held) if x is not None)

        with self._submission_lock:
            queued_at = time.monotonic()
            hold = SUBMIT_MAX_HOLD if queued_at - self._last_queued < SUBMIT_MAX_HOLD else 0.0
            self._last_queued = queued_at

            for submission in submissions:
                lifetime = submission.expires_at - now
                hold_until = queued_at + min(hold, max(0.0, lifetime - SUBMIT_MAX_HOLD))
                self._submit_queue.put_nowait(QueuedSubmission(submission, hold_until=hold_until))

    def get_successful_submission_count(self) -> int:
        """Get the number of accepted factorizations.

        Returns:
            int: The number of accepted factorizations.
        """
        with self._submission_lock:
            return self._successful_submissions

    def close(self) -> None:
        """Flush any pending submissions, stop the submission worker and retain any unfinished assignments.

        The worker is given until its submissions stop resolving for FLUSH_TIMEOUT, not counting any wait for a rate
        limit. An interrupt received in the meantime stops it at once.

        Raises:
            OSError: If the pending submissions or the unfinished assignments cannot be written.
        """
        if self._submitting:
            self._closing_since = time.monotonic()
            self._closing.set()
            self._await_worker()

        try:
            self._journal.compact()
            self._release_unjournaled()
        finally:
            if self._assignments is not None:
                self._assignments.compact()

        if self._successful_submissions > 0:
            logger.info("Successfully submitted {} factorizations to {}", self._successful_submissions, self.name)

        if (pending := self._journal.pending_count) > 0:
            logger.warning(
                "{} result{} could not be submitted to {}, and will be resubmitted on the next run",
                pending,
                "" if pending == 1 else "s",
                self.name,
            )

    def _release_unjournaled(self, composites: Iterable[int] | None = None) -> None:
        """Release the assignments retained for results the journal failed to record."""
        if self._assignments is None:
            return

        with self._assigned_lock:
            released = self._unjournaled.copy() if composites is None else self._unjournaled.intersection(composites)

            if not released:
                return

            self._unjournaled -= released

            for n in released:
                self._assigned.pop(n, None)

            self._assignments.release(released)

    def _await_worker(self) -> None:
        """Wait for the submission worker to stop, telling it to give up if an interrupt arrives."""
        level = self._interrupts.level
        announced = False

        while self._submit_thread.is_alive():
            self._submit_thread.join(POLL_INTERVAL)

            if self._give_up.is_set() or not self._submit_thread.is_alive():
                continue

            rate_limit = self._http_client.rate_limited_until - time.time()

            if self._interrupts.level > level:
                logger.warning("Abandoning the remaining submissions to {} due to an interrupt", self.name)
                self._give_up.set()
            elif rate_limit > 0 and self._interrupts.level >= ABORT:
                logger.warning("Not waiting for the {} rate limit to end before exiting", self.name)
                self._give_up.set()
            elif rate_limit > 0 and not announced:
                logger.info(
                    "Waiting {:.0f} seconds for the {} rate limit to end before submitting the remaining results. "
                    "Interrupt to exit now and submit them on the next run",
                    rate_limit,
                    self.name,
                )
                announced = True

    def _validate_criteria(self, criteria: FetchCriteria) -> None:  # ruff: ignore[empty-method-without-abstract-decorator] (Optional hook)
        """Reject criteria the service cannot honor by raising a ValueError. By default, all criteria are supported."""

    @abstractmethod
    def _request_composites(self, criteria: FetchCriteria) -> list[int]:
        """Request composites from the service, which may return fewer than requested or none at all.

        Returns:
            list[int]: The composite numbers returned by the service.

        Raises:
            ValueError: If the response is malformed.
        """

    @abstractmethod
    def _submit_number(self, submission: Submission) -> SubmitOutcome:
        """Make a single attempt to submit a factorization to the service.

        Returns:
            SubmitOutcome: The outcome of the submission attempt.

        Raises:
            Interrupted: If the submission was interrupted before completion.
        """

    def _submit_numbers(self, submissions: Sequence[Submission]) -> list[SubmitOutcome]:
        """Make a single attempt to submit a batch of factorizations. By default, each is submitted individually.

        Returns:
            list[SubmitOutcome]: The outcomes of the submission attempts.

        Raises:
            Interrupted: If the submission was interrupted before completion.
        """
        return [self._submit_number(submission) for submission in submissions]

    def _service_request(  # ruff: ignore[too-many-arguments] (Mirrors HttpClient.request)
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, int | str] | None = None,
        data: Mapping[str, str] | None = None,
        files: Mapping[str, tuple[None, str]] | None = None,
        json: object = None,
        timeout: float,
        wait: Callable[[float], bool],
    ) -> requests.Response:
        """Perform a fetch or submission request, retrying transient failures until the wait abandons the attempt.

        Returns:
            requests.Response: The HTTP response.
        """
        return self._http_client.request(
            method,
            url,
            params=params,
            data=data,
            files=files,
            json=json,
            timeout=timeout,
            max_attempts=None,
            wait=wait,
        )

    def _submission_wait(self, delay: float) -> bool:
        """Wait before another attempt at a submission, for as long as the attempt should continue.

        Returns:
            bool: True if the submission should be abandoned, False otherwise.
        """
        was_closing = self._closing.is_set()
        deadline = time.monotonic() + delay

        while not self._give_up.is_set():
            now = time.monotonic()
            remaining = deadline - now

            if remaining <= 0:
                return False

            if self._closing.is_set():
                if not was_closing:
                    return False

                # Waiting for a rate limit or a request interval isn't a sign of a broken service.
                rate_limit_ended = now + self._http_client.rate_limited_until - time.time()
                stalled_since = max(
                    self._last_progress, self._closing_since, rate_limit_ended, self._http_client.next_request_at
                )
                allowance = stalled_since + FLUSH_TIMEOUT - now

                if allowance <= 0:
                    logger.warning("Giving up on submitting to {}, as it is not responding", self.name)
                    self._give_up.set()
                    break

                deadline = min(deadline, now + allowance)
                remaining = deadline - now

            self._give_up.wait(min(remaining, POLL_INTERVAL))

        return True

    def _collect(self, held: list[QueuedSubmission], timeout: float) -> None:
        """Move everything queued into held, first waiting up to timeout for something to be queued."""
        with contextlib.suppress(queue.Empty):
            held.append(self._submit_queue.get(timeout=timeout) if timeout > 0 else self._submit_queue.get_nowait())

            while True:
                held.append(self._submit_queue.get_nowait())

    def _batch_ready(self, held: Sequence[QueuedSubmission]) -> bool:
        """Determine whether to send a batch now rather than wait for more submissions.

        Returns:
            bool: True if anything is held and a batch is full, the backend is closing or a hold has ended.
        """
        if not held:
            return False

        return (
            len(held) >= self.submit_batch_size
            or self._closing.is_set()
            or min(queued.hold_until for queued in held) <= time.monotonic()
        )

    def _take_batch(self, held: list[QueuedSubmission]) -> list[QueuedSubmission]:
        """Take up to submit_batch_size submissions from the front of held, discarding any that have expired.

        Returns:
            list[QueuedSubmission]: The batch, which is empty if everything taken had expired.
        """
        batch: list[QueuedSubmission] = []
        expired: list[Submission] = []
        now = time.time()

        while held and len(batch) < self.submit_batch_size:
            queued = held.pop(0)

            # The composite behind an expired assignment may have already been reassigned.
            if queued.submission.expires_at <= now:
                expired.append(queued.submission)
            else:
                batch.append(queued)

        self._discard_expired(expired)
        return batch

    def _discard_expired(self, submissions: Sequence[Submission]) -> None:
        """Handle submissions that have expired without being submitted."""
        for submission in submissions:
            logger.warning(
                "Discarding the result for {}, as it was not submitted in time",
                format_number(submission.n, truncate=False),
            )

        self._journal.resolve(submissions)
        self._release_unjournaled(x.n for x in submissions)

    def _submit_batch(self, batch: Sequence[QueuedSubmission]) -> bool:
        """Make an attempt at a batch of submissions.

        Returns:
            bool: True if any submission was queued for another attempt.
        """
        outcomes = self._submit_numbers([queued.submission for queued in batch])
        now = time.time()
        resolved: list[Submission] = []
        expired: list[Submission] = []
        retrying = False

        for queued, outcome in zip(batch, outcomes, strict=True):
            if outcome == SubmitOutcome.DEFERRED:
                continue

            if outcome != SubmitOutcome.FAILED:
                resolved.append(queued.submission)
                continue

            queued.attempts += 1

            if queued.submission.expires_at <= now:
                expired.append(queued.submission)
            elif queued.attempts < SUBMIT_ATTEMPTS:
                self._submit_queue.put_nowait(queued)
                retrying = True
            else:
                logger.warning(
                    "Leaving the result for {} for the next run after {} failed attempts",
                    format_number(queued.submission.n, truncate=False),
                    queued.attempts,
                )

        self._journal.resolve(resolved)
        self._release_unjournaled(x.n for x in resolved)
        self._discard_expired(expired)

        if resolved:
            self._last_progress = time.monotonic()
            logger.info("{}", self._describe_submissions(outcomes))

        with self._submission_lock:
            self._successful_submissions += outcomes.count(SubmitOutcome.ACCEPTED)

        return retrying

    def _describe_submissions(self, outcomes: Sequence[SubmitOutcome]) -> str:
        """Describe the outcomes of a batch of submissions for the log.

        Returns:
            str: A summary of how many submissions in the batch were accepted.
        """
        accepted = outcomes.count(SubmitOutcome.ACCEPTED)
        noun = "factorization" if len(outcomes) == 1 else "factorizations"

        if accepted == len(outcomes):
            return f"{self.name} accepted {accepted} {noun}"

        return f"{self.name} accepted {accepted} of {len(outcomes)} {noun}"

    def _submit_queued(self) -> None:
        """Submit queued numbers until closed and the queue is empty or no more retries are possible.

        Submissions are held until a batch is full, the oldest hold ends or the backend is closing. Spacing between
        batches is left to the HTTP client.

        Raises:
            Interrupted: If the submission process is interrupted.
        """
        held: list[QueuedSubmission] = []
        delay = RETRY_DELAY

        while not self._give_up.is_set():
            if not self._batch_ready(held):
                timeout = POLL_INTERVAL

                if held:
                    timeout = min(timeout, min(queued.hold_until for queued in held) - time.monotonic())

                self._collect(held, timeout)

                if not held and self._closing.is_set() and self._submit_queue.empty():
                    return

                continue

            # Anything queued while waiting for the service joins the batch.
            self._http_client.await_ready(self._submission_wait)
            self._collect(held, 0.0)
            batch = self._take_batch(held)

            if not batch:
                continue

            if self._submit_batch(batch):
                if self._submission_wait(delay):
                    return

                delay = min(MAX_DELAY, delay * 2)
            else:
                delay = RETRY_DELAY

    def _submit_worker(self) -> None:
        """Background worker that submits queued numbers, leaving whatever it cannot send for the next run."""
        try:
            self._submit_queued()
        except Interrupted:
            logger.debug("Abandoned a submission to {}, leaving it for the next run", self.name)
        except Exception:  # ruff: ignore[blind-except] (The results are kept, and the run must shut down.)
            logger.exception("The {} submission worker failed", self.name)


def create_backend(config: Config, stats: FactoringStats, interrupts: InterruptState) -> Backend:
    """Build the backend selected in the configuration.

    Returns:
        Backend: The configured backend.
    """
    # Imported here to avoid a circular import: both backends subclass BaseBackend from this module.
    from factortool.factordb import FactorDB  # ruff: ignore[import-outside-top-level]
    from factortool.mersenne_ca import MersenneCA  # ruff: ignore[import-outside-top-level]

    if config.backend == "mersenne_ca":
        return MersenneCA(config, stats, interrupts)

    return FactorDB(config, stats, interrupts)
