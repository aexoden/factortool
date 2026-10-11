# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Protocol describing a source of composite numbers and a destination for factorizations.

Also provides BaseBackend, which implements the fetch and submission behavior shared by the concrete backends.
"""

from __future__ import annotations

import enum
import queue
import threading
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

import requests

from loguru import logger

from factortool.http import MAX_DELAY, HttpClient, PermanentHttpError, build_user_agent
from factortool.interrupt import ABORT, POLL_INTERVAL, Interrupted, InterruptState
from factortool.number import Number
from factortool.submissions import Submission, SubmissionJournal

if TYPE_CHECKING:
    from factortool.config import Config
    from factortool.stats import FactoringStats

# How long to wait between consecutive submissions to a service.
SUBMIT_SPACING = 0.2

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


@dataclass
class QueuedSubmission:
    """A submission waiting for the submission worker."""

    submission: Submission
    carried_over: bool = False
    attempts: int = 0


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

    def fetch(self, criteria: FetchCriteria) -> set[Number]:
        """Fetch up to criteria.count matching composites, possibly returning an empty set.

        Raises:
            PermanentHttpError: If a permanent client error is encountered and should not be retried.
            ValueError: If criteria are invalid or contain unsupported constraints.
        """
        ...

    def submit(self, numbers: Collection[Number]) -> None:
        """Queue factored numbers for submission."""
        ...

    def get_successful_submission_count(self) -> int:
        """Get the number of accepted factorizations."""
        ...

    def close(self) -> None:
        """Flush any pending submissions and release resources, retaining unsent for the next run."""
        ...


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

    # The largest number of composites to submit in a single batch.
    submit_batch_size: ClassVar[int] = 1

    # How long, in seconds, a submission that keeps failing stays available for sending, unless its assignment expires
    # sooner.
    submission_lifetime: ClassVar[float] = 86400.0

    def __init__(
        self,
        config: Config,
        stats: FactoringStats,
        cooldown_period: float,
        identity: str,
        interrupts: InterruptState | None = None,
    ) -> None:
        """Initialize the backend and queue the submissions a previous run didn't send.

        Args:
            config (Config): Application configuration.
            stats (FactoringStats): Factoring statistics.
            cooldown_period (float): Cooldown period between requests in seconds.
            identity (str): Account name used with the service, included in the User-Agent. May be empty.
            interrupts (InterruptState | None): Interrupt state for interrupt signals, or None for a private one.
        """
        self._config = config
        self._stats = stats
        self._cooldown_period = cooldown_period
        self._interrupts = interrupts if interrupts is not None else InterruptState()
        self._journal = SubmissionJournal(config.pending_submissions_path, config.backend)
        self._submit_queue: queue.Queue[QueuedSubmission] = queue.Queue()

        carried_over = self._journal.load()

        for submission in carried_over:
            self._submit_queue.put_nowait(QueuedSubmission(submission, carried_over=True))

        if carried_over:
            logger.info(
                "Resubmitting {} result{} from a previous run", len(carried_over), "" if len(carried_over) == 1 else "s"
            )

        rate_limited_until = min(self._journal.rate_limited_until, time.time() + MAX_DELAY)

        self._http_client = HttpClient(
            self.name,
            cooldown_period,
            build_user_agent(identity, config.user_agent),
            rate_limited_until,
            self._journal.note_rate_limit,
        )

        # Set once the backend is closing, after which the worker stops when the queue is empty.
        self._closing = threading.Event()
        self._closing_since = 0.0

        # Set to make the worker stop prematurely, even if the queue is not empty.
        self._give_up = threading.Event()

        self._last_progress = time.monotonic()
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

    def fetch(self, criteria: FetchCriteria) -> set[Number]:  # ruff: ignore[complex-structure]
        """Fetch up to criteria.count matching composites, possibly returning an empty set.

        Composites above criteria.max_digits are discarded even if the service would otherwise return them. Retries
        continue until composites are found, unless an interrupt arrives, in which case an empty set is returned.

        Returns:
            set[Number]: A set of composite numbers matching the criteria.

        Raises:
            PermanentHttpError: If a permanent client error is encountered and should not be retried.
            ValueError: If the service does not support the requested criteria.
        """
        self._validate_criteria(criteria)

        if criteria.count == 0:
            return set()

        delay = max(0.1, self._cooldown_period)

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
                    for n in composites
                    if criteria.max_digits is None or len(str(n)) <= criteria.max_digits
                }

                if numbers:
                    logger.info("Fetched {} numbers from {}", len(numbers), self.name)
                    return numbers

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

    def submit(self, numbers: Collection[Number]) -> None:
        """Add factored numbers to the submission queue."""
        now = time.time()
        submissions = [
            Submission(
                n=number.n,
                prime_factors=tuple(number.prime_factors),
                composite_factors=tuple(number.composite_factors),
                expires_at=number.expires_at if number.expires_at is not None else now + self.submission_lifetime,
            )
            for number in numbers
            if len(number.prime_factors) > 0
        ]

        self._journal.add(submissions)

        for submission in submissions:
            self._submit_queue.put_nowait(QueuedSubmission(submission))

    def get_successful_submission_count(self) -> int:
        """Get the number of accepted factorizations.

        Returns:
            int: The number of accepted factorizations.
        """
        with self._submission_lock:
            return self._successful_submissions

    def close(self) -> None:
        """Flush any pending submissions and stop the submission worker.

        The worker is given until its submissions stop resolving for FLUSH_TIMEOUT, not counting any wait for a rate
        limit. An interrupt received in the meantime stops it at once.

        Raises:
            OSError: If the pending submissions cannot be written.
        """
        if self._submitting:
            self._closing_since = time.monotonic()
            self._closing.set()
            self._await_worker()

        self._journal.compact()

        if self._successful_submissions > 0:
            logger.info("Successfully submitted {} factorizations to {}", self._successful_submissions, self.name)

        if (pending := self._journal.pending_count) > 0:
            logger.warning(
                "{} result{} could not be submitted to {}, and will be resubmitted on the next run",
                pending,
                "" if pending == 1 else "s",
                self.name,
            )

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

                # Waiting for the rate limit to expire isn't a sign of a broken service.
                rate_limit_ended = now + self._http_client.rate_limited_until - time.time()
                stalled_since = max(self._last_progress, self._closing_since, rate_limit_ended)
                allowance = stalled_since + FLUSH_TIMEOUT - now

                if allowance <= 0:
                    logger.warning("Giving up on submitting to {}, as it is not responding", self.name)
                    self._give_up.set()
                    break

                deadline = min(deadline, now + allowance)
                remaining = deadline - now

            self._give_up.wait(min(remaining, POLL_INTERVAL))

        return True

    def _next_batch(self) -> list[QueuedSubmission]:
        """Take up to submit_batch_size submissions from the queue.

        Returns:
            list[QueuedSubmission]: The batch, which is empty if nothing was queued in time.
        """
        batch: list[QueuedSubmission] = []
        now = time.time()

        while len(batch) < self.submit_batch_size:
            try:
                queued = self._submit_queue.get_nowait() if batch else self._submit_queue.get(timeout=POLL_INTERVAL)
            except queue.Empty:
                break

            if (queued.carried_over or queued.attempts > 0) and queued.submission.expires_at <= now:
                logger.warning("Discarding the result for {}, as it was not submitted in time", queued.submission.n)
                self._journal.resolve([queued.submission])
                continue

            batch.append(queued)

        return batch

    def _submit_batch(self, batch: Sequence[QueuedSubmission]) -> bool:
        """Make an attempt at a batch of submissions.

        Returns:
            bool: True if any submission was queued for another attempt.
        """
        outcomes = self._submit_numbers([queued.submission for queued in batch])
        resolved: list[Submission] = []
        retrying = False

        for queued, outcome in zip(batch, outcomes, strict=True):
            if outcome != SubmitOutcome.FAILED:
                resolved.append(queued.submission)
                continue

            queued.attempts += 1

            if queued.attempts < SUBMIT_ATTEMPTS:
                self._submit_queue.put_nowait(queued)
                retrying = True
            else:
                logger.warning(
                    "Leaving the result for {} for the next run after {} failed attempts",
                    queued.submission.n,
                    queued.attempts,
                )

        self._journal.resolve(resolved)

        if resolved:
            self._last_progress = time.monotonic()

        with self._submission_lock:
            self._successful_submissions += outcomes.count(SubmitOutcome.ACCEPTED)

        return retrying

    def _submit_queued(self) -> None:
        """Submit queued numbers until closed and the queue is empty or no more retries are possible.

        Raises:
            Interrupted: If the submission process is interrupted.
        """
        delay = max(0.1, self._cooldown_period)

        while not self._give_up.is_set():
            batch = self._next_batch()

            if not batch:
                if self._closing.is_set() and self._submit_queue.empty():
                    return

                continue

            if self._submit_batch(batch):
                if self._submission_wait(delay):
                    return

                delay = min(MAX_DELAY, delay * 2)
            else:
                delay = max(0.1, self._cooldown_period)
                time.sleep(SUBMIT_SPACING)

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
