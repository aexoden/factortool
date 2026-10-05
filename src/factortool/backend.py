# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Protocol describing a source of composite numbers and a destination for factorizations.

Also provides BaseBackend, which implements the fetch and submission behavior shared by the concrete backends.
"""

from __future__ import annotations

import queue
import threading
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

import requests

from loguru import logger

from factortool.http import MAX_DELAY, HttpClient
from factortool.number import Number

if TYPE_CHECKING:
    from factortool.config import Config
    from factortool.stats import FactoringStats

# How long to wait between consecutive submissions to a service.
SUBMIT_SPACING = 0.2

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


class Backend(Protocol):
    """A remote service that hands out composite numbers and accepts their factorizations."""

    @property
    def assigns_work(self) -> bool:
        """Whether the service reserves fetched composites for this client."""
        ...

    def fetch(self, criteria: FetchCriteria) -> set[Number]:
        """Fetch up to criteria.count matching composites, possibly returning an empty set.

        Raises:
            ValueError: If criteria are invalid or contain unsupported constraints.
        """
        ...

    def submit(self, numbers: Collection[Number]) -> None:
        """Queue factored numbers for submission."""
        ...

    def get_successful_submission_count(self) -> int:
        """Get the number of successful submissions."""
        ...

    def close(self) -> None:
        """Flush any pending submissions and release resources."""
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
    submission behavior.
    """

    # Name of the service, used for logging.
    name: ClassVar[str]

    # Whether the service reserves fetched composites for this client.
    assigns_work: bool

    # A descriptive name for a single submission unit, used for logging.
    submission_unit: ClassVar[str]

    def __init__(self, config: Config, stats: FactoringStats, cooldown_period: float) -> None:
        """Initialize the backend and start its submission worker."""
        self._config = config
        self._stats = stats
        self._cooldown_period = cooldown_period
        self._http_client = HttpClient(self.name, cooldown_period)
        self._submit_queue: queue.Queue[Number] = queue.Queue()
        self._stop_event = threading.Event()
        self._successful_submissions = 0
        self._submission_lock = threading.Lock()

        self._submit_thread = threading.Thread(
            target=self._submit_worker, name=f"{type(self).__name__}-Submission-Worker", daemon=True
        )
        self._submit_thread.start()

    def fetch(self, criteria: FetchCriteria) -> set[Number]:
        """Fetch up to criteria.count matching composites, possibly returning an empty set.

        Composites above criteria.max_digits are discarded even if the service would otherwise return them.

        Returns:
            set[Number]: A set of composite numbers matching the criteria.

        Raises:
            ValueError: If the service does not support the requested criteria.
        """
        self._validate_criteria(criteria)

        if criteria.count == 0:
            return set()

        delay = max(0.1, self._cooldown_period)

        while True:
            try:
                composites = parse_composites(self._request_composites(criteria))
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
                time.sleep(NO_WORK_DELAY)
                continue

            time.sleep(delay)
            delay = min(MAX_DELAY, delay * 2)

    def submit(self, numbers: Collection[Number]) -> None:
        """Add factored numbers to the submission queue."""
        for number in numbers:
            if len(number.prime_factors) > 0:
                self._submit_queue.put_nowait(number)

    def get_successful_submission_count(self) -> int:
        """Get the number of successful submissions.

        Returns:
            int: The number of successful submissions, counted in units of submission_unit.
        """
        with self._submission_lock:
            return self._successful_submissions

    def close(self) -> None:
        """Flush any pending submissions and stop the submission worker."""
        self._stop_event.set()
        self._submit_thread.join()

        if self._successful_submissions > 0:
            logger.info(
                "Successfully submitted {} {} to {}", self._successful_submissions, self.submission_unit, self.name
            )

    def _validate_criteria(self, criteria: FetchCriteria) -> None:  # ruff: ignore[empty-method-without-abstract-decorator] (Optional hook)
        """Reject criteria the service cannot honor by raising a ValueError. By default, all criteria are supported."""

    @abstractmethod
    def _request_composites(self, criteria: FetchCriteria) -> str:
        """Request composites from the service, which may return fewer than requested or none at all.

        Returns:
            str: The response body, containing one composite per line.
        """

    @abstractmethod
    def _submit_number(self, number: Number) -> int:
        """Submit a single factored number to the service.

        Returns:
            int: The number of successful submissions, counted in units of submission_unit.
        """

    def _service_request(  # ruff: ignore[too-many-arguments] (Mirrors HttpClient.request)
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, int | str] | None = None,
        data: Mapping[str, str] | None = None,
        files: Mapping[str, tuple[None, str]] | None = None,
        timeout: float,
    ) -> requests.Response:
        """Perform a fetch or submission request, retrying transient failures indefinitely.

        Returns:
            requests.Response: The HTTP response.
        """
        return self._http_client.request(
            method, url, params=params, data=data, files=files, timeout=timeout, max_attempts=None
        )

    def _submit_worker(self) -> None:
        """Background worker that submits queued numbers until closed and the queue is drained."""
        while not self._stop_event.is_set() or not self._submit_queue.empty():
            try:
                number = self._submit_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            successes = self._submit_number(number)

            with self._submission_lock:
                self._successful_submissions += successes

            self._submit_queue.task_done()
            time.sleep(SUBMIT_SPACING)


def create_backend(config: Config, stats: FactoringStats) -> Backend:
    """Build the backend selected in the configuration.

    Returns:
        Backend: The configured backend.
    """
    # Imported here to avoid a circular import: both backends subclass BaseBackend from this module.
    from factortool.factordb import FactorDB  # ruff: ignore[import-outside-top-level]
    from factortool.mersenne_ca import MersenneCA  # ruff: ignore[import-outside-top-level]

    if config.backend == "mersenne_ca":
        return MersenneCA(config, stats)

    return FactorDB(config, stats)
