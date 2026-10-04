# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Interface for interacting with FactorDB."""

from __future__ import annotations

import datetime
import queue
import re
import threading
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

import requests

from loguru import logger
from pydantic import BaseModel

from factortool.http import MAX_DELAY, HttpClient
from factortool.number import Number

if TYPE_CHECKING:
    from factortool.backend import FetchCriteria
    from factortool.config import Config
    from factortool.stats import FactoringStats


class FactorDBSessionData(BaseModel):
    """Session data for FactorDB login persistence."""

    cookies: dict[str, str]
    expiry: datetime.datetime


class FactorDB:
    """Interface for interacting with FactorDB."""

    # FactorDB does not currently have any sort of reservation system.
    assigns_work = False

    def __init__(self, config: Config, stats: FactoringStats) -> None:
        """Initialize the FactorDB interface."""
        self._config = config
        self._stats = stats
        self._submit_queue: queue.Queue[Number] = queue.Queue()
        self._stop_event = threading.Event()
        self._successful_submissions = 0
        self._submission_lock = threading.Lock()
        self._http_client = HttpClient("FactorDB", config.factordb_cooldown_period)

        self._load_session()

        self._submit_thread = threading.Thread(
            target=self._submit_worker, name="FactorDB-Submission-Worker", daemon=True
        )
        self._submit_thread.start()

    def fetch(self, criteria: FetchCriteria) -> set[Number]:
        """Fetch composite numbers from FactorDB.

        Supports all criteria. Requests are capped at 50 numbers. Especially when filtering by max_digits, but feasibly
        also if FactorDB runs out of work, the number of fetched numbers may be fewer than requested, including zero.

        Returns:
            set[Number]: Set of fetched numbers.
        """
        if criteria.count == 0:
            return set()

        # Limit to a maximum of 50 numbers per request to avoid overloading FactorDB. This is intended to be a temporary
        # measure.
        number_count = min(criteria.count, 50)

        params = {
            "t": 3,
            "mindig": criteria.min_digits,
            "perpage": number_count,
            "start": criteria.skip_count,
            "download": 1,
        }

        delay = self._config.factordb_cooldown_period

        while True:
            try:
                response = self._http_client.request(
                    "GET", "https://factordb.com/listtype.php", params=params, timeout=3.0, max_attempts=None
                )
                numbers = {
                    Number(x, self._config, self._stats, self)
                    for x in map(int, response.text.strip().split("\n"))
                    if criteria.max_digits is None or len(str(x)) <= criteria.max_digits
                }
            except ValueError as e:
                logger.error("Failed to parse response from FactorDB: {}. Retrying in {} seconds...", e, delay)
                time.sleep(delay)
                delay = min(MAX_DELAY, delay * 2)
            except requests.RequestException as e:
                logger.error("Failed to fetch numbers from FactorDB: {}. Retrying in {} seconds...", e, delay)
                time.sleep(delay)
                delay = min(MAX_DELAY, delay * 2)
            else:
                logger.info("Fetched {} numbers from FactorDB", len(numbers))
                return numbers

    def submit(self, numbers: Collection[Number]) -> None:
        """Add factored numbers to the submission queue."""
        for number in numbers:
            if len(number.prime_factors) > 0:
                self._submit_queue.put_nowait(number)

    def get_successful_submission_count(self) -> int:
        """Get the number of successful factor submissions.

        Returns:
            int: The number of factors successfully submitted to FactorDB.
        """
        with self._submission_lock:
            return self._successful_submissions

    def close(self) -> None:
        """Close the FactorDB interface, ensuring all submissions are complete."""
        self._stop_event.set()
        self._submit_thread.join()

        if self._successful_submissions > 0:
            logger.info("Successfully submitted {} factors to FactorDB", self._successful_submissions)

    def _submit_worker(self) -> None:
        """Background worker that submits factored numbers to FactorDB."""
        spacing = 0.2

        while not self._stop_event.is_set() or not self._submit_queue.empty():
            try:
                number = self._submit_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            factors = sorted(set(number.prime_factors))

            # If there are no composite factors, avoid sending the trivial largest factor.
            if len(number.composite_factors) == 0:
                factors.pop()

            for factor in factors:
                self._submit_factor(number.n, factor)
                time.sleep(spacing)

            self._submit_queue.task_done()

    def _submit_factor(self, number: int, factor: int) -> None:
        """Submit a single factor to FactorDB."""
        url = "https://factordb.com/reportfactor.php"
        payload = {"number": str(number), "factor": str(factor)}

        try:
            self._http_client.request("POST", url, data=payload, timeout=3.0, max_attempts=None)
            logger.debug("Submitted factor {} for n={}", factor, number)
            with self._submission_lock:
                self._successful_submissions += 1
        except requests.RequestException as e:
            logger.error("Error submitting factor {} for n{}: {}", factor, number, e)

    def _check_factordb_response(self, response_text: str) -> bool:
        with self._config.factordb_response_path.open(mode="w", encoding="utf-8") as f:
            f.write(response_text)

        logged_in_pattern = r"Logged in as <b>(.*)</b>"
        match = re.search(logged_in_pattern, response_text)
        if match:
            logger.info("FactorDB reports logged in as {}", match.group(1))
        elif self._config.factordb_username:
            logger.warning("Attemping to relogin as FactorDB reports not logged in")
            self._login()
        else:
            logger.warning("Results were submitted anonymously as no FactorDB username is configured")

        success_pattern = r"Found (\d+) factors and \d+ ECM/P-1/P\+1 results."
        match = re.search(success_pattern, response_text)
        if match:
            factors_found = int(match.group(1))
            logger.info("FactorDB reports {} factors were added to the database", factors_found)
            return True

        logger.error("Could not find expected success message in FactorDB reponse")
        return False

    def _login(self) -> bool:
        if not self._config.factordb_username:
            logger.warning("Results will be submitted anonymously as no FactorDB username is set")
            return False

        login_url = "https://factordb.com/login.php"

        login_data = {
            "user": self._config.factordb_username,
            "pass": self._config.factordb_password,
            "dlogin": "Login",
        }

        try:
            self._http_client.request("POST", login_url, data=login_data, timeout=5.0, max_attempts=5)
        except requests.RequestException as e:
            logger.error("FactorDB login failed: {}", e)
            return False

        self._save_session()

        return True

    def _load_session(self) -> None:
        if self._config.factordb_session_path.exists():
            with self._config.factordb_session_path.open("r", encoding="utf-8") as f:
                session_data = FactorDBSessionData.model_validate_json(f.read())

            if datetime.datetime.now(tz=datetime.UTC) < session_data.expiry - datetime.timedelta(hours=1):
                self._http_client.set_cookies(session_data.cookies)
                return

        self._login()

    def _save_session(self) -> None:
        expiry = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(days=21)

        session_data = FactorDBSessionData(
            cookies=self._http_client.get_cookies(),
            expiry=expiry,
        )

        with self._config.factordb_session_path.open("w", encoding="utf-8") as f:
            f.write(session_data.model_dump_json())
