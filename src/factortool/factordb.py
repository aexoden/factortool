# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Interface for interacting with FactorDB."""

from __future__ import annotations

import datetime
import re
import time

from typing import TYPE_CHECKING

import requests

from loguru import logger
from pydantic import BaseModel

from factortool.backend import SUBMIT_SPACING, BaseBackend

if TYPE_CHECKING:
    from factortool.backend import FetchCriteria
    from factortool.config import Config
    from factortool.number import Number
    from factortool.stats import FactoringStats


class FactorDBSessionData(BaseModel):
    """Session data for FactorDB login persistence."""

    cookies: dict[str, str]
    expiry: datetime.datetime


class FactorDB(BaseBackend):
    """Interface for interacting with FactorDB."""

    name = "FactorDB"

    # FactorDB does not currently have any sort of reservation system.
    assigns_work = False

    submission_unit = "factors"

    def __init__(self, config: Config, stats: FactoringStats) -> None:
        """Initialize the FactorDB interface."""
        super().__init__(config, stats, config.factordb_cooldown_period, config.factordb_username)

        self._load_session()

    def _request_composites(self, criteria: FetchCriteria) -> str:
        """Request composites from FactorDB.

        FactorDB does not itself support max_digits, but BaseBackend automatically filters through its public fetch
        method. All other criteria are respected. Requests are capped at 50 numbers.

        Returns:
            str: The response body, containing one composite per line.
        """
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

        return self._service_request("GET", "https://factordb.com/listtype.php", params=params, timeout=3.0).text

    def _submit_number(self, number: Number) -> int:
        """Submit each prime factor of a number to FactorDB individually.

        Returns:
            int: The number of factors successfully submitted.
        """
        factors = sorted(set(number.prime_factors))

        # If there are no composite factors, avoid sending the trivial largest factor.
        if len(number.composite_factors) == 0:
            factors.pop()

        successes = 0

        for i, factor in enumerate(factors):
            if i > 0:
                time.sleep(SUBMIT_SPACING)

            successes += self._submit_factor(number.n, factor)

        return successes

    def _submit_factor(self, number: int, factor: int) -> bool:
        """Submit a single factor to FactorDB.

        Returns:
            bool: True if the factor was successfully submitted, False otherwise.
        """
        url = "https://factordb.com/reportfactor.php"
        payload = {"number": str(number), "factor": str(factor)}

        try:
            self._service_request("POST", url, data=payload, timeout=3.0)
        except requests.RequestException as e:
            logger.error("Error submitting factor {} for n{}: {}", factor, number, e)
            return False

        logger.debug("Submitted factor {} for n={}", factor, number)
        return True

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
