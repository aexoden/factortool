# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import datetime
import re
import time

from collections.abc import Collection

import requests

from loguru import logger
from pydantic import BaseModel

from factortool.config import Config
from factortool.number import Number, format_results
from factortool.stats import FactoringStats


class FactorDBSessionData(BaseModel):
    cookies: dict[str, str]
    expiry: datetime.datetime


class FactorDB:
    _config: Config
    _session: requests.Session
    _stats: FactoringStats

    def __init__(self, config: Config, stats: FactoringStats) -> None:
        self._config = config
        self._stats = stats
        self._load_session()

    def fetch(self, min_digits: int, number_count: int, skip_count: int) -> set[Number]:
        params = {
            "t": 3,
            "mindig": min_digits,
            "perpage": number_count,
            "start": skip_count,
            "download": 1,
        }

        try:
            response = requests.get("https://factordb.com/listtype.php", params=params, timeout=10)
            response.raise_for_status()
            numbers = {Number(x, self._config, self._stats) for x in map(int, response.text.strip().split("\n"))}
            logger.info("Fetched {} numbers from FactorDB", len(numbers))
        except requests.RequestException as e:
            logger.error("Failed to fetch numbers from FactorDB: {}", e)
            numbers: set[Number] = set()

        return numbers

    def submit(self, numbers: Collection[Number]) -> bool:
        url = "https://factordb.com/report.php"

        report = format_results(numbers) + "\n"

        data: dict[str, int | str] = {
            "report": report,
            "format": 7,
        }

        # Submit the results.
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self._session.post(url, data=data)
                response.raise_for_status()

                if self._check_factordb_response(response.text):
                    logger.info("Successfully submitted {} results to FactorDB", len(numbers))
                    return True

                logger.error("FactorDB submission response did not contain expected success message")
            except requests.RequestException as e:
                logger.error("FactorDB submission attempt {} failed: {}", attempt + 1, e)

                if attempt < max_retries - 1:
                    sleep_time = self._config.factordb_cooldown_period
                    logger.info("Retrying in {} seconds...", sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.error("Failed to submit results to FactorDB after {} attempts", max_retries)

        return False

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
            response = self._session.post(login_url, data=login_data)
            response.raise_for_status()
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
                self._session = requests.Session()
                self._session.cookies.update(session_data.cookies)  # type: ignore
                return

        self._session = requests.Session()
        self._login()

    def _save_session(self) -> None:
        expiry = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(days=21)

        session_data = FactorDBSessionData(
            cookies=self._session.cookies.get_dict(),
            expiry=expiry,
        )

        with self._config.factordb_session_path.open("w", encoding="utf-8") as f:
            f.write(session_data.model_dump_json())
