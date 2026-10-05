# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Interface for interacting with the mersenne.ca Aliquot composite service.

The service hands out the smallest known unfactored composites blocking incomplete Aliquot sequences, reserving each for
an hour, and forwards reported factorizations to FactorDB within minutes. See https://www.mersenne.ca/aliquot/?compositelist=1
for the endpoint documentation.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING, Any, cast, override

import requests

from loguru import logger

from factortool.backend import BaseBackend
from factortool.number import format_factorization

if TYPE_CHECKING:
    from factortool.backend import FetchCriteria
    from factortool.config import Config
    from factortool.number import Number
    from factortool.stats import FactoringStats

API_URL = "https://www.mersenne.ca/aliquot/index.php"


def check_factorization(n: int, factors: list[int]) -> bool:
    """Check that a list of factors actually multiplies back to the original number.

    Returns:
        bool: True if the factors are consistent with the number.
    """
    return len(factors) > 0 and math.prod(factors) == n


class MersenneCA(BaseBackend):
    """Interface for interacting with the mersenne.ca Aliquot composite service."""

    name = "mersenne.ca"

    # Fetched composites are reserved for this client for an hour.
    assigns_work = True

    submission_unit = "factorizations"

    def __init__(self, config: Config, stats: FactoringStats) -> None:
        """Initialize the mersenne.ca interface."""
        super().__init__(config, stats, config.mersenne_ca_cooldown_period)

        if not config.gimps_login:
            logger.error("No GIMPS login is configured; mersenne.ca requires one to assign and accept work")

    @override
    def _validate_criteria(self, criteria: FetchCriteria) -> None:
        """Reject unsupported criteria.

        Supports all criteria except the skip count, which does not apply. In addition, max_digits is required.

        Raises:
            ValueError: If no maximum digit count was supplied, or if a skip count was specified.
        """
        if criteria.max_digits is None:
            msg = "mersenne.ca requires a maximum digit count"
            raise ValueError(msg)

        # The service assigns distinct work to each caller, so there is nothing to skip.
        if criteria.skip_count != 0:
            msg = "mersenne.ca does not support a skip count"
            raise ValueError(msg)

    def _request_composites(self, criteria: FetchCriteria) -> str:
        """Request composites assigned by mersenne.ca. An empty body means no work is available.

        Returns:
            str: The response body, containing one composite per line.
        """
        params: dict[str, int | str] = {
            "composites_to_factor": criteria.count,
            "min_digits": criteria.min_digits,
            # Guaranteed by _validate_criteria.
            "max_digits": cast("int", criteria.max_digits),
            "gimps_login": self._config.gimps_login,
        }

        return self._service_request("GET", API_URL, params=params, timeout=30.0).text

    def _submit_number(self, number: Number) -> int:
        """Report a single composite's factorization, complete or partial.

        Returns:
            int: 1 if the service accepted the factorization, otherwise 0.
        """
        factors = number.prime_factors + number.composite_factors

        if not check_factorization(number.n, factors):
            logger.error("Refusing to report an inconsistent factorization for {}", number.n)
            return 0

        # The service expects a multipart POST, matching the documented "curl -F" invocation.
        payload: dict[str, tuple[None, str]] = {
            "compositefactorization": (None, format_factorization(number, "*")),
            "gimps_login": (None, self._config.gimps_login),
        }

        try:
            response = self._service_request("POST", API_URL, files=payload, timeout=30.0)
        except requests.RequestException as e:
            logger.error("Error reporting factorization for {}: {}", number.n, e)
            return 0

        return int(self._log_response(number, response))

    @staticmethod
    def _log_response(number: Number, response: requests.Response) -> bool:
        """Log the response and return whether the service accepted the submission.

        Returns:
            bool: Whether the service accepted the submission.
        """
        try:
            body: Any = response.json()
        except ValueError:
            logger.error("mersenne.ca returned a malformed response for {}", number.n)
            return False

        if not isinstance(body, dict):
            logger.error("mersenne.ca returned a malformed response for {}", number.n)
            return False

        fields = cast("dict[str, Any]", body)

        if warning := fields.get("warning"):
            logger.warning("mersenne.ca reported a warning for {}: {}", number.n, warning)

        if error := fields.get("error"):
            logger.error("mersenne.ca reported an error for {}: {}", number.n, error)
            return False

        logger.debug("Reported factorization for {}", number.n)
        return True
