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

if TYPE_CHECKING:
    from collections.abc import Sequence

import requests

from loguru import logger

from factortool.backend import BaseBackend, SubmitOutcome, parse_composites
from factortool.http import PermanentHttpError
from factortool.number import format_factorization

if TYPE_CHECKING:
    from factortool.backend import FetchCriteria
    from factortool.config import Config
    from factortool.interrupt import InterruptState
    from factortool.stats import FactoringStats
    from factortool.submissions import Submission

API_URL = "https://www.mersenne.ca/aliquot/index.php"


def check_factorization(n: int, factors: Sequence[int]) -> bool:
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
    assignment_lifetime = 3600.0
    submission_lifetime = assignment_lifetime

    def __init__(self, config: Config, stats: FactoringStats, interrupts: InterruptState | None = None) -> None:
        """Initialize the mersenne.ca interface.

        Raises:
            ValueError: If no GIMPS login is configured.
        """
        super().__init__(config, stats, config.mersenne_ca_cooldown_period, config.gimps_login, interrupts)

        # In theory, this should never happen because the configuration would be rejected, but this is provides a
        # fallback in case of a configuration created without validation.
        if not config.gimps_login:
            msg = "No GIMPS login is configured; mersenne.ca requires one to assign and accept work"
            raise ValueError(msg)

        self._start_submitting()

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

    def _request_composites(self, criteria: FetchCriteria) -> list[int]:
        """Request composites assigned by mersenne.ca. An empty body means no work is available.

        Returns:
            list[int]: The composite numbers assigned by mersenne.ca.
        """
        params: dict[str, int | str] = {
            "composites_to_factor": criteria.count,
            "min_digits": criteria.min_digits,
            # Guaranteed by _validate_criteria.
            "max_digits": cast("int", criteria.max_digits),
            "gimps_login": self._config.gimps_login,
        }

        return parse_composites(
            self._service_request("GET", API_URL, params=params, timeout=30.0, wait=self._interrupts.wait).text
        )

    def _submit_number(self, submission: Submission) -> SubmitOutcome:
        """Make a single attempt to report a composite's factorization, complete or partial.

        Returns:
            SubmitOutcome: The outcome of the submission.
        """
        if not check_factorization(submission.n, submission.prime_factors + submission.composite_factors):
            logger.error("Refusing to report an inconsistent factorization for {}", submission.n)
            return SubmitOutcome.REJECTED

        # The service expects a multipart POST, matching the documented "curl -F" invocation.
        payload: dict[str, tuple[None, str]] = {
            "compositefactorization": (None, format_factorization(submission, "*")),
            "gimps_login": (None, self._config.gimps_login),
        }

        try:
            response = self._service_request("POST", API_URL, files=payload, timeout=30.0, wait=self._submission_wait)
        except PermanentHttpError as e:
            logger.error("Discarding the factorization for {} that mersenne.ca will not accept: {}", submission.n, e)
            return SubmitOutcome.REJECTED
        except requests.RequestException as e:
            logger.warning("Error reporting factorization for {}: {}", submission.n, e)
            return SubmitOutcome.FAILED

        return self._log_response(submission.n, response)

    @staticmethod
    def _log_response(n: int, response: requests.Response) -> SubmitOutcome:
        """Log the response and determine whether the submission was accepted.

        Returns:
            SubmitOutcome: The outcome of the submission.
        """
        try:
            body: Any = response.json()
        except ValueError:
            body = None

        if not isinstance(body, dict):
            logger.warning("mersenne.ca returned a malformed response for {}", n)
            return SubmitOutcome.FAILED

        fields = cast("dict[str, Any]", body)

        if warning := fields.get("warning"):
            logger.warning("mersenne.ca reported a warning for {}: {}", n, warning)

        if error := fields.get("error"):
            logger.error("mersenne.ca reported an error for {}: {}", n, error)
            return SubmitOutcome.REJECTED

        logger.debug("Reported factorization for {}", n)
        return SubmitOutcome.ACCEPTED
