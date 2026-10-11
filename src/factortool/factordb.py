# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>
"""Interface for interacting with FactorDB through its JSON-RPC API. See https://factordb.com/api.php."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn, override

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

import requests

from loguru import logger
from pydantic import BaseModel, TypeAdapter

from factortool.backend import BaseBackend, SubmitOutcome
from factortool.http import PermanentHttpError
from factortool.interrupt import Interrupted

if TYPE_CHECKING:
    from factortool.backend import FetchCriteria
    from factortool.config import Config
    from factortool.interrupt import InterruptState
    from factortool.stats import FactoringStats
    from factortool.submissions import Submission

API_URL = "https://factordb.com/rpc"

# The most rows list_by_type returns for a single request.
MAX_FETCH_COUNT = 1000

# The most report_factors calls to send in a single batch. FactorDB runs the calls in a batch sequentially and applies
# per-client quotas.
SUBMIT_BATCH_SIZE = 25

# The timeout for a batch of report_factors calls and the extra allowed for each call beyond the first.
SUBMIT_TIMEOUT = 30.0
SUBMIT_TIMEOUT_PER_CALL = 5.0

# JSON-RPC errors indicating a malformed or unsupported request that retrying cannot fix.
PERMANENT_RPC_ERROR_CODES = frozenset({-32700, -32600, -32601, -32602})

# The error report_factors returns when none of the submitted factors divides the target.
REJECTED_FACTORS_ERROR = (-32000, "no valid factor")


class RpcError(requests.RequestException):
    """An error reported by FactorDB in a JSON-RPC response."""

    def __init__(self, code: int, message: str) -> None:
        """Initialize the error from the code and message reported by FactorDB."""
        super().__init__(f"FactorDB reported error {code}: {message}")
        self.code = code
        self.message = message


class RpcErrorDetail(BaseModel):
    """The error object of a failed JSON-RPC call."""

    code: int
    message: str


class RpcResponse(BaseModel):
    """A single JSON-RPC response, carrying either a result or an error."""

    id: int | None
    result: Any = None
    error: RpcErrorDetail | None = None


RPC_RESPONSES = TypeAdapter(list[RpcResponse])


class ListedNumber(BaseModel):
    """A row returned by list_by_type. The preview is the full decimal only for small numbers."""

    fid: int
    digits: int
    preview: str


class ListResult(BaseModel):
    """The result of list_by_type."""

    rows: list[ListedNumber]


class NumberRecord(BaseModel):
    """The result of get_number. The decimal is omitted if FactorDB considers the number too large."""

    decimal: str | None = None


class ReportResult(BaseModel):
    """The result of report_factors, returned if at least one factor was accepted."""

    status: str
    credited: int = 0


class Identity(BaseModel):
    """The result of whoami."""

    found: bool
    login: str = ""


def raise_rpc_error(error: RpcErrorDetail) -> NoReturn:
    """Raise the exception corresponding to a JSON-RPC error.

    Raises:
        PermanentHttpError: If the error indicates a request that retrying cannot fix.
        RpcError: For any other error.
    """
    if error.code in PERMANENT_RPC_ERROR_CODES:
        msg = f"FactorDB reported error {error.code}: {error.message}"
        raise PermanentHttpError(msg)

    raise RpcError(error.code, error.message)


def is_rejection(error: Exception) -> bool:
    """Determine whether an error is FactorDB rejecting every submitted factor.

    Returns:
        bool: True if the error reports that none of the factors divide the target.
    """
    return isinstance(error, RpcError) and (error.code, error.message) == REJECTED_FACTORS_ERROR


def parse_rpc_batch(body: object, count: int) -> list[RpcResponse]:
    """Extract the individual responses from a batch JSON-RPC response, in request order.

    Errors reported for individual calls are left in their responses rather than raised.

    Returns:
        list[RpcResponse]: The response to each call, in the order the calls were made.

    Raises:
        ValueError: If the response is malformed.
        PermanentHttpError: If the batch as a whole failed in a way retrying cannot fix.
        RpcError: If the batch as a whole failed for any other reason.
    """
    if isinstance(body, dict):
        response = RpcResponse.model_validate(body)

        if response.error is not None:
            raise_rpc_error(response.error)

        msg = "FactorDB answered a batch of JSON-RPC calls with a single result"
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] (The response was malformed)

    batch = RPC_RESPONSES.validate_python(body)

    for response in batch:
        if response.id is None:
            if response.error is not None:
                raise_rpc_error(response.error)

            msg = "FactorDB returned a JSON-RPC result without an id"
            raise ValueError(msg)

    responses = {response.id: response for response in batch if response.id is not None}

    if sorted(responses) != list(range(count)):
        msg = f"FactorDB returned JSON-RPC calls {sorted(responses)} when {count} were made"
        raise ValueError(msg)

    return [responses[i] for i in range(count)]


def rpc_result(response: RpcResponse) -> object:
    """Extract the result of a single JSON-RPC call.

    Returns:
        object: The result of the call.

    Raises:
        ValueError: If the response carries neither a result nor an error.
        PermanentHttpError: If the call failed in a way retrying cannot fix.
        RpcError: If the call failed for any other reason.
    """
    if response.error is not None:
        raise_rpc_error(response.error)

    if "result" not in response.model_fields_set:
        msg = f"FactorDB returned neither a result nor an error for JSON-RPC call {response.id}"
        raise ValueError(msg)

    return response.result


def parse_rpc_responses(body: object, count: int) -> list[Any]:
    """Extract the results from a batch JSON-RPC response, in request order.

    Returns:
        list[Any]: The result of each call, in the order the calls were made.

    Raises:
        ValueError: If the response is malformed.
        PermanentHttpError: If a call failed in a way retrying cannot fix.
        RpcError: If a call failed for any other reason.
    """
    return [rpc_result(response) for response in parse_rpc_batch(body, count)]


class FactorDB(BaseBackend):
    """Interface for interacting with FactorDB."""

    name = "FactorDB"

    # FactorDB does not currently have any sort of reservation system.
    assigns_work = False
    assignment_lifetime = 0.0

    submit_batch_size = SUBMIT_BATCH_SIZE

    def __init__(self, config: Config, stats: FactoringStats, interrupts: InterruptState | None = None) -> None:
        """Initialize the FactorDB interface, verifying any configured API token.

        Raises:
            Interrupted: If an interrupt arrived while waiting to verify the configured API token.
            PermanentHttpError: If FactorDB does not recognize the configured API token.
            requests.RequestException: If the configured API token could not be verified.
        """
        super().__init__(config, stats, config.factordb_cooldown_period, "", interrupts)

        self._signed_in = False
        self._credited_count = 0

        if not config.factordb_api_token:
            logger.warning("No FactorDB API token is configured; results will be submitted anonymously")
            self._start_submitting()
            return

        self._http_client.session.headers["X-Fdb-User-Token"] = config.factordb_api_token

        try:
            (result,) = self._rpc(
                [("whoami", {"session": config.factordb_api_token})],
                timeout=5.0,
                wait=self._interrupts.wait,
                max_attempts=5,
            )
            identity = Identity.model_validate(result)
        except Interrupted, PermanentHttpError:
            self.close()
            raise
        except (requests.RequestException, ValueError) as e:
            self.close()
            msg = f"Unable to verify the FactorDB API token: {e}"
            raise requests.RequestException(msg) from e

        if not identity.found:
            self.close()
            msg = "FactorDB does not recognize the configured API token"
            raise PermanentHttpError(msg)

        logger.info("Signed in to FactorDB as {}", identity.login)
        self._signed_in = True
        self._start_submitting()

    def _rpc(
        self,
        calls: Sequence[tuple[str, Mapping[str, object]]],
        *,
        timeout: float,
        wait: Callable[[float], bool],
        max_attempts: int | None = None,
    ) -> list[Any]:
        """Make a batch of JSON-RPC calls.

        HTTP failures are retried until the wait function says to abandon the request unless max_attempts is given.
        Errors reported by FactorDB for individual calls are not retried.

        Returns:
            list[Any]: The result of each call, in the order the calls were made.
        """
        responses = self._rpc_batch(calls, timeout=timeout, wait=wait, max_attempts=max_attempts)
        return [rpc_result(response) for response in responses]

    def _rpc_batch(
        self,
        calls: Sequence[tuple[str, Mapping[str, object]]],
        *,
        timeout: float,
        wait: Callable[[float], bool],
        max_attempts: int | None = None,
    ) -> list[RpcResponse]:
        """Make a batch of JSON-RPC calls, leaving any errors reported for individual calls in their responses.

        Returns:
            list[RpcResponse]: The raw responses for each call, in the order the calls were made.
        """
        payload = [
            {"jsonrpc": "2.0", "id": i, "method": method, "params": params} for i, (method, params) in enumerate(calls)
        ]

        if max_attempts is None:
            response = self._service_request("POST", API_URL, json=payload, timeout=timeout, wait=wait)
        else:
            response = self._http_client.request(
                "POST", API_URL, json=payload, timeout=timeout, max_attempts=max_attempts, wait=wait
            )

        return parse_rpc_batch(response.json(), len(calls))

    def _request_composites(self, criteria: FetchCriteria) -> list[int]:
        """Request composites from FactorDB, smallest first.

        All criteria are respected. Requests are capped at MAX_FETCH_COUNT numbers.

        Returns:
            list[int]: The composite numbers returned by FactorDB.

        Raises:
            ValueError: If the response is malformed.
        """
        params: dict[str, object] = {
            "table": "C",
            "min_digits": criteria.min_digits,
            "offset": criteria.skip_count,
            "limit": min(criteria.count, MAX_FETCH_COUNT),
        }

        if criteria.max_digits is not None:
            params["max_digits"] = criteria.max_digits

        (result,) = self._rpc([("list_by_type", params)], timeout=10.0, wait=self._interrupts.wait)
        rows = ListResult.model_validate(result).rows

        # Larger numbers are only previewed.
        truncated = [row for row in rows if len(row.preview) != row.digits]
        decimals: dict[int, str | None] = {}

        if truncated:
            calls = [("get_number", {"target": {"id": row.fid}, "decimal": True}) for row in truncated]
            results = self._rpc(calls, timeout=30.0, wait=self._interrupts.wait)
            decimals = {
                row.fid: NumberRecord.model_validate(result).decimal
                for row, result in zip(truncated, results, strict=True)
            }

        composites: list[int] = []

        for row in rows:
            decimal = decimals.get(row.fid, row.preview)

            if decimal is None:
                logger.warning("Skipping FactorDB number {}, as its decimal expansion was omitted", row.fid)
                continue

            if len(decimal) != row.digits:
                logger.warning(
                    "Skipping FactorDB number {}, as its decimal expansion has {} digits rather than {}",
                    row.fid,
                    len(decimal),
                    row.digits,
                )
                continue

            composites.append(int(decimal))

        return composites

    def _submit_number(self, submission: Submission) -> SubmitOutcome:
        """Make a single attempt to submit the distinct prime factors of a number to FactorDB.

        Returns:
            SubmitOutcome: The outcome of the submission attempt.
        """
        return self._submit_numbers([submission])[0]

    @staticmethod
    def _factors_to_report(submission: Submission) -> list[int]:
        """Select the factors of a number that should be reported to FactorDB.

        These are its distinct prime factors, excluding the trivial largest factor if there are no composite factors.
        The removal of the trivial largest factor occurs before duplicate factors are removed, ensuring that if the
        largest factor is duplicated, it is nonetheless submitted.

        Returns:
            list[int]: The factors to report, which may be empty.
        """
        factors = sorted(submission.prime_factors)

        # If there are no composite factors, avoid sending the trivial largest factor.
        if len(submission.composite_factors) == 0:
            factors.pop()

        # Remove duplicate factors to avoid redundant submissions.
        factors = sorted(set(factors))

        # It shouldn't really happen, but if the only prime factor is the number itself, skip it.
        if factors == [submission.n]:
            return []

        return factors

    @override
    def _submit_numbers(self, submissions: Sequence[Submission]) -> list[SubmitOutcome]:
        """Make a single attempt to submit the distinct prime factors of each number to FactorDB in a single batch.

        HTTP failures are retried by the underlying client until the submission is abandoned. Errors and malformed
        responses are left for another attempt, unless they are permanent or a rejection of every factor.

        Returns:
            list[SubmitOutcome]: The outcomes of the submission attempts for each number.
        """
        outcomes = [SubmitOutcome.REJECTED] * len(submissions)
        reports = [(i, factors) for i, x in enumerate(submissions) if (factors := self._factors_to_report(x))]

        if not reports:
            return outcomes

        calls = [
            (
                "report_factors",
                {
                    "target": {"expr": str(submissions[i].n)},
                    "factors": [str(factor) for factor in factors],
                    "credit": self._signed_in,
                },
            )
            for i, factors in reports
        ]
        timeout = SUBMIT_TIMEOUT + SUBMIT_TIMEOUT_PER_CALL * (len(calls) - 1)

        try:
            responses = self._rpc_batch(calls, timeout=timeout, wait=self._submission_wait)
        except PermanentHttpError as e:
            logger.error("Discarding the factors for {} numbers that FactorDB will not accept: {}", len(calls), e)
            return outcomes
        except (requests.RequestException, ValueError) as e:
            logger.warning("Error submitting factors for {} numbers to FactorDB: {}", len(calls), e)

            for i, _ in reports:
                outcomes[i] = SubmitOutcome.FAILED

            return outcomes

        for (i, factors), response in zip(reports, responses, strict=True):
            outcomes[i] = self._report_outcome(submissions[i].n, factors, response)

        return outcomes

    def _report_outcome(self, n: int, factors: Sequence[int], response: RpcResponse) -> SubmitOutcome:
        """Interpret the response to a single report_factors call.

        Returns:
            SubmitOutcome: The outcome of the submission attempt for the given number.
        """
        try:
            result = rpc_result(response)
            report = ReportResult.model_validate(result)
        except PermanentHttpError as e:
            logger.error("Error submitting factors {} for n={}: {}", factors, n, e)
            return SubmitOutcome.REJECTED
        except (requests.RequestException, ValueError) as e:
            if is_rejection(e):
                logger.warning("FactorDB did not accept factors {} for n={}: {}", factors, n, e)
                return SubmitOutcome.REJECTED

            logger.warning("Error submitting factors {} for n={}: {}", factors, n, e)
            return SubmitOutcome.FAILED

        if report.status == "C":
            logger.warning("FactorDB did not accept factors {} for n={}: {}", factors, n, result)
            return SubmitOutcome.REJECTED

        logger.debug("Submitted factors {} for n={}: {}", factors, n, result)
        self._credited_count += report.credited

        return SubmitOutcome.ACCEPTED

    @override
    def close(self) -> None:
        """Flush any pending submissions, then log how many factors FactorDB credited to the account."""
        super().close()

        if not self._signed_in or self.get_successful_submission_count() == 0:
            return

        # A submission repeated after its response was lost reports no credit for factors credited the first time.
        logger.info("FactorDB credited at least {} factors to your account", self._credited_count)
