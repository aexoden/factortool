# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Protocol describing a source of composite numbers and a destination for factorizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Collection

    from factortool.number import Number


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
