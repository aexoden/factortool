# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""The factoring work done on a single composite."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from factortool.constants import ECM_MAX_LEVEL, MIN_FACTORABLE

# The methods run once on each composite before ECM.
PretestMethod = Literal["tf", "rho", "pm1"]


class CompositeProgress(BaseModel):
    """The standard-mode methods already run on a composite.

    A composite's factors inherit its progress, as each method is run once per composite and the work it did on the
    whole also covers the factors of its parts.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    n: int = Field(ge=MIN_FACTORABLE)

    # The methods run on the composite before ECM.
    completed: frozenset[PretestMethod] = frozenset()

    # The highest ECM level run on the composite, or zero if none has been.
    ecm_level: int = Field(default=0, ge=0, le=ECM_MAX_LEVEL)

    def after(self, method: PretestMethod) -> CompositeProgress:
        """The progress once the given method has also been run.

        Returns:
            CompositeProgress: The updated progress.
        """
        return self.model_copy(update={"completed": self.completed | {method}})

    def for_factor(self, n: int) -> CompositeProgress:
        """The progress inherited by a factor of this composite.

        Returns:
            CompositeProgress: The same progress, for the given factor.
        """
        return self.model_copy(update={"n": n})

    def merged(self, other: CompositeProgress) -> CompositeProgress:
        """The combined progress of two records for the same composite.

        Returns:
            CompositeProgress: The progress with the methods of both and the higher ECM level.
        """
        return self.model_copy(
            update={"completed": self.completed | other.completed, "ecm_level": max(self.ecm_level, other.ecm_level)}
        )
