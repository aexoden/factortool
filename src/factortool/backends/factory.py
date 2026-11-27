# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Selection of the backend the configuration names."""

from __future__ import annotations

from typing import TYPE_CHECKING

from factortool.backends.factordb import FactorDB
from factortool.backends.mersenne_ca import MersenneCA

if TYPE_CHECKING:
    from factortool.backends.base import Backend
    from factortool.config import Config
    from factortool.interrupt import InterruptState
    from factortool.stats import FactoringStats


def create_backend(config: Config, stats: FactoringStats, interrupts: InterruptState) -> Backend:
    """Build the backend selected in the configuration.

    Returns:
        Backend: The configured backend.
    """
    if config.backend == "mersenne_ca":
        return MersenneCA(config, stats, interrupts)

    return FactorDB(config, stats, interrupts)
