# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

from pathlib import Path

from factortool.number import factor_tf
from factortool.stats import FactoringStats


def test_tf() -> None:
    n = 15825810
    stats = FactoringStats(Path("stats.json"), read_only=True)
    assert factor_tf(n, stats) == [2, 3, 5, 7, 11, 13, 17, 31]
