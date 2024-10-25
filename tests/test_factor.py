# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

from factortool.number import factor_tf


def test_tf() -> None:
    n = 15825810
    assert factor_tf(n) == [2, 3, 5, 7, 11, 13, 17, 31]
