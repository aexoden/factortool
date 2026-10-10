# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the analyzer."""

from __future__ import annotations

import sys

from unittest.mock import Mock

import pytest

from factortool.analyzer.main import main


@pytest.mark.parametrize("digits", ["0", "-5", "many"])
def test_an_unusable_digit_count_is_rejected(monkeypatch: pytest.MonkeyPatch, digits: str) -> None:
    """Test that a digit count no number could have is refused, with the status of a configuration error."""
    monkeypatch.setattr(sys, "argv", ["analyzer", "--digits", digits])
    monkeypatch.setattr("factortool.analyzer.main.setup_logger", Mock(), raising=True)

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 1
