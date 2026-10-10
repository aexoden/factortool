# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Shared fixtures for factortool tests."""

from __future__ import annotations

import pytest

from factortool.tools import tool_failures


# Every test that reaches a tool would otherwise have to ask for this.
@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def _fresh_tool_failures() -> None:
    """Keep the tool failures of one test from counting toward another's."""
    tool_failures.reset()
