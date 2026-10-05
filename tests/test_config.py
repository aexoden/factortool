# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for configuration validation."""

from __future__ import annotations

import pytest

from pydantic import ValidationError

from .helpers import make_config


@pytest.mark.parametrize(
    "user_agent",
    ["", "custom/1.0", "custom/1.0 (Account; +https://example.com)", "custom/1.0 ~"],
)
def test_user_agent_accepts_valid_overrides(user_agent: str) -> None:
    """Accept empty or printable ASCII overrides without changing their value."""
    assert make_config(user_agent=user_agent).user_agent == user_agent


@pytest.mark.parametrize(
    "user_agent",
    [
        pytest.param("用户", id="unicode"),
        pytest.param("Renée", id="non-ascii-latin1"),
        pytest.param("custom\rvalue", id="carriage-return"),
        pytest.param("custom\nvalue", id="newline"),
        pytest.param("custom\r\nvalue", id="crlf"),
        pytest.param("custom\tvalue", id="tab"),
        pytest.param("custom\x00value", id="null"),
        pytest.param("custom\x1fvalue", id="control-character"),
        pytest.param("custom\x7fvalue", id="delete"),
        pytest.param(" custom/1.0", id="leading-space"),
        pytest.param("custom/1.0 ", id="trailing-space"),
        pytest.param(" ", id="whitespace-only"),
    ],
)
def test_user_agent_rejects_invalid_overrides(user_agent: str) -> None:
    """Reject unsafe overrides through configuration validation."""
    with pytest.raises(ValidationError, match="user_agent must contain printable ASCII only") as exc_info:
        make_config(user_agent=user_agent)

    assert [error["loc"] for error in exc_info.value.errors()] == [("user_agent",)]
