# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Shared helpers for factortool tests."""

from __future__ import annotations

from pathlib import Path

from factortool.config import Config
from factortool.number import Number
from factortool.stats import FactoringStats


def make_config(**overrides: object) -> Config:
    """Build a complete configuration the way the application does, via pydantic, so validation matches.

    Returns:
        Config: The validated configuration.
    """
    return Config.model_validate(
        {
            "assignment_state_path": "assignment_state.json",
            "backend": "factordb",
            "batch_state_path": "batch_state.json",
            "cado_nfs_path": "cado-nfs.py",
            "factordb_cooldown_period": 1.0,
            "factordb_response_path": "response.html",
            "factordb_session_path": "session.json",
            "factordb_username": "",
            "factordb_password": "",
            "factoring_mode": "standard",
            "gimps_login": "",
            "max_siqs_digits": 100,
            "max_threads": 1,
            "mersenne_ca_cooldown_period": 1.0,
            "result_output_path": "results",
            "stats_path": "stats.json",
            "user_agent": "",
            "work_path": "work",
            "yafu_path": "yafu",
            "yafu_ini_path": None,
            **overrides,
        }
    )


def make_number(n: int) -> Number:
    """Build a Number detached from any backend.

    Returns:
        Number: The constructed Number instance.
    """
    return Number(n, make_config(), FactoringStats(Path("stats.json"), read_only=True), None)
