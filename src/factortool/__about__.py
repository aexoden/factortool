# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024-2026 Jason Lynch <jason@aexoden.com>
"""Factortool package information."""

from __future__ import annotations

import importlib.metadata

PROJECT_URL = "https://github.com/aexoden/factortool"

# Reported when the version cannot be determined. Running via "uv run" always installs it, so this is only reached if
# importing directly from a source tree.
UNKNOWN_VERSION = "0.0.0+unknown"


def _get_version() -> str:
    """Read the version recorded in the package metadata.

    Returns:
        str: The version string of the package.
    """
    try:
        return importlib.metadata.version("factortool")
    except importlib.metadata.PackageNotFoundError:
        return UNKNOWN_VERSION


__version__ = _get_version()
