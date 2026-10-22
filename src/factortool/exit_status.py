# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Exit statuses of factortool."""

from __future__ import annotations

from enum import IntEnum


class ExitStatus(IntEnum):
    """Exit statuses of factortool."""

    # The run finished, possibly cut short by a time limit.
    SUCCESS = 0

    # Python's uncaught exception status. It is never used deliberately by factortool, but instead indicates a bug or
    # other unanticipated problem.
    INTERNAL_ERROR = 1

    # Invalid arguments, configuration or state data, or unusable state or working directories.
    INVALID_INPUT = 2

    # Another instance is using the state or working directory.
    ALREADY_RUNNING = 3

    # An external tool failed repeatedly, so its installation is considered broken.
    TOOL_FAILURE = 4

    # The backend refused a request with a permanent HTTP error.
    PERMANENT_HTTP_ERROR = 5

    # State or results could not be saved, or the backend could not be shut down.
    CLEANUP_FAILED = 6

    # The run was interrupted or asked to end by a signal. By convention, this is 128 plus the number of SIGINT.
    INTERRUPTED = 130
