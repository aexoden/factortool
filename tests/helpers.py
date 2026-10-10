# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Shared helpers for factortool tests."""

from __future__ import annotations

import contextlib
import sys
import time

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

from factortool.config import Config
from factortool.interrupt import InterruptState
from factortool.number import Number
from factortool.stats import FactoringStats


def make_config(**overrides: object) -> Config:
    """Build a configuration the way the application does, via pydantic, so validation and defaults match.

    Returns:
        Config: The validated configuration.
    """
    return Config.model_validate(
        {"backend": "factordb", "cado_nfs_path": "cado-nfs.py", "max_threads": 1, "yafu_path": "yafu", **overrides}
    )


def make_number(n: int) -> Number:
    """Build a Number detached from any backend.

    Returns:
        Number: The constructed Number instance.
    """
    return Number(n, make_config(), FactoringStats(Path("stats.json"), read_only=True), None)


@contextlib.contextmanager
def installed_interrupts() -> Generator[InterruptState]:
    """Provide an interrupt state that is responding to signals, for the duration of the block.

    Yields:
        InterruptState: The installed interrupt state.
    """
    interrupts = InterruptState()
    interrupts.install()

    try:
        yield interrupts
    finally:
        interrupts.uninstall()


# Generous upper bound on how long a killed tool's helper may take to stop.
TOOL_STOP_TIMEOUT = 5.0

# How long a heartbeat file must stay unchanged before its writer is considered stopped.
HEARTBEAT_QUIET = 0.3

# Gives up on its own after half a minute, so we don't wait indefinitely.
_HEARTBEAT = """
import sys, time
for _ in range(600):
    with open(sys.argv[1], "a") as f:
        f.write(".")
    time.sleep(0.05)
"""

_TOOL = f"""
import subprocess, sys
subprocess.Popen([sys.executable, "-c", {_HEARTBEAT!r}, sys.argv[1]]).wait()
"""


def make_tool_with_helper(heartbeat_path: Path) -> list[str]:
    """Build a command for a stand-in tool that starts a helper process, as YAFU and CADO-NFS do.

    The helper appends to heartbeat_path until it is killed, so a test can tell whether it outlived the tool.

    Returns:
        list[str]: The command to run the stand-in tool with the helper.
    """
    return [sys.executable, "-c", _TOOL, str(heartbeat_path)]


def wait_for_heartbeat(heartbeat_path: Path) -> None:
    """Wait until a stand-in tool's helper has started."""
    deadline = time.monotonic() + TOOL_STOP_TIMEOUT

    while not heartbeat_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)


def heartbeat_stopped(heartbeat_path: Path) -> bool:
    """Wait for a stand-in tool's helper to stop.

    Returns:
        bool: True if the helper stopped writing its heartbeat within TOOL_STOP_TIMEOUT.
    """
    deadline = time.monotonic() + TOOL_STOP_TIMEOUT
    size = heartbeat_path.stat().st_size

    while time.monotonic() < deadline:
        time.sleep(HEARTBEAT_QUIET)
        previous, size = size, heartbeat_path.stat().st_size

        if size == previous:
            return True

    return False
