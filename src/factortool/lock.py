# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Locks that keep two instances from using the same directory."""

from __future__ import annotations

import contextlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

from filelock import FileLock, Timeout

LOCK_FILE_NAME = "factortool.lock"


class DirectoryInUseError(Exception):
    """Raised when another instance holds the lock on a directory."""

    def __init__(self, directory: Path) -> None:
        """Initialize the error for the directory that is in use."""
        super().__init__(f"Another instance of factortool is using {directory}")
        self.directory = directory


def lock_directories(directories: Iterable[Path]) -> contextlib.ExitStack:
    """Take an exclusive lock on each directory, creating any that do not exist.

    The locks belong to the process, so they are released when it exits.

    Returns:
        contextlib.ExitStack: An exit stack that manages the acquired locks.

    Raises:
        DirectoryInUseError: If another instance holds the lock on any of the directories.
        OSError: If a directory or its lock file cannot be created.
    """
    with contextlib.ExitStack() as stack:
        for directory in dict.fromkeys(path.resolve() for path in directories):
            directory.mkdir(parents=True, exist_ok=True)
            lock = FileLock(directory / LOCK_FILE_NAME, blocking=False)

            try:
                lock.acquire()
            except Timeout as e:
                raise DirectoryInUseError(directory) from e

            stack.callback(lock.release)

        return stack.pop_all()
