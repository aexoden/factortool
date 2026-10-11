# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for the directory locks."""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from factortool.lock import LOCK_FILE_NAME, DirectoryInUseError, lock_directories

# Holds the lock on a directory until it is killed.
_HOLDER = """
import sys
from pathlib import Path
from factortool.lock import lock_directories

with lock_directories([Path(sys.argv[1])]):
    print("locked", flush=True)
    sys.stdin.read()
"""


def test_locking_creates_missing_directories(tmp_path: Path) -> None:
    """Test that a directory that does not exist yet is created along with its lock file."""
    directory = tmp_path / "nested" / "state"

    with lock_directories([directory]):
        assert (directory / LOCK_FILE_NAME).is_file()


def test_a_locked_directory_is_refused_until_it_is_released(tmp_path: Path) -> None:
    """Test that a directory can only be locked once at a time."""
    with lock_directories([tmp_path]):
        with pytest.raises(DirectoryInUseError) as raised:
            lock_directories([tmp_path])

        assert raised.value.directory == tmp_path.resolve()

    with lock_directories([tmp_path]):
        pass


def test_a_directory_given_twice_is_locked_once(tmp_path: Path) -> None:
    """Test that the same directory under two spellings does not conflict with itself."""
    (tmp_path / "work").mkdir()

    with lock_directories([tmp_path, tmp_path / "work" / ".."]), pytest.raises(DirectoryInUseError):
        lock_directories([tmp_path])


def test_a_refused_directory_releases_the_ones_already_locked(tmp_path: Path) -> None:
    """Test that failing to lock one directory does not leave the earlier ones locked."""
    state, work = tmp_path / "state", tmp_path / "work"

    with lock_directories([work]):
        with pytest.raises(DirectoryInUseError) as raised:
            lock_directories([state, work])

        assert raised.value.directory == work.resolve()

        with lock_directories([state]):
            pass


def test_a_killed_instance_releases_its_lock(tmp_path: Path) -> None:
    """Test that another process's lock is honored, and that it does not outlive a process that is killed."""
    with subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", _HOLDER, str(tmp_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as holder:
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "locked"

            with pytest.raises(DirectoryInUseError):
                lock_directories([tmp_path])
        finally:
            holder.kill()

    with lock_directories([tmp_path]):
        pass
