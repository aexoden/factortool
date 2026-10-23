# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for factortool utility functions."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import time

from typing import TYPE_CHECKING
from unittest.mock import Mock

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

import pytest

from loguru import logger

from factortool.exit_status import ExitStatus
from factortool.lock import LOCK_FILE_NAME, lock_directories
from factortool.util import (
    NUMBER_DISPLAY_WIDTH,
    WORK_DIR_MARKER,
    ConfigArguments,
    LogLevel,
    format_number,
    get_work_dir,
    remove_stale_work_dirs,
    rewrite_yafu_ini,
    setup_logger,
)

from .helpers import HEARTBEAT_QUIET, wait_for_heartbeat


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (7, "7 <1>"),
        (123456789, "123456789 <9>"),
        (int("1234567890" * 2 + "1234567"), "123456789012345678901234567 <27>"),
        (int("1234567890" * 2 + "12345678"), "123456789012...789012345678 <28>"),
        (int("1234567890" * 3 + "123"), "123456789012...234567890123 <33>"),
        (int("9" * 150), "999999999999...99999999999 <150>"),
    ],
)
def test_format_number(n: int, expected: str) -> None:
    """Test that every number shows its digit count, and that long numbers are truncated to the maximum width."""
    assert format_number(n) == expected


@pytest.mark.parametrize("digits", [28, 31, 32, 33, 99, 100, 999, 1000, 4300])
def test_format_number_truncates_to_display_width(digits: int) -> None:
    """Test that a number too long to display in full, including its digit count, fills exactly the display width."""
    formatted = format_number(10 ** (digits - 1) + 7)

    assert formatted.startswith("1")
    assert formatted.endswith(f"7 <{digits}>")
    assert len(formatted) == NUMBER_DISPLAY_WIDTH


def test_format_number_without_truncation() -> None:
    """Test that a number can be shown in full, still followed by its digit count."""
    n = int("1234567890" * 10)

    assert format_number(n, truncate=False) == f"{n} <100>"


def test_work_dir_is_isolated_and_removed(tmp_path: Path) -> None:
    """Test that each invocation gets a fresh directory that is cleaned up afterward."""
    base = tmp_path / "work"

    with get_work_dir(base, None, "test-") as work_dir:
        assert work_dir.is_dir()
        assert work_dir.parent == base
        assert (work_dir / WORK_DIR_MARKER).is_file()
        (work_dir / "siqs.dat").write_text("scratch", encoding="utf-8")

    assert not work_dir.exists()
    assert list(base.iterdir()) == []


def test_work_dir_nests_concurrent_invocations_separately(tmp_path: Path) -> None:
    """Tests that overlapping invocations do not share a directory."""
    base = tmp_path / "work"

    with get_work_dir(base, None, "test-") as first, get_work_dir(base, None, "test-") as second:
        assert first != second


def test_work_dir_places_ini(tmp_path: Path) -> None:
    """Tests that yafu.ini is made available inside the directory."""
    ini = tmp_path / "yafu.ini"
    ini.write_text("ggnfs_dir=/opt/ggnfs/\n", encoding="utf-8")

    with get_work_dir(tmp_path / "work", ini, "yafu-") as work_dir:
        assert (work_dir / "yafu.ini").read_text(encoding="utf-8") == "ggnfs_dir=/opt/ggnfs/\n"


def make_marked_dir(path: Path) -> None:
    """Create a directory carrying the marker of a factortool working directory."""
    path.mkdir(parents=True)
    (path / WORK_DIR_MARKER).touch()


def test_stale_work_dirs_are_removed_and_everything_else_is_kept(tmp_path: Path) -> None:
    """Test that only the tools' leftover working directories are removed, along with their contents.

    A directory is only taken to be a leftover if it has the marker, so a similarly named one kept under work_path,
    such as a tool's installation, is left alone.
    """
    base = tmp_path / "work"
    outside = tmp_path / "outside"
    make_marked_dir(outside)
    make_marked_dir(base / "yafu-abc")
    (base / "yafu-abc" / "nested").mkdir()
    (base / "yafu-abc" / "nested" / "siqs.dat").write_text("scratch", encoding="utf-8")
    make_marked_dir(base / "cado-nfs-def")
    (base / "cado-nfs-def" / "c60.db").write_text("scratch", encoding="utf-8")
    (base / "cado-nfs-2.3.0").mkdir()
    (base / "cado-nfs-2.3.0" / "cado-nfs.py").write_text("kept", encoding="utf-8")
    make_marked_dir(base / "other")
    (base / "yafu-file").write_text("kept", encoding="utf-8")
    (base / "factortool.lock").write_text("", encoding="utf-8")
    kept = ["cado-nfs-2.3.0", "factortool.lock", "other", "yafu-file"]

    # Creating a symbolic link may need extra privileges on Windows.
    with contextlib.suppress(OSError):
        (base / "yafu-link").symlink_to(outside, target_is_directory=True)
        kept.append("yafu-link")

    remove_stale_work_dirs(base)

    assert sorted(path.name for path in base.iterdir()) == kept
    assert (base / "cado-nfs-2.3.0" / "cado-nfs.py").is_file()
    assert outside.is_dir()


def test_a_stale_work_dir_that_cannot_be_removed_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that a failure to remove one stale directory is not fatal and doesn't keep the others."""
    base = tmp_path / "work"
    make_marked_dir(base / "yafu-abc")
    make_marked_dir(base / "cado-nfs-def")
    rmtree = shutil.rmtree
    warning = Mock()
    monkeypatch.setattr("factortool.util.logger.warning", warning, raising=True)

    def fail_on_yafu(path: Path) -> None:
        if path.name.startswith("yafu-"):
            raise PermissionError(path)

        rmtree(path)

    monkeypatch.setattr("factortool.util.shutil.rmtree", fail_on_yafu, raising=True)

    remove_stale_work_dirs(base)

    assert [path.name for path in base.iterdir()] == ["yafu-abc"]
    assert warning.call_args.args[1] == base / "yafu-abc"


# An instance that is killed outright, leaving its tool running in the working directory it was given.
_DYING_INSTANCE = """
import sys
from pathlib import Path
from factortool.lock import lock_directories
from factortool.tools import run_tool
from factortool.util import YAFU_WORK_DIR_PREFIX, get_work_dir

base = Path(sys.argv[1])

with lock_directories([base]), get_work_dir(base, None, YAFU_WORK_DIR_PREFIX) as work_dir:
    run_tool([sys.executable, "-c", sys.argv[2], sys.argv[3]], work_dir, 1)
"""

# Records its process ID beside its heartbeat, then beats until it is killed or half a minute has passed.
_ORPHANED_TOOL = """
import os, sys, time
from pathlib import Path

heartbeat = Path(sys.argv[1])
heartbeat.with_suffix(".pid").write_text(str(os.getpid()))

for _ in range(600):
    with heartbeat.open("a") as f:
        f.write(".")
    time.sleep(0.05)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot remove a directory a running process is in")
def test_a_killed_instance_has_its_directories_removed_while_its_tools_still_run(tmp_path: Path) -> None:
    """Test that the next run removes a killed instance's working directory even though its tool is still running.

    A tool in its own process group outlives an instance that is killed outright, and the lock does not, but nothing
    remains to collect the tool's results, so its directory is of no further use.
    """
    # Work around mypy not understanding that this test is skipped on Windows.
    if sys.platform == "win32":
        return

    base = tmp_path / "work"
    heartbeat = tmp_path / "heartbeat"
    command = [sys.executable, "-c", _DYING_INSTANCE, str(base), _ORPHANED_TOOL, str(heartbeat)]

    try:
        with subprocess.Popen(command) as instance:  # ruff: ignore[subprocess-without-shell-equals-true]
            wait_for_heartbeat(heartbeat)
            instance.kill()

        size = heartbeat.stat().st_size
        time.sleep(HEARTBEAT_QUIET)

        assert heartbeat.stat().st_size > size

        with lock_directories([base]):
            remove_stale_work_dirs(base)

        assert sorted(path.name for path in base.iterdir()) == [LOCK_FILE_NAME]
    finally:
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(heartbeat.with_suffix(".pid").read_text()), signal.SIGKILL)


def test_stale_work_dirs_tolerate_a_missing_base(tmp_path: Path) -> None:
    """Test that a working directory that doesn't exist yet has nothing to remove."""
    remove_stale_work_dirs(tmp_path / "work")

    assert not (tmp_path / "work").exists()


def test_rewrite_yafu_ini_makes_relative_paths_absolute(tmp_path: Path) -> None:
    """Tests that relative paths in yafu.ini are converted to absolute paths."""
    result = rewrite_yafu_ini("ggnfs_dir=factor/lasieve5_64/bin/avx512/\n", tmp_path)

    assert result == f"ggnfs_dir={tmp_path / 'factor' / 'lasieve5_64' / 'bin' / 'avx512'}{os.sep}\n"


def test_rewrite_yafu_ini_preserves_everything_else(tmp_path: Path) -> None:
    """Tests that absolute paths, comments, non-path options and valueless flags are preserved."""
    original = (
        "% ggnfs_dir=commented/out/\n"
        "terse\n"
        "plan=normal\n"
        "B1ecm=11000\n"
        "cado_dir=/home/user/cado-nfs/\n"
        "tune_info=CPU,LINUX64,1.0,2.0\n"
    )

    assert rewrite_yafu_ini(original, tmp_path) == original


def test_rewrite_yafu_ini_handles_file_options_without_trailing_separator(tmp_path: Path) -> None:
    """Tests that file-valued options aren't given a spurious trailing separator."""
    result = rewrite_yafu_ini("ecm_path=../ecm-install/bin/ecm\n", tmp_path / "yafu")

    assert result == f"ecm_path={tmp_path / 'ecm-install' / 'bin' / 'ecm'}\n"


def test_get_work_dir_tolerates_missing_ini(tmp_path: Path) -> None:
    """Tests that a missing yafu.ini is not fatal."""
    with get_work_dir(tmp_path / "work", tmp_path / "absent.ini", "test-") as work_dir:
        assert not (work_dir / "yafu.ini").exists()


@pytest.fixture
def restored_logger() -> Iterator[None]:
    """Restore loguru's default handler after a test replaces the handlers, rather than leave one bound to capsys."""
    yield
    logger.remove()

    if sys.__stderr__ is not None:
        logger.add(sys.__stderr__)


@pytest.mark.usefixtures("restored_logger")
@pytest.mark.parametrize(
    ("level", "shown"), [(None, ["INFO", "WARNING"]), ("WARNING", ["WARNING"]), ("DEBUG", ["DEBUG", "INFO", "WARNING"])]
)
def test_setup_logger_shows_messages_at_or_above_the_level_on_stderr(
    capsys: pytest.CaptureFixture[str], level: LogLevel | None, shown: list[str]
) -> None:
    """Test that log messages go to standard error, filtered by the level, which defaults to INFO."""
    setup_logger(level)

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")

    captured = capsys.readouterr()
    assert not captured.out
    assert [line.split(" | ")[2].strip() for line in captured.err.splitlines()] == shown


@pytest.mark.usefixtures("restored_logger")
def test_setup_logger_replaces_the_previous_setup(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that setting up the logger again changes the level rather than adding a second handler."""
    setup_logger("ERROR")
    setup_logger("INFO")

    logger.info("info message")

    assert len(capsys.readouterr().err.splitlines()) == 1


@pytest.mark.parametrize(("argument", "expected"), [([], None), (["--log_level", "debug"], "DEBUG")])
def test_config_arguments_accept_a_log_level_in_any_case(argument: list[str], expected: str | None) -> None:
    """Test that the log level option is optional, and accepts a level in any case."""
    assert ConfigArguments().parse_args(argument).log_level == expected


def test_config_arguments_reject_an_unknown_log_level(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that an unknown log level is reported as invalid usage."""
    with pytest.raises(SystemExit) as raised:
        ConfigArguments().parse_args(["--log_level", "verbose"])

    assert raised.value.code == ExitStatus.INVALID_INPUT
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(("argument", "expected"), [([], None), (["--max_threads", "8"], 8)])
def test_config_arguments_accept_a_thread_count(argument: list[str], expected: int | None) -> None:
    """Test that the thread count option is optional, leaving the configured setting unchanged by default."""
    assert ConfigArguments().parse_args(argument).max_threads == expected
