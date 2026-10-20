# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for factortool utility functions."""

from __future__ import annotations

import os

from typing import TYPE_CHECKING

import pytest

from factortool.util import NUMBER_DISPLAY_WIDTH, format_number, get_work_dir, rewrite_yafu_ini

if TYPE_CHECKING:
    from pathlib import Path


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
