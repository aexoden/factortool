# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for configuration validation."""

from __future__ import annotations

import json
import sys

from pathlib import Path

import pytest

from loguru import logger
from pydantic import ValidationError

from factortool.config import Config, read_config

from .helpers import make_config

CADO_NFS = {"cado_nfs_path": "cado-nfs.py", "use_nfs_cado": True}
BOTH_NFS = {**CADO_NFS, "use_nfs_yafu": True}

REQUIRED_SETTINGS = {
    "backend": "factordb",
    "max_threads": 1,
    "yafu_path": "yafu",
}


def write_config(tmp_path: Path, settings: dict[str, object]) -> Path:
    """Write a configuration file with the given settings.

    Returns:
        Path: The path of the configuration file.
    """
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path


def read_config_logging(path: Path) -> tuple[Config, list[str]]:
    """Read a configuration file, capturing the log messages.

    Returns:
        tuple[Config, list[str]]: The configuration and the log messages.
    """
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")

    try:
        return read_config(path), messages
    finally:
        logger.remove(sink)


def read_invalid_config(path: Path) -> list[str]:
    """Read a configuration file that should be rejected, capturing the log messages.

    Returns:
        list[str]: The log messages.
    """
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")

    try:
        with pytest.raises(SystemExit) as exc_info:
            read_config(path)
    finally:
        logger.remove(sink)

    assert exc_info.value.code == 1
    return messages


def test_config_can_be_constructed_directly() -> None:
    """Test that constructing a configuration directly validates and applies defaults like reading a file does."""
    config = Config(backend="factordb", max_threads=1, yafu_path=Path("yafu"))

    assert config == make_config()

    with pytest.raises(ValidationError, match="gimps_login"):
        Config(backend="mersenne_ca", max_threads=1, yafu_path=Path("yafu"))


def test_config_is_immutable() -> None:
    """Test that a configuration cannot be changed once built, as it is shared across threads."""
    config = make_config()

    with pytest.raises(ValidationError, match="frozen"):
        config.max_threads = 2

    assert config.model_copy(update={"max_threads": 2}) == make_config(max_threads=2)
    assert config == make_config()


def test_mersenne_ca_requires_gimps_login() -> None:
    """Test that selecting the mersenne.ca backend without a GIMPS login is rejected."""
    with pytest.raises(ValidationError, match="gimps_login"):
        make_config(backend="mersenne_ca")

    assert make_config(backend="mersenne_ca", gimps_login="tester").gimps_login == "tester"


def test_read_config_rejects_mersenne_ca_without_gimps_login(tmp_path: Path) -> None:
    """Test that reading a mersenne.ca configuration file without a GIMPS login is rejected."""
    (message,) = read_invalid_config(write_config(tmp_path, {**REQUIRED_SETTINGS, "backend": "mersenne_ca"}))

    assert "gimps_login" in message


def test_cado_nfs_requires_its_path() -> None:
    """Test that the path to CADO-NFS is only required when CADO-NFS is enabled."""
    with pytest.raises(ValidationError, match="use_nfs_cado requires cado_nfs_path"):
        make_config(use_nfs_cado=True)

    assert make_config().cado_nfs_path is None
    assert make_config(**CADO_NFS).cado_nfs_path == Path("cado-nfs.py")


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("max_threads", 0),
        ("max_threads", -1),
        ("max_siqs_digits", 0),
        ("factordb_cooldown_period", -0.5),
        ("factordb_cooldown_period", float("inf")),
        ("mersenne_ca_cooldown_period", -0.5),
        ("mersenne_ca_cooldown_period", float("nan")),
    ],
)
def test_out_of_range_settings_are_rejected(setting: str, value: float) -> None:
    """Test that a setting outside its usable range is rejected rather than failing once work has been fetched."""
    with pytest.raises(ValidationError) as exc_info:
        make_config(**{setting: value})

    assert [error["loc"] for error in exc_info.value.errors()] == [(setting,)]


def test_read_config_rejects_out_of_range_settings(tmp_path: Path) -> None:
    """Test that reading a configuration file with an out of range setting names the setting."""
    (message,) = read_invalid_config(write_config(tmp_path, {**REQUIRED_SETTINGS, "max_threads": 0}))

    assert "max_threads" in message


def make_executable(path: Path) -> Path:
    """Create an empty file that may be executed.

    Returns:
        Path: The path of the file.
    """
    path.touch()
    path.chmod(0o755)
    return path


def test_tools_that_are_present_have_no_problems(tmp_path: Path) -> None:
    """Test that executable tools and an existing yafu.ini are accepted."""
    yafu_ini_path = tmp_path / "custom.ini"
    yafu_ini_path.touch()
    config = make_config(
        cado_nfs_path=make_executable(tmp_path / "cado-nfs.py"),
        use_nfs_cado=True,
        yafu_ini_path=yafu_ini_path,
        yafu_path=make_executable(tmp_path / "yafu"),
    )

    assert config.find_tool_problems() == []


def test_missing_tools_are_all_reported(tmp_path: Path) -> None:
    """Test that every missing tool and file is reported at once."""
    config = make_config(
        cado_nfs_path=tmp_path / "cado-nfs.py",
        use_nfs_cado=True,
        yafu_ini_path=tmp_path / "custom.ini",
        yafu_path=tmp_path / "yafu",
    )

    assert config.find_tool_problems() == [
        f"cado_nfs_path ({tmp_path / 'cado-nfs.py'}) is not an executable file",
        f"yafu_path ({tmp_path / 'yafu'}) is not an executable file",
        f"yafu_ini_path ({tmp_path / 'custom.ini'}) is not a file",
    ]


def test_cado_nfs_and_the_default_yafu_ini_are_only_needed_when_used(tmp_path: Path) -> None:
    """Test that a disabled CADO-NFS and the optional yafu.ini beside YAFU may both be missing."""
    config = make_config(cado_nfs_path=tmp_path / "cado-nfs.py", yafu_path=make_executable(tmp_path / "yafu"))

    assert config.find_tool_problems() == []


def test_a_tool_that_is_a_directory_is_a_problem(tmp_path: Path) -> None:
    """Test that a directory is not mistaken for a tool."""
    assert make_config(yafu_path=tmp_path).find_tool_problems() == [f"yafu_path ({tmp_path}) is not an executable file"]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no execute permission")
def test_a_tool_without_execute_permission_is_a_problem(tmp_path: Path) -> None:
    """Test that a file that cannot be executed is not accepted as a tool."""
    yafu_path = tmp_path / "yafu"
    yafu_path.touch(mode=0o644)

    assert make_config(yafu_path=yafu_path).find_tool_problems() == [
        f"yafu_path ({yafu_path}) is not an executable file"
    ]


def test_read_config_defaults_optional_settings(tmp_path: Path) -> None:
    """Test that reading a configuration file with only the required settings applies the default optional settings."""
    config, messages = read_config_logging(write_config(tmp_path, REQUIRED_SETTINGS))

    assert config == make_config()
    assert messages == []


@pytest.mark.parametrize("setting", sorted(REQUIRED_SETTINGS))
def test_read_config_names_missing_required_settings(tmp_path: Path, setting: str) -> None:
    """Test that reading a configuration file missing a required setting is rejected."""
    path = write_config(tmp_path, {key: value for key, value in REQUIRED_SETTINGS.items() if key != setting})

    assert read_invalid_config(path) == [f"Configuration file {path} is missing required setting(s): {setting}\n"]


def test_read_config_reports_missing_and_invalid_settings(tmp_path: Path) -> None:
    """Test that reading a configuration file reports both missing and invalid settings."""
    settings = {key: value for key, value in REQUIRED_SETTINGS.items() if key != "yafu_path"}

    missing, invalid = read_invalid_config(write_config(tmp_path, {**settings, "max_threads": "many"}))

    assert missing.endswith("missing required setting(s): yafu_path\n")
    assert "max_threads" in invalid


def test_read_config_warns_about_unrecognized_settings(tmp_path: Path) -> None:
    """Test that reading a configuration file with unrecognized settings logs warnings but does not reject the file."""
    path = write_config(tmp_path, {**REQUIRED_SETTINGS, "use_nfs": True, "max_siqs_digit": 90})

    config, messages = read_config_logging(path)

    assert config == Config.model_validate(REQUIRED_SETTINGS)
    assert messages == [
        "Ignoring unrecognized configuration setting: max_siqs_digit\n",
        "Ignoring unrecognized configuration setting: use_nfs\n",
    ]


def test_dist_config_matches_defaults() -> None:
    """Test that the distributed configuration file matches the default settings."""
    dist = json.loads((Path(__file__).parent.parent / "config.dist.json").read_text(encoding="utf-8"))
    defaults = Config.model_validate(REQUIRED_SETTINGS)

    assert dist.keys() == Config.model_fields.keys()
    assert {key: value for key, value in dist.items() if key not in REQUIRED_SETTINGS} == {
        key: value for key, value in defaults.model_dump(mode="json").items() if key not in REQUIRED_SETTINGS
    }


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


@pytest.mark.parametrize(
    ("digits", "overrides", "expected"),
    [
        pytest.param(56, {}, ("siqs",), id="below-nfs-cado-minimum"),
        pytest.param(57, CADO_NFS, ("siqs", "nfs_cado"), id="nfs-cado-minimum"),
        pytest.param(84, BOTH_NFS, ("siqs", "nfs_cado"), id="below-yafu-minimum"),
        pytest.param(85, BOTH_NFS, ("siqs", "nfs_cado", "nfs_yafu"), id="yafu-minimum"),
        pytest.param(101, BOTH_NFS, ("nfs_cado", "nfs_yafu"), id="above-max-siqs-digits"),
        pytest.param(101, {"use_nfs_cado": False, "use_nfs_yafu": True}, ("nfs_yafu",), id="cado-disabled"),
        pytest.param(101, {"use_nfs_cado": False, "use_nfs_yafu": False}, ("siqs",), id="siqs-as-last-resort"),
    ],
)
def test_final_methods_for_digits(digits: int, overrides: dict[str, object], expected: tuple[str, ...]) -> None:
    """Allow only the enabled final methods suited to the size, falling back to SIQS when none are."""
    assert make_config(**overrides).final_methods.for_digits(digits) == expected
