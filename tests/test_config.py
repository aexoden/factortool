# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jason Lynch <jason@aexoden.com>
"""Tests for configuration validation."""

from __future__ import annotations

import errno
import json
import os
import sys

from pathlib import Path

import pytest

from loguru import logger
from pydantic import ValidationError

from factortool.config import Config, load_config, read_config
from factortool.exit_status import ExitStatus

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

    assert exc_info.value.code == ExitStatus.INVALID_INPUT
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
        ("factordb_request_interval", -0.5),
        ("factordb_request_interval", float("inf")),
        ("mersenne_ca_request_interval", -0.5),
        ("mersenne_ca_request_interval", float("nan")),
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


def test_read_config_rejects_unrecognized_settings(tmp_path: Path) -> None:
    """Test that a setting that isn't recognized, such as a misspelled one, is an error rather than a silent default."""
    path = write_config(tmp_path, {**REQUIRED_SETTINGS, "use_nfs": True, "max_siqs_digit": 90})

    assert read_invalid_config(path) == [
        f"Configuration file {path} has unrecognized setting(s): max_siqs_digit, use_nfs\n"
    ]


def test_config_rejects_unrecognized_settings() -> None:
    """Test that a configuration built directly rejects an unrecognized setting as well."""
    with pytest.raises(ValidationError, match="max_siqs_digit"):
        make_config(max_siqs_digit=90)


def test_state_files_live_in_the_state_directory() -> None:
    """Test that every state file is found under state_path."""
    config = make_config(state_path="somewhere")

    assert config.assignment_state_path == Path("somewhere/assignment_state.json")
    assert config.batch_state_path == Path("somewhere/batch_state.json")
    assert config.pending_submissions_path == Path("somewhere/pending_submissions.jsonl")
    assert config.stats_path == Path("somewhere/stats.json")
    assert config.tool_state_path == Path("somewhere/tool_state.json")


def test_read_config_rejects_the_removed_state_file_settings(tmp_path: Path) -> None:
    """Test that a setting that used to name a state file is an error, rather than silently starting fresh state."""
    settings = {**REQUIRED_SETTINGS, "pending_submissions_path": "unsent.jsonl", "stats_path": "old.json"}

    (message,) = read_invalid_config(write_config(tmp_path, settings))

    assert "no longer supported: pending_submissions_path, stats_path" in message
    assert "state_path" in message


def test_read_config_rejects_the_renamed_cooldown_settings(tmp_path: Path) -> None:
    """Test that a cooldown setting is an error naming its replacement, as its meaning changed along with its name."""
    settings = {**REQUIRED_SETTINGS, "factordb_cooldown_period": 1.0}

    (message,) = read_invalid_config(write_config(tmp_path, settings))

    assert "no longer supported: factordb_cooldown_period" in message
    assert "Use factordb_request_interval instead" in message


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


@pytest.mark.parametrize(("log_level", "expected"), [("debug", "DEBUG"), ("Warning", "WARNING"), ("TRACE", "TRACE")])
def test_log_level_is_case_insensitive(log_level: str, expected: str) -> None:
    """Test that a log level is accepted in any case, and stored in the form loguru expects."""
    assert make_config(log_level=log_level).log_level == expected


@pytest.mark.parametrize("log_level", ["", "verbose", 10])
def test_log_level_rejects_unknown_levels(log_level: object) -> None:
    """Test that a log level that is not one of loguru's standard levels is rejected."""
    with pytest.raises(ValidationError, match="log_level"):
        make_config(log_level=log_level)


def test_override_replaces_only_the_settings_given() -> None:
    """Test that an override changes the given settings, leaving those given as None and the rest untouched."""
    config = make_config(log_level="warning", max_threads=4)

    overridden = config.override(log_level="debug", max_threads=None)

    assert overridden.log_level == "DEBUG"
    assert overridden.model_dump(exclude={"log_level"}) == config.model_dump(exclude={"log_level"})
    assert config.override(log_level=None) is config


def test_override_validates_the_overrides() -> None:
    """Test that an override is held to the same validation as the configuration file."""
    with pytest.raises(ValidationError, match="max_threads"):
        make_config().override(max_threads=0)


def load_invalid_config(path: Path, **overrides: object) -> list[str]:
    """Load a configuration that should be rejected, capturing the log messages.

    Returns:
        list[str]: The log messages.
    """
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")

    try:
        with pytest.raises(SystemExit) as exc_info:
            load_config(path, **overrides)
    finally:
        logger.remove(sink)

    assert exc_info.value.code == ExitStatus.INVALID_INPUT
    return messages


def test_load_config_applies_the_overrides(tmp_path: Path) -> None:
    """Test that the options given override the configuration file, and those given as None leave it unchanged."""
    path = write_config(tmp_path, {**REQUIRED_SETTINGS, "factoring_mode": "yafu", "log_level": "warning"})

    config = load_config(path, factoring_mode=None, log_level="DEBUG", max_threads=8)

    assert (config.factoring_mode, config.log_level, config.max_threads) == ("yafu", "DEBUG", 8)


def test_load_config_reports_a_missing_file(tmp_path: Path) -> None:
    """Test that a missing configuration file is a configuration error."""
    path = tmp_path / "missing.json"

    assert load_invalid_config(path) == [f"Unable to read configuration file {path}: {os.strerror(errno.ENOENT)}\n"]


def test_load_config_reports_a_file_that_cannot_be_read(tmp_path: Path) -> None:
    """Test that a configuration path that can't be read is a configuration error, rather than an internal error."""
    # The reason given varies by platform.
    [message] = load_invalid_config(tmp_path)

    assert message.startswith(f"Unable to read configuration file {tmp_path}: ")


def test_load_config_reports_a_file_that_is_not_utf8(tmp_path: Path) -> None:
    """Test that a configuration file that isn't valid UTF-8 is a configuration error, rather than an internal error."""
    path = tmp_path / "config.json"
    path.write_bytes(b"\xff\xfe{}")

    assert any("Invalid JSON" in message for message in load_invalid_config(path))


@pytest.mark.parametrize("max_threads", [0, -3])
def test_load_config_names_an_invalid_override(tmp_path: Path, max_threads: int) -> None:
    """Test that an invalid override is reported as the option it came from, rather than raised."""
    messages = load_invalid_config(write_config(tmp_path, REQUIRED_SETTINGS), max_threads=max_threads)

    assert messages == [f"--max_threads ({max_threads}) is invalid: Input should be greater than 0\n"]


def test_load_config_reports_an_override_the_rest_of_the_configuration_rejects(tmp_path: Path) -> None:
    """Test that an override that is valid on its own, but not with the other settings, is reported."""
    messages = load_invalid_config(write_config(tmp_path, REQUIRED_SETTINGS), backend="mersenne_ca")

    assert messages == ["Configuration error with the given options: the mersenne_ca backend requires gimps_login\n"]
