# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

from pathlib import Path

from factortool.config import read_config


def main() -> None:
    try:
        _config = read_config(Path("config.json"))
    except FileNotFoundError:
        print("Configuration file not found")
