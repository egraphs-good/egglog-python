"""
Version Bumper for Cargo.toml and Changelog.md

This script automates the process of version bumping for Rust projects managed with Cargo. It reads the version
from the cargo.toml file, increments it based on the specified component (major, minor, or patch), and updates
both the cargo.toml and changelog.md files accordingly.

It will also print out the new version number.

Usage:
    Run the script from the command line, specifying the type of version increment as an argument:
    $ python bump_version.py [major|minor|patch]

Arguments:
---------
    major - Increments the major component of the version, sets minor and patch to 0
    minor - Increments the minor component of the version, sets patch to 0
    patch - Increments the patch component of the version

From https://chat.openai.com/share/6b08906d-23a3-4193-9f4e-87076ce56ddb


"""

import datetime
import re
import sys
from pathlib import Path


def bump_version(major: int, minor: int, patch: int, part: str) -> str:
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    return f"{major}.{minor}.{patch}"


def update_cargo_toml(file_path: Path, new_version: str) -> None:
    content = file_path.read_text()
    content = re.sub(r'version = "(\d+\.\d+\.\d+)"', f'version = "{new_version}"', content, count=1)
    file_path.write_text(content)


def update_changelog(file_path: Path, new_version: str) -> None:
    today = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    content = file_path.read_text()
    new_section = f"## UNRELEASED\n\n## {new_version} ({today})"
    content = content.replace("## UNRELEASED", new_section, 1)
    file_path.write_text(content)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("major", "minor", "patch"):
        print("Usage: python bump_version.py [major|minor|patch]")
        sys.exit(1)

    part = sys.argv[1]
    cargo_path = Path("Cargo.toml")
    changelog_path = Path("docs/changelog.md")

    cargo_content = cargo_path.read_text()
    version_match = re.search(r'version = "(\d+)\.(\d+)\.(\d+)"', cargo_content)
    if version_match:
        major, minor, patch = map(int, version_match.groups())
    else:
        print("Current version not found in cargo.toml.")
        sys.exit(1)

    new_version = bump_version(major, minor, patch, part)
    old_version = f"{major}.{minor}.{patch}"
    update_cargo_toml(cargo_path, new_version)
    update_changelog(changelog_path, new_version)
    print(new_version)
