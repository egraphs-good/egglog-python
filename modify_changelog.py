"""
Changelog Modifier and Version Bumper for Cargo.toml and Changelog.md

This script automates the process of version bumping and changelog updates for Rust projects managed with Cargo.
It reads the version from the cargo.toml file, increments it based on the specified component (major, minor, or patch),
and updates both the cargo.toml and changelog.md files accordingly.

It can also add PR entries to the UNRELEASED section of the changelog.

Usage:
    Version bumping:
    $ python modify_changelog.py bump_version [major|minor|patch]

    Adding PR entry:
    $ python modify_changelog.py update_changelog <number> <title>

Subcommands:
-----------
    bump_version - Increments version and updates changelog
        major - Increments the major component of the version, sets minor and patch to 0
        minor - Increments the minor component of the version, sets patch to 0
        patch - Increments the patch component of the version

    update_changelog - Add a PR entry to the UNRELEASED section
        number - PR number
        title - PR title

From https://chat.openai.com/share/6b08906d-23a3-4193-9f4e-87076ce56ddb

"""

import argparse
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


def find_unreleased_section(lines):
    """Find the line number where UNRELEASED section starts and ends."""
    unreleased_start = None
    content_start = None

    for i, line in enumerate(lines):
        if line.strip() == "## UNRELEASED":
            unreleased_start = i
            continue

        if unreleased_start is not None and content_start is None:
            # Skip empty lines after ## UNRELEASED
            if line.strip() == "":
                continue
            content_start = i
            break

    return unreleased_start, content_start


def update_changelog_version(file_path: Path, new_version: str) -> None:
    """Update changelog for version bump - replaces UNRELEASED with versioned section."""
    today = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    content = file_path.read_text()
    new_section = f"## UNRELEASED\n\n## {new_version} ({today})"
    content = content.replace("## UNRELEASED", new_section, 1)
    file_path.write_text(content)


def update_changelog_pr(file_path: Path, pr_number: str, pr_title: str, pr_url: str) -> bool:
    """Update the changelog with the new PR entry. If entry exists, update it; otherwise add new entry."""
    # Read the current changelog
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Find the UNRELEASED section
    unreleased_start, content_start = find_unreleased_section(lines)

    if unreleased_start is None:
        print("ERROR: Could not find '## UNRELEASED' section in changelog")
        return False

    if content_start is None:
        print("ERROR: Could not find content start after UNRELEASED section")
        return False

    # Create the new entry
    new_entry = f"- {pr_title} [#{pr_number}]({pr_url})\n"

    # Check if this PR entry already exists and update it if so
    existing_entry_index = None
    for i, line in enumerate(lines[content_start:], start=content_start):
        if f"[#{pr_number}]" in line:
            existing_entry_index = i
            break
        # Stop checking when we reach the next section
        if line.startswith("## ") and line.strip() != "## UNRELEASED":
            break

    if existing_entry_index is not None:
        # Update existing entry
        lines[existing_entry_index] = new_entry
        print(f"Updated changelog entry for PR #{pr_number}: {pr_title}")
    else:
        # Insert the new entry at the beginning of the unreleased content
        lines.insert(content_start, new_entry)
        print(f"Added changelog entry for PR #{pr_number}: {pr_title}")

    # Write the updated changelog
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return True


def handle_bump_version(args):
    """Handle version bump subcommand."""
    part = args.bump_type
    cargo_path = Path("Cargo.toml")
    changelog_path = Path("docs/changelog.md")

    if not cargo_path.exists():
        print("ERROR: Cargo.toml not found.")
        sys.exit(1)

    if not changelog_path.exists():
        print("ERROR: Changelog file not found.")
        sys.exit(1)

    cargo_content = cargo_path.read_text()
    version_match = re.search(r'version = "(\d+)\.(\d+)\.(\d+)"', cargo_content)
    if version_match:
        major, minor, patch = map(int, version_match.groups())
    else:
        print("Current version not found in cargo.toml.")
        sys.exit(1)

    new_version = bump_version(major, minor, patch, part)
    update_cargo_toml(cargo_path, new_version)
    update_changelog_version(changelog_path, new_version)
    print(new_version)


def handle_update_changelog(args):
    """Handle update changelog subcommand."""
    pr_number = args.number
    pr_title = args.title

    # Construct PR URL from repository info and PR number
    # Default to the egglog-python repository
    pr_url = f"https://github.com/egraphs-good/egglog-python/pull/{pr_number}"

    changelog_path = Path(getattr(args, "changelog_path", "docs/changelog.md"))

    if not changelog_path.exists():
        print(f"ERROR: Changelog file not found: {changelog_path}")
        sys.exit(1)

    success = update_changelog_pr(changelog_path, pr_number, pr_title, pr_url)
    if not success:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Changelog modifier and version bumper")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Bump version subcommand
    bump_parser = subparsers.add_parser("bump_version", help="Bump version and update changelog")
    bump_parser.add_argument("bump_type", choices=["major", "minor", "patch"], help="Type of version bump")

    # Update changelog subcommand
    changelog_parser = subparsers.add_parser("update_changelog", help="Add PR entry to changelog")
    changelog_parser.add_argument("number", help="Pull request number")
    changelog_parser.add_argument("title", help="Pull request title")
    changelog_parser.add_argument("--changelog-path", default="docs/changelog.md", help="Path to changelog file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "bump_version":
        handle_bump_version(args)
    elif args.command == "update_changelog":
        handle_update_changelog(args)


if __name__ == "__main__":
    main()
