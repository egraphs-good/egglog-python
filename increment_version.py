"""
Version Bumper for Cargo.toml and Changelog.md

This script automates the process of version bumping for Rust projects managed with Cargo. It reads the version
from the cargo.toml file, increments it based on the specified component (major, minor, or patch), and updates
both the cargo.toml and changelog.md files accordingly.

It will also print out the new version number.

Additionally, this script can add PR entries to the UNRELEASED section of the changelog.

Usage:
    Version bumping:
    $ python increment_version.py [major|minor|patch]
    
    Adding PR entry:
    $ python increment_version.py --add-pr --pr-number=123 --pr-title="Fix bug" --pr-url="https://github.com/..."

Arguments:
---------
    major - Increments the major component of the version, sets minor and patch to 0
    minor - Increments the minor component of the version, sets patch to 0
    patch - Increments the patch component of the version
    --add-pr - Add a PR entry to the UNRELEASED section
    --pr-number - PR number (required with --add-pr)
    --pr-title - PR title (required with --add-pr)
    --pr-url - PR URL (required with --add-pr)

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
            else:
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
    """Update the changelog with the new PR entry. Returns True if successful, False if entry already exists."""
    
    # Read the current changelog
    with open(file_path, 'r', encoding='utf-8') as f:
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
    
    # Check if this PR entry already exists to avoid duplicates
    for line in lines[content_start:]:
        if f"[#{pr_number}]" in line:
            print(f"Changelog entry for PR #{pr_number} already exists")
            return False
        # Stop checking when we reach the next section
        if line.startswith("## ") and not line.strip() == "## UNRELEASED":
            break
    
    # Insert the new entry at the beginning of the unreleased content
    lines.insert(content_start, new_entry)
    
    # Write the updated changelog
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Added changelog entry for PR #{pr_number}: {pr_title}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Version bumper and changelog updater')
    
    # Create mutually exclusive group for version bump vs PR add
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('bump_type', nargs='?', choices=['major', 'minor', 'patch'], 
                      help='Type of version bump')
    group.add_argument('--add-pr', action='store_true', help='Add PR entry to changelog')
    
    # PR-specific arguments
    parser.add_argument('--pr-number', help='Pull request number (required with --add-pr)')
    parser.add_argument('--pr-title', help='Pull request title (required with --add-pr)')
    parser.add_argument('--pr-url', help='Pull request URL (required with --add-pr)')
    parser.add_argument('--changelog-path', default='docs/changelog.md', help='Path to changelog file')
    
    args = parser.parse_args()
    
    # Handle PR addition
    if args.add_pr:
        if not all([args.pr_number, args.pr_title, args.pr_url]):
            print("ERROR: --pr-number, --pr-title, and --pr-url are required with --add-pr")
            sys.exit(1)
        
        changelog_path = Path(args.changelog_path)
        if not changelog_path.exists():
            print(f"ERROR: Changelog file not found: {changelog_path}")
            sys.exit(1)
        
        success = update_changelog_pr(changelog_path, args.pr_number, args.pr_title, args.pr_url)
        if not success:
            sys.exit(1)
        return
    
    # Handle version bump (existing functionality)
    if not args.bump_type:
        print("ERROR: Either specify bump type (major|minor|patch) or use --add-pr")
        sys.exit(1)
    
    part = args.bump_type
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
    update_changelog_version(changelog_path, new_version)
    print(new_version)


if __name__ == "__main__":
    # For backward compatibility, support old command line format
    if len(sys.argv) == 2 and sys.argv[1] in ("major", "minor", "patch"):
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
        update_changelog_version(changelog_path, new_version)
        print(new_version)
    else:
        main()
