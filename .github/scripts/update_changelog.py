#!/usr/bin/env python3
"""
Script to automatically update the changelog with PR information.
"""

import argparse
import re
import sys
from pathlib import Path


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


def update_changelog(changelog_path, pr_number, pr_title, pr_url):
    """Update the changelog with the new PR entry."""
    
    # Read the current changelog
    with open(changelog_path, 'r', encoding='utf-8') as f:
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
    with open(changelog_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Added changelog entry for PR #{pr_number}: {pr_title}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Update changelog with PR information')
    parser.add_argument('--pr-number', required=True, help='Pull request number')
    parser.add_argument('--pr-title', required=True, help='Pull request title')
    parser.add_argument('--pr-url', required=True, help='Pull request URL')
    parser.add_argument('--changelog-path', default='docs/changelog.md', help='Path to changelog file')
    
    args = parser.parse_args()
    
    changelog_path = Path(args.changelog_path)
    
    if not changelog_path.exists():
        print(f"ERROR: Changelog file not found: {changelog_path}")
        sys.exit(1)
    
    success = update_changelog(changelog_path, args.pr_number, args.pr_title, args.pr_url)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()