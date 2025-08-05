#!/usr/bin/env python3
"""
Simple test for the changelog update script.
"""

import os
import tempfile
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from update_changelog import update_changelog, find_unreleased_section


def test_find_unreleased_section():
    """Test finding the unreleased section."""
    lines = [
        "# Changelog\n",
        "\n",
        "## UNRELEASED\n",
        "\n",
        "- Some existing entry\n",
        "\n",
        "## 1.0.0\n",
        "- Released version\n"
    ]
    
    unreleased_start, content_start = find_unreleased_section(lines)
    assert unreleased_start == 2, f"Expected unreleased_start=2, got {unreleased_start}"
    assert content_start == 4, f"Expected content_start=4, got {content_start}"
    print("✓ find_unreleased_section test passed")


def test_update_changelog():
    """Test updating the changelog."""
    # Create a temporary changelog file
    changelog_content = """# Changelog

## UNRELEASED

- Existing entry

## 1.0.0

- Released version
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(changelog_content)
        temp_path = f.name
    
    try:
        # Update the changelog
        result = update_changelog(temp_path, "999", "Test PR", "https://example.com/pr/999")
        assert result == True, "update_changelog should return True on success"
        
        # Read the updated content
        with open(temp_path, 'r') as f:
            updated_content = f.read()
        
        # Check that the entry was added
        assert "- Test PR [#999](https://example.com/pr/999)" in updated_content
        
        # Check that it was added in the right place (after UNRELEASED)
        lines = updated_content.split('\n')
        unreleased_idx = lines.index("## UNRELEASED")
        entry_idx = None
        for i, line in enumerate(lines):
            if "Test PR [#999]" in line:
                entry_idx = i
                break
        
        assert entry_idx is not None, "Entry should be found"
        assert entry_idx > unreleased_idx, "Entry should be after UNRELEASED section"
        
        # Test duplicate detection
        result2 = update_changelog(temp_path, "999", "Test PR", "https://example.com/pr/999")
        assert result2 == False, "update_changelog should return False for duplicates"
        
        print("✓ update_changelog test passed")
        
    finally:
        # Clean up
        os.unlink(temp_path)


def main():
    """Run all tests."""
    print("Running changelog automation tests...")
    
    test_find_unreleased_section()
    test_update_changelog()
    
    print("✓ All tests passed!")


if __name__ == '__main__':
    main()