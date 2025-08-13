"""Tests for modify_changelog.py script."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_modify_changelog_help():
    """Test that the script shows help correctly."""
    result = subprocess.run(
        [sys.executable, "modify_changelog.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
        check=False,
    )
    assert result.returncode == 0
    assert "Changelog modifier and version bumper" in result.stdout
    assert "bump_version" in result.stdout
    assert "update_changelog" in result.stdout


def test_bump_version_subcommand_help():
    """Test that bump_version subcommand shows help correctly."""
    result = subprocess.run(
        [sys.executable, "modify_changelog.py", "bump_version", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
        check=False,
    )
    assert result.returncode == 0
    assert "Type of version bump" in result.stdout
    assert "major" in result.stdout
    assert "minor" in result.stdout
    assert "patch" in result.stdout


def test_update_changelog_subcommand_help():
    """Test that update_changelog subcommand shows help correctly."""
    result = subprocess.run(
        [sys.executable, "modify_changelog.py", "update_changelog", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
        check=False,
    )
    assert result.returncode == 0
    assert "Pull request number" in result.stdout
    assert "Pull request title" in result.stdout


@pytest.mark.parametrize(
    ("start_version", "bump_type", "expected_version"),
    [
        pytest.param("1.2.3", "patch", "1.2.4", id="patch_bump"),
        pytest.param("1.2.3", "minor", "1.3.0", id="minor_bump"),
        pytest.param("1.2.3", "major", "2.0.0", id="major_bump"),
    ],
)
def test_bump_version(start_version, bump_type, expected_version):
    """Test version bumping with different increment types."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock Cargo.toml
        cargo_content = f'''[package]
name = "test-package"
version = "{start_version}"
edition = "2021"
'''
        cargo_path = temp_path / "Cargo.toml"
        cargo_path.write_text(cargo_content)

        # Create mock changelog
        changelog_content = f"""# Changelog

## UNRELEASED

## {start_version} (2024-01-01)

- Some old change
"""
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        changelog_path = docs_dir / "changelog.md"
        changelog_path.write_text(changelog_content)

        # Run the script
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent / "modify_changelog.py"),
                "bump_version",
                bump_type,
            ],
            capture_output=True,
            text=True,
            cwd=temp_path,
            check=False,
        )

        assert result.returncode == 0
        assert result.stdout.strip() == expected_version

        # Check Cargo.toml was updated
        updated_cargo = cargo_path.read_text()
        assert f'version = "{expected_version}"' in updated_cargo

        # Check changelog was updated
        updated_changelog = changelog_path.read_text()
        assert "## UNRELEASED" in updated_changelog
        assert f"## {expected_version} (" in updated_changelog


def test_update_changelog_new_entry():
    """Test adding a new PR entry to changelog."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock changelog
        changelog_content = """# Changelog

## UNRELEASED

## 1.2.3 (2024-01-01)

- Some old change
"""
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        changelog_path = docs_dir / "changelog.md"
        changelog_path.write_text(changelog_content)

        # Run the script
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent / "modify_changelog.py"),
                "update_changelog",
                "123",
                "Fix important bug",
            ],
            capture_output=True,
            text=True,
            cwd=temp_path,
            check=False,
        )

        assert result.returncode == 0
        assert "Added changelog entry for PR #123: Fix important bug" in result.stdout

        # Check changelog was updated
        updated_changelog = changelog_path.read_text()
        assert "- Fix important bug [#123](https://github.com/egraphs-good/egglog-python/pull/123)" in updated_changelog


def test_update_changelog_duplicate_entry():
    """Test that modifying PR title updates the existing changelog entry instead of making a new one."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock changelog with existing entry
        changelog_content = """# Changelog

## UNRELEASED

- Fix important bug [#123](https://github.com/egraphs-good/egglog-python/pull/123)

## 1.2.3 (2024-01-01)

- Some old change
"""
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        changelog_path = docs_dir / "changelog.md"
        changelog_path.write_text(changelog_content)

        # Run the script with updated title for same PR
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent / "modify_changelog.py"),
                "update_changelog",
                "123",
                "Fix critical security bug",
            ],
            capture_output=True,
            text=True,
            cwd=temp_path,
            check=False,
        )

        assert result.returncode == 0
        assert "Updated changelog entry for PR #123: Fix critical security bug" in result.stdout

        # Check that the changelog was updated, not duplicated
        updated_changelog = changelog_path.read_text()
        assert (
            "- Fix critical security bug [#123](https://github.com/egraphs-good/egglog-python/pull/123)"
            in updated_changelog
        )
        assert "- Fix important bug [#123]" not in updated_changelog  # Old entry should be gone
        assert updated_changelog.count("[#123]") == 1  # Should only have one entry for PR 123


def test_update_changelog_missing_file():
    """Test error handling when changelog file is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run the script without creating changelog file
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent / "modify_changelog.py"),
                "update_changelog",
                "123",
                "Fix important bug",
            ],
            capture_output=True,
            text=True,
            cwd=temp_path,
            check=False,
        )

        assert result.returncode == 1
        assert "ERROR: Changelog file not found" in result.stdout


def test_bump_version_missing_cargo():
    """Test error handling when Cargo.toml file is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create changelog but no Cargo.toml
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        changelog_path = docs_dir / "changelog.md"
        changelog_path.write_text("# Changelog\n\n## UNRELEASED\n")

        # Run the script
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent.parent / "modify_changelog.py"), "bump_version", "patch"],
            capture_output=True,
            text=True,
            cwd=temp_path,
            check=False,
        )

        assert result.returncode == 1
        assert "ERROR: Cargo.toml not found" in result.stdout


def test_custom_changelog_path():
    """Test using custom changelog path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock changelog in custom location
        changelog_content = """# Changelog

## UNRELEASED

## 1.2.3 (2024-01-01)

- Some old change
"""
        custom_changelog_path = temp_path / "CHANGELOG.md"
        custom_changelog_path.write_text(changelog_content)

        # Run the script with custom path
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent / "modify_changelog.py"),
                "update_changelog",
                "456",
                "Add new feature",
                "--changelog-path",
                "CHANGELOG.md",
            ],
            capture_output=True,
            text=True,
            cwd=temp_path,
            check=False,
        )

        assert result.returncode == 0
        assert "Added changelog entry for PR #456: Add new feature" in result.stdout

        # Check changelog was updated
        updated_changelog = custom_changelog_path.read_text()
        assert "- Add new feature [#456](https://github.com/egraphs-good/egglog-python/pull/456)" in updated_changelog
