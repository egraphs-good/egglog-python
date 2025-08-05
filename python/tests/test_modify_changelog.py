"""Tests for modify_changelog.py script."""

import tempfile
import subprocess
import sys
from pathlib import Path
import pytest


def test_modify_changelog_help():
    """Test that the script shows help correctly."""
    result = subprocess.run([sys.executable, "modify_changelog.py", "--help"], 
                           capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
    assert result.returncode == 0
    assert "Changelog modifier and version bumper" in result.stdout
    assert "bump_version" in result.stdout
    assert "update_changelog" in result.stdout


def test_bump_version_subcommand_help():
    """Test that bump_version subcommand shows help correctly."""
    result = subprocess.run([sys.executable, "modify_changelog.py", "bump_version", "--help"], 
                           capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
    assert result.returncode == 0
    assert "Type of version bump" in result.stdout
    assert "major" in result.stdout
    assert "minor" in result.stdout
    assert "patch" in result.stdout


def test_update_changelog_subcommand_help():
    """Test that update_changelog subcommand shows help correctly."""
    result = subprocess.run([sys.executable, "modify_changelog.py", "update_changelog", "--help"], 
                           capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
    assert result.returncode == 0
    assert "Pull request number" in result.stdout
    assert "Pull request title" in result.stdout


def test_bump_version_patch():
    """Test version bumping with patch increment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock Cargo.toml
        cargo_content = '''[package]
name = "test-package"
version = "1.2.3"
edition = "2021"
'''
        cargo_path = temp_path / "Cargo.toml"
        cargo_path.write_text(cargo_content)
        
        # Create mock changelog
        changelog_content = '''# Changelog

## UNRELEASED

## 1.2.3 (2024-01-01)

- Some old change
'''
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        changelog_path = docs_dir / "changelog.md"
        changelog_path.write_text(changelog_content)
        
        # Run the script
        result = subprocess.run([sys.executable, 
                               str(Path(__file__).parent.parent.parent / "modify_changelog.py"), 
                               "bump_version", "patch"], 
                              capture_output=True, text=True, cwd=temp_path)
        
        assert result.returncode == 0
        assert result.stdout.strip() == "1.2.4"
        
        # Check Cargo.toml was updated
        updated_cargo = cargo_path.read_text()
        assert 'version = "1.2.4"' in updated_cargo
        
        # Check changelog was updated
        updated_changelog = changelog_path.read_text()
        assert "## UNRELEASED" in updated_changelog
        assert "## 1.2.4 (" in updated_changelog


def test_bump_version_minor():
    """Test version bumping with minor increment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock Cargo.toml
        cargo_content = '''[package]
name = "test-package"
version = "1.2.3"
edition = "2021"
'''
        cargo_path = temp_path / "Cargo.toml"
        cargo_path.write_text(cargo_content)
        
        # Create mock changelog
        changelog_content = '''# Changelog

## UNRELEASED

## 1.2.3 (2024-01-01)

- Some old change
'''
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        changelog_path = docs_dir / "changelog.md"
        changelog_path.write_text(changelog_content)
        
        # Run the script
        result = subprocess.run([sys.executable, 
                               str(Path(__file__).parent.parent.parent / "modify_changelog.py"), 
                               "bump_version", "minor"], 
                              capture_output=True, text=True, cwd=temp_path)
        
        assert result.returncode == 0
        assert result.stdout.strip() == "1.3.0"
        
        # Check Cargo.toml was updated
        updated_cargo = cargo_path.read_text()
        assert 'version = "1.3.0"' in updated_cargo


def test_bump_version_major():
    """Test version bumping with major increment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock Cargo.toml
        cargo_content = '''[package]
name = "test-package"
version = "1.2.3"
edition = "2021"
'''
        cargo_path = temp_path / "Cargo.toml"
        cargo_path.write_text(cargo_content)
        
        # Create mock changelog
        changelog_content = '''# Changelog

## UNRELEASED

## 1.2.3 (2024-01-01)

- Some old change
'''
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        changelog_path = docs_dir / "changelog.md"
        changelog_path.write_text(changelog_content)
        
        # Run the script
        result = subprocess.run([sys.executable, 
                               str(Path(__file__).parent.parent.parent / "modify_changelog.py"), 
                               "bump_version", "major"], 
                              capture_output=True, text=True, cwd=temp_path)
        
        assert result.returncode == 0
        assert result.stdout.strip() == "2.0.0"
        
        # Check Cargo.toml was updated
        updated_cargo = cargo_path.read_text()
        assert 'version = "2.0.0"' in updated_cargo


def test_update_changelog_new_entry():
    """Test adding a new PR entry to changelog."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock changelog
        changelog_content = '''# Changelog

## UNRELEASED

## 1.2.3 (2024-01-01)

- Some old change
'''
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        changelog_path = docs_dir / "changelog.md"
        changelog_path.write_text(changelog_content)
        
        # Run the script
        result = subprocess.run([sys.executable, 
                               str(Path(__file__).parent.parent.parent / "modify_changelog.py"), 
                               "update_changelog", "123", "Fix important bug"], 
                              capture_output=True, text=True, cwd=temp_path)
        
        assert result.returncode == 0
        assert "Added changelog entry for PR #123: Fix important bug" in result.stdout
        
        # Check changelog was updated
        updated_changelog = changelog_path.read_text()
        assert "- Fix important bug [#123](https://github.com/egraphs-good/egglog-python/pull/123)" in updated_changelog


def test_update_changelog_duplicate_entry():
    """Test that duplicate PR entries are not added."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock changelog with existing entry
        changelog_content = '''# Changelog

## UNRELEASED

- Fix important bug [#123](https://github.com/egraphs-good/egglog-python/pull/123)

## 1.2.3 (2024-01-01)

- Some old change
'''
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        changelog_path = docs_dir / "changelog.md"
        changelog_path.write_text(changelog_content)
        
        # Run the script
        result = subprocess.run([sys.executable, 
                               str(Path(__file__).parent.parent.parent / "modify_changelog.py"), 
                               "update_changelog", "123", "Fix important bug"], 
                              capture_output=True, text=True, cwd=temp_path)
        
        assert result.returncode == 1
        assert "Changelog entry for PR #123 already exists" in result.stdout


def test_update_changelog_missing_file():
    """Test error handling when changelog file is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run the script without creating changelog file
        result = subprocess.run([sys.executable, 
                               str(Path(__file__).parent.parent.parent / "modify_changelog.py"), 
                               "update_changelog", "123", "Fix important bug"], 
                              capture_output=True, text=True, cwd=temp_path)
        
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
        result = subprocess.run([sys.executable, 
                               str(Path(__file__).parent.parent.parent / "modify_changelog.py"), 
                               "bump_version", "patch"], 
                              capture_output=True, text=True, cwd=temp_path)
        
        assert result.returncode == 1
        assert "ERROR: Cargo.toml not found" in result.stdout


def test_custom_changelog_path():
    """Test using custom changelog path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock changelog in custom location
        changelog_content = '''# Changelog

## UNRELEASED

## 1.2.3 (2024-01-01)

- Some old change
'''
        custom_changelog_path = temp_path / "CHANGELOG.md"
        custom_changelog_path.write_text(changelog_content)
        
        # Run the script with custom path
        result = subprocess.run([sys.executable, 
                               str(Path(__file__).parent.parent.parent / "modify_changelog.py"), 
                               "update_changelog", "456", "Add new feature",
                               "--changelog-path", "CHANGELOG.md"], 
                              capture_output=True, text=True, cwd=temp_path)
        
        assert result.returncode == 0
        assert "Added changelog entry for PR #456: Add new feature" in result.stdout
        
        # Check changelog was updated
        updated_changelog = custom_changelog_path.read_text()
        assert "- Add new feature [#456](https://github.com/egraphs-good/egglog-python/pull/456)" in updated_changelog