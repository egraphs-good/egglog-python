# Automatic Changelog Generation

This repository automatically generates changelog entries for new PRs using a GitHub Action.

## How it works

1. **Trigger**: When a PR is opened or edited
2. **Processing**: The action runs a Python script that:
   - Parses the `docs/changelog.md` file
   - Finds the "## UNRELEASED" section
   - Adds a new entry with format: `- PR_TITLE [#PR_NUMBER](PR_URL)`
   - Checks for duplicates to avoid repeated entries
3. **Update**: Commits the changes back to the PR branch

## Files

- `.github/workflows/update-changelog.yml` - GitHub Action workflow
- `.github/scripts/update_changelog.py` - Python script that updates the changelog

## Safety features

- Only runs for PRs from the same repository (not forks)
- Prevents infinite loops by excluding commits made by GitHub Action
- Includes duplicate detection
- Proper error handling and logging

## Manual usage

You can also run the script manually:

```bash
python .github/scripts/update_changelog.py \
  --pr-number="123" \
  --pr-title="My PR Title" \
  --pr-url="https://github.com/egraphs-good/egglog-python/pull/123"
```