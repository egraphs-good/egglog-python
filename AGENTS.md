# Repo Guidance

## Overview

- This repo combines the high-level Python bindings in `python/egglog/`, the Rust extension in `src/`, and the Sphinx docs in `docs/`.
- Public Python APIs are exported from `python/egglog/__init__.py`.
- The compiled `python/egglog/bindings.cpython-*.so` artifact is generated and should not be edited directly.

## Common Commands

- `uv sync --all-extras` installs the full dev environment.
- `uv sync --reinstall-package egglog --all-extras` rebuilds the Rust extension after changes in `src/`.
- `uv run pytest --benchmark-disable -q` runs the Python tests without benchmark calibration.
- `make mypy` runs the type checker.
- `make stubtest` checks the runtime against the type stubs.
- `make docs` builds the docs.

## Docs

- Use the Context7 MCP server for egglog documentation instead of copying external doc summaries into this file.
- Keep general workflows in the how-to guides, and keep Python-specific runtime/reference examples in `docs/reference/python-integration.md`.
- If a PR adds or updates a changelog entry in `docs/changelog.md`, keep it aligned with the final code changes.
- For a clean docs rebuild, clear `docs/_build/`; the MyST-NB execution cache lives in `docs/_build/.jupyter_cache`.

## Python bindings

- Prefer relative imports inside `python/egglog`.
- When changing public high-level APIs, update the public docs, stubs, and pretty/freeze round-trip expectations together.
- Higher-order callable type probing should stay isolated from the live ruleset: copy declarations and run with no current ruleset so inference does not register temporary unnamed functions or rewrites.

## Array API

- Start with `python/egglog/exp/array_api.py` and `python/tests/test_array_api.py`.
- `Vec[...]` is a primitive sort; avoid rewrites or unions that merge distinct vec values.
- Guard vector indexing rewrites with explicit bounds checks.

## CI

- When debugging GitHub Actions logs, prefer the private `$github-actions-rest-logs` skill or the equivalent REST API flow with `GITHUB_PAT_TOKEN`.

## Verification

- Prefer the minimal code change and the minimal diff that solves the task; only broaden the change if the smaller fix is not sufficient.
- Run `make mypy` for typing changes.
- Run targeted pytest for touched modules.
- Run `make docs` for docs or public API changes.
