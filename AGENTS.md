# Repo Guidance

## Overview

- This repo combines the high-level Python bindings in `python/egglog/`, the Rust extension in `src/`, and the Sphinx docs in `docs/`.
- Public Python APIs are exported from `python/egglog/__init__.py`.
- The compiled `python/egglog/bindings.cpython-*.so` artifact is generated and should not be edited directly.

## Common Commands

- `uv sync --all-extras` installs the full dev environment.
- `uv sync --reinstall-package egglog --all-extras` rebuilds the Rust extension after changes in `src/`.
- `make test` is the default command for a full test sweep; it runs the suite in `--release` mode and uses the repo's parallel runner.
- `uv run pytest --benchmark-disable -q` runs the Python tests without benchmark calibration.
- `make mypy` runs the type checker.
- `make stubtest` checks the runtime against the type stubs.
- `make docs` builds the docs.

## Docs

- Use the Context7 MCP server for egglog documentation instead of copying external doc summaries into this file.
- Keep general workflows in the how-to guides, and keep Python-specific runtime/reference examples in `docs/reference/python-integration.md`.
- Before changing Python-facing `egglog` APIs or expression-inspection patterns, read `docs/reference/python-integration.md`.
- Before changing how egglog concepts map into Python declarations, relations, actions, or sort/function definitions, read `docs/reference/egglog-translation.md`.
- If a change touches both local Python ergonomics and egglog-language mapping, read `docs/reference/python-integration.md` first, then `docs/reference/egglog-translation.md`.
- If a PR adds or updates a changelog entry in `docs/changelog.md`, keep it aligned with the final code changes.
- For a clean docs rebuild, clear `docs/_build/`; the MyST-NB execution cache lives in `docs/_build/.jupyter_cache`.

## Python bindings

- Prefer relative imports inside `python/egglog`.
- When changing public high-level APIs, update the public docs, stubs, and pretty/freeze round-trip expectations together.
- Higher-order callable type probing should stay isolated from the live ruleset: copy declarations and run with no current ruleset so inference does not register temporary unnamed functions or rewrites.
- If a helper returns a primitive or container and must participate in rewrites or higher-order container ops, implement it as an exact Rust builtin.
- If that helper can naturally be partial, prefer a fallible builtin on the builtin/base sort itself over wrapping the result in a subsystem-local optional sort.
- If that partial builtin may be undefined for some inputs, remember the runtime behavior precisely:
  - in actions, undefined partial primitive results error
  - in facts / antecedent-style use, undefined partial primitive calls cause the rule to skip rather than firing
  - in higher-order container ops, the container primitive should define the policy for undefined callback results, for example skipping those entries or failing the whole op
- Use partial builtins in antecedent/lookup positions by default; do not materialize them in actions unless definedness has already been proven.
- In Python, only pass exact builtins or partials of exact builtins into higher-order container ops for this workflow.
- Do not add Python-bodied primitive/container helpers or anonymous lambda callbacks for these paths.
- Prefer `x == y` over `eq(x).to(y)` for ordinary equality facts, checks, and rule antecedents when the sort uses the default equality relation.
- Only fall back to `eq(x).to(y)` when the sort overloads `__eq__` and you explicitly need the raw equality relation rather than the overloaded meaning.
- Prefer the correct container shape for higher-order ops:
  - use the container whose semantics match the operation rather than forcing the logic through whatever higher-order primitive already exists
  - use `Set` for support / keys / uniqueness-preserving operations
  - use `MultiSet` only when multiplicity is semantically required
  - use `Vec` when order, positional access, or stable sequencing is part of the semantics
  - use `Map` when the operation is keyed and values matter, not just the key support
  - if the right higher-order container op does not exist for the right container type, add it in Rust instead of forcing the logic through a mismatched container
- A bare expression in an action already materializes that expression; do not write `union(x).with_(x)` when `x` alone has the same effect.

## Array API

- Start with `python/egglog/exp/array_api.py` and `python/tests/test_array_api.py`.
- `Vec[...]` is a primitive sort; avoid rewrites or unions that merge distinct vec values.
- Guard vector indexing rewrites with explicit bounds checks.

## CI

- When debugging GitHub Actions logs, prefer the private `$github-actions-rest-logs` skill or the equivalent REST API flow with `GITHUB_PAT_TOKEN`.

## Verification

- Prefer the minimal code change and the minimal diff that solves the task; only broaden the change if the smaller fix is not sufficient.
- For long-running profiling or trace probes, run them with explicit timeouts, check for lingering worker processes before and after, and inspect memory usage after any timeout or manual kill before starting the next experiment.
- For raw engine/container primitive smoke tests, prefer `/Users/saul/p/egg-smol/tests/map.egg` with `cargo run --manifest-path /Users/saul/p/egg-smol/Cargo.toml -- /Users/saul/p/egg-smol/tests/map.egg`.
- Do not rerun broad suites after doc-only or comment-only edits; run the smallest targeted check that could actually fail.
- If you need to run the full repo test suite, prefer `make test` over ad hoc `cargo test` or broad `pytest` invocations.
- Run `make mypy` for typing changes.
- Run targeted pytest for touched modules.
- Run `make docs` for docs or public API changes.
