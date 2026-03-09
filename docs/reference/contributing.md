# Contributing

## Quick Reference

For rapid development, here are the most common commands:

```bash
# Setup
uv sync --all-extras

# After changing Rust code in src/
uv sync --reinstall-package egglog --all-extras

# Testing
uv run pytest --benchmark-disable -vvv              # All tests
uv run pytest -k "test_name" --benchmark-disable    # Specific test

# Code quality
uv run ruff format .                                # Format code
uv run ruff check --fix .                           # Fix linting issues
make mypy                                           # Type checking
make stubtest                                       # Validate type stubs

# Documentation
make docs                                           # Build documentation
```

## Project Overview

This repository provides Python bindings for the Rust library `egglog`, enabling the use of e-graphs in Python for optimization, symbolic computation, and analysis. It is a hybrid project combining:
- **Python code** in `python/egglog/` - The main Python API and library
- **Rust code** in `src/` - PyO3-based bindings to the egglog Rust library
- **Documentation** in `docs/` - Sphinx-based documentation

## Development

This package is in active development and welcomes contributions!

Feel free to bring up any questions/comments on [the Zulip chat](https://egraphs.zulipchat.com/) or open an issue.

All feedback is welcome and encouraged, it's great to hear about anything that works well or could be improved.

### Getting Started

To get started locally developing with this project, fork it and clone it to your local machine.

Using [the Github CLI](https://github.com/cli/cli#installation) this would be:

```bash
brew install gh
gh repo fork egraphs-good/egglog-python --clone
cd egglog-python
```

Then [install Rust](https://www.rust-lang.org/tools/install) and get a Python environment set up with a compatible version. Using [uv](https://docs.astral.sh/uv/getting-started/installation/) this would be:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the package in editable mode with the development dependencies:

```bash
uv sync --all-extras
```

Anytime you change the rust code, you can run `uv sync --reinstall-package egglog --all-extras` to force recompiling the rust code.

If you would like to download a new version of the visualizer source, run `make clean; make`. This will download
the most recent released version from the github actions artifact in the [egraph-visualizer](https://github.com/egraphs-good/egraph-visualizer) repo. It is checked in because it's a pain to get cargo to include only one git ignored file while ignoring the rest of the files that were ignored.

### Verify Your Setup

After installation, verify everything works:

```bash
# Quick validation (runs in ~30s)
uv run pytest -k "test_simple" --benchmark-disable
make mypy
uv run ruff check .
```

If these pass, you're ready to contribute!

### Running Tests

To run the tests, you can use the `pytest` command:

```bash
uv run pytest
```

All code must pass ruff linters and formaters. This will be checked automatically by the pre-commit if you run `pre-commit install`.

To run it manually, you can use:

```bash
uv run pre-commit run --all-files ruff
```

If you make changes to the rust bindings, you can check that the stub files accurately reflect the rust code by running:

```bash
make stubtest
```

All code must all pass MyPy type checking. To run that locally use:

```bash
make mypy
```

Finally, to build the docs locally and test that they work, you can run:

```bash
make docs
```

## Repository Structure

### Core Directories

- `python/egglog/` - Main Python package source code
- `python/tests/` - Python test suite (pytest-based)
- `src/` - Rust source code for Python bindings (PyO3)
- `docs/` - Documentation source files (Sphinx)
- `test-data/` - Test data files
- `pyproject.toml` - Python project configuration and dependencies
- `Cargo.toml` - Rust project configuration
- `uv.lock` - Locked dependencies (managed by uv)

### Rust Source Organization

- `src/lib.rs` - Main library entry point
- `src/egraph.rs` - E-graph implementation
- `src/conversions.rs` - Type conversions between Python and Rust
- `src/py_object_sort.rs` - Python object handling
- `src/extract.rs` - Extraction functionality
- `src/error.rs` - Error handling
- `src/serialize.rs` - Serialization support
- `src/termdag.rs` - Term DAG operations
- `src/utils.rs` - Utility functions

### Python Source Organization

#### Public Interface
All public Python APIs are exported from the top-level `egglog` module. Anything that is public should be exported in `python/egglog/__init__.py` at the top level.

#### Lower-Level Bindings
The `egglog.bindings` module provides lower-level access to the Rust implementation for advanced use cases.

#### Core Python Files
- `python/egglog/__init__.py` - Top-level module exports, defines the public API
- `python/egglog/egraph.py` - Main EGraph class and e-graph management
- `python/egglog/egraph_state.py` - E-graph state and execution management
- `python/egglog/runtime.py` - Runtime system for expression evaluation and method definitions
- `python/egglog/builtins.py` - Built-in types (i64, f64, String, Vec, etc.) and operations
- `python/egglog/declarations.py` - Class, function, and method declaration decorators
- `python/egglog/conversion.py` - Type conversion between Python and egglog types
- `python/egglog/pretty.py` - Pretty printing for expressions and e-graph visualization
- `python/egglog/deconstruct.py` - Deconstruction of Python values into egglog expressions
- `python/egglog/thunk.py` - Lazy evaluation support
- `python/egglog/type_constraint_solver.py` - Type inference and constraint solving
- `python/egglog/config.py` - Configuration settings
- `python/egglog/ipython_magic.py` - IPython/Jupyter integration
- `python/egglog/visualizer_widget.py` - Interactive visualization widget
- `python/egglog/version_compat.py` - Python version compatibility utilities
- `python/egglog/examples/` - End-to-end samples and tutorials demonstrating the API
- `python/egglog/exp/` - Experimental Array API integrations and code generation helpers

The compiled extension artifact `python/egglog/bindings.cpython-*.so` is generated by `uv sync` and should not be edited manually.

## Architecture Overview

### Python-Rust Integration

This is a hybrid Python-Rust project that uses **PyO3** to expose Rust functionality to Python:

1. **Rust Core**: High-performance e-graph operations in `src/` using the upstream `egglog` Rust library
2. **PyO3 Bridge**: Bindings in `src/conversions.rs` handle type conversion between Python and Rust
3. **Python API**: High-level, Pythonic interface in `python/egglog/` wraps the Rust bindings
4. **Build Process**: Maturin compiles Rust code into a Python extension module (`bindings.cpython-*.so`)

### When Rust Rebuilds Are Needed

**Rust changes require rebuild:**
- Any edits to files in `src/`
- Changes to `Cargo.toml` dependencies
- Run: `uv sync --reinstall-package egglog --all-extras`

**Python-only changes (no rebuild needed):**
- Edits to `python/egglog/*.py` files
- Changes to tests in `python/tests/`
- Documentation updates

### The Conversion Registry

The library uses a global conversion registry to map Python types to egglog types. This registry:
- Is populated when you define classes with `@egraph.class_` or similar decorators
- Persists across e-graph instances in the same Python process
- Is automatically reset between test runs (see `python/tests/conftest.py`)

## Code Standards

### Python Code Style

- **Line length**: 120 characters maximum
- **Type hints**: Use type annotations for public APIs and functions
- **Formatting**: Use Ruff for code formatting and linting
- **Testing**: Write tests using pytest in `python/tests/`
- **Docstrings**: Use clear, concise docstrings for public functions and classes

### Ruff Configuration
The project uses Ruff for linting and formatting with specific rules:
- Allows uppercase variable names (N806, N802)
- Allows star imports (F405, F403)
- Allows `exec` and subprocess usage (S102, S307, S603)
- Allows `Any` type annotations (ANN401)
- Test files don't require full type annotations

See `pyproject.toml` for complete Ruff configuration.

### Type Checking
- **mypy** is used for static type checking
- Run `make mypy` to type check Python code
- Run `make stubtest` to validate type stubs against runtime behavior
- Exclusions: `__snapshots__`, `_build`, `conftest.py`

### Naming Conventions
- Python: snake_case for functions and variables, PascalCase for classes
- Rust: Follow standard Rust conventions

### Rust Code Standards

- **Edition**: Rust 2024 (experimental)
- **FFI**: Uses PyO3 for Python bindings
- **Main library**: Uses egglog from git (egraphs-good/egglog, fix-fn-bug branch)

## Testing Strategy

### Test Organization

Tests are located in `python/tests/` with the following categories:

1. **Unit tests**: Test individual functions and classes
2. **Integration tests**: Test complete workflows
3. **Snapshot tests**: Use syrupy for snapshot testing of complex outputs
4. **Benchmarks**: Performance testing with pytest-benchmark and pytest-codspeed
5. **Parallel testing**: Use pytest-xdist for faster test runs
6. **Type checking**: Validate type stubs and annotations

### Running Specific Tests

```bash
# All tests
uv run pytest --benchmark-disable -vvv --durations=10

# Specific test file
uv run pytest python/tests/test_high_level.py --benchmark-disable

# Specific test function
uv run pytest -k "test_simple" --benchmark-disable

# Tests matching a pattern
uv run pytest -k "convert" --benchmark-disable
```

### Test Markers

The project uses pytest markers to categorize tests:
- `@pytest.mark.slow` - Longer-running tests (deselect with `-m "not slow"`)

## Common Development Patterns

### Python API Design
- Define e-graph classes by inheriting from `egglog.Expr`
- Use `@egraph.function` decorator for functions
- Use `@egraph.method` decorator for methods
- Leverage type annotations for better IDE support

### Working with Values
- Use `get_literal_value(expr)` or the `.value` property to get Python values from primitives
- Use pattern matching with `match`/`case` for destructuring egglog primitives
- Use `get_callable_fn(expr)` to get the underlying Python function from a callable expression
- Use `get_callable_args(expr)` to get arguments to a callable

### Parallelism
- The underlying Rust library uses Rayon for parallelism
- Control worker thread count via `RAYON_NUM_THREADS` environment variable
- Defaults to single thread if not set

### Rust-Python Integration
- Use PyO3's `#[pyclass]` and `#[pymethods]` macros
- Handle errors with appropriate Python exceptions
- Convert between Rust and Python types in `conversions.rs`

## Troubleshooting

### Rust changes not taking effect
**Problem**: Modified Rust code but changes aren't visible in Python
**Solution**: Run `uv sync --reinstall-package egglog --all-extras` to force rebuild

### Test failures after dependency changes
**Problem**: Tests fail with import errors after updating dependencies
**Solution**: Run `uv sync --all-extras` to sync dependencies

### Type stub mismatches
**Problem**: MyPy reports errors about bindings module
**Solution**: Run `make stubtest` to validate stubs, may need to regenerate type stubs

### Pre-commit hooks failing
**Problem**: Pre-commit hooks reject commits
**Solution**: Run `uv run ruff format .` and `uv run ruff check --fix .` to auto-fix issues

### Documentation build fails
**Problem**: `make docs` fails with errors
**Solution**: Ensure all extras are installed with `uv sync --all-extras`, check Sphinx warnings

## Debugging

To debug the Rust parts of this project, follow the [PyO3 debugging guide](https://pyo3.rs/main/debugging.html#debugger-specific-setup).
Debug symbols are turned on by default.

### Performance

[`py-spy`](https://github.com/benfred/py-spy) is installed as a development dependency and can be used to profile Python code.
If there is a performance sensitive piece of code, you could isolate it in a file and profile it locally with:

```bash
uv run py-spy record  --format speedscope  --  python -O tmp.py
```

## Making Changes

All changes that impact users should be documented in the `docs/changelog.md` file. Please also add tests for any new features
or bug fixes.

When you are ready to submit your changes, please open a pull request. The CI will run the tests and check the code style.

### Changelog Automation

When you open a pull request, a GitHub Action automatically adds an entry to the UNRELEASED section of the changelog using your PR title and number. This ensures the changelog stays up-to-date without manual intervention.

### Contributing Guidelines

When making changes:
1. Update or add tests in `python/tests/` for Python changes
2. Run the full test suite before committing
3. Ensure type checking passes with `make mypy`
4. Build documentation if changing public APIs
5. Follow existing code patterns and style
6. Keep changes minimal and focused
7. Ensure the automatic changelog entry in `docs/changelog.md` (added when opening the PR) accurately reflects your change and add manual notes if additional clarification is needed

## Documentation

We use the [Diátaxis framework](https://diataxis.fr/) to organize our documentation. The "explanation" section has
been renamed to "Blog" since most of the content there is more like a blog post than a reference manual. It uses
the [ABlog](https://ablog.readthedocs.io/en/stable/index.html#how-it-works) extension.

Documentation is built with Sphinx:
- Source files in `docs/`
- Build with `make docs`
- Output in `docs/_build/html/`
- Hosted on ReadTheDocs

## Performance Considerations

- The library uses Rust for performance-critical operations
- Benchmarking is done via CodSpeed for continuous performance monitoring
- Profile with release builds (`cargo build --release`) when needed

## Governance

The governance is currently informal, with Saul Shanabrook as the lead maintainer. If the project grows and there
are more contributors, we will formalize the governance structure in a way to allow it to be multi-stakeholder and
to spread out the power and responsibility.
