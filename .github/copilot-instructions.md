# GitHub Copilot Instructions for egglog-python

## Project Overview

This repository provides Python bindings for the Rust library `egglog`, enabling the use of e-graphs in Python for optimization, symbolic computation, and analysis. It is a hybrid project combining:
- **Python code** in `python/egglog/` - The main Python API and library
- **Rust code** in `src/` - PyO3-based bindings to the egglog Rust library
- **Documentation** in `docs/` - Sphinx-based documentation

## Repository Structure

- `python/egglog/` - Main Python package source code
- `python/tests/` - Python test suite (pytest-based)
- `src/` - Rust source code for Python bindings (PyO3)
- `docs/` - Documentation source files (Sphinx)
- `test-data/` - Test data files
- `pyproject.toml` - Python project configuration and dependencies
- `Cargo.toml` - Rust project configuration
- `uv.lock` - Locked dependencies (managed by uv)

## Build and Development Commands

### Prerequisites
- **uv** - Package manager (https://github.com/astral-sh/uv)
- **Rust toolchain** - Version 1.79.0 (specified in rust-toolchain.toml)
- **Python** - 3.10+ (see .python-version)

### Common Commands

```bash
# Install dependencies
uv sync --extra dev --locked

# Run tests
uv run pytest --benchmark-disable -vvv --durations=10

# Type checking with mypy
make mypy

# Stub testing
make stubtest

# Build documentation
make docs

# Format code (auto-run by pre-commit)
uv run ruff format .

# Lint code (auto-run by pre-commit)
uv run ruff check --fix .
```

## Python Code Standards

### General Guidelines
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

### Testing
- Tests are located in `python/tests/`
- Use pytest with snapshot testing (pytest-inline-snapshot)
- Benchmarks use pytest-benchmark and CodSpeed
- Run tests with: `uv run pytest --benchmark-disable -vvv`

## Rust Code Standards

### General Guidelines
- **Edition**: Rust 2024
- **FFI**: Uses PyO3 for Python bindings
- **Main library**: Uses egglog from git (currently saulshanabrook/egg-smol branch)

### File Organization
- `src/lib.rs` - Main library entry point
- `src/egraph.rs` - E-graph implementation
- `src/conversions.rs` - Type conversions between Python and Rust
- `src/py_object_sort.rs` - Python object handling
- `src/extract.rs` - Extraction functionality
- `src/error.rs` - Error handling
- `src/serialize.rs` - Serialization support
- `src/termdag.rs` - Term DAG operations
- `src/utils.rs` - Utility functions

## Code Style Preferences

1. **Imports**: Follow Ruff's import sorting
2. **Naming**: 
   - Python: snake_case for functions and variables, PascalCase for classes
   - Rust: Follow standard Rust conventions
3. **Comments**: Use clear, explanatory comments for complex logic
4. **Documentation**: Keep docs synchronized with code changes

## Pre-commit Hooks

The repository uses pre-commit with:
- `ruff-check` with auto-fix
- `ruff-format` for formatting
- `uv-lock` to keep lockfile updated

Run `pre-commit install` to enable automatic checking.

## Dependencies

### Python Dependencies
- **Core**: typing-extensions, black, graphviz, anywidget
- **Array support**: scikit-learn, array_api_compat, numba, numpy>2
- **Dev tools**: ruff, pre-commit, mypy, jupyterlab
- **Testing**: pytest, pytest-benchmark, syrupy (inline snapshots)
- **Docs**: sphinx and related packages

### Rust Dependencies
- **PyO3**: Python bindings framework
- **egglog**: Core e-graph library
- **egraph-serialize**: Serialization support
- **serde_json**: JSON handling

## Contributing Guidelines

When making changes:
1. Update or add tests in `python/tests/` for Python changes
2. Run the full test suite before committing
3. Ensure type checking passes with `make mypy`
4. Build documentation if changing public APIs
5. Follow existing code patterns and style
6. Keep changes minimal and focused

## Common Patterns

### Python API Design
- Use `@egraph.class_` decorator for e-graph classes
- Use `@egraph.function` for functions
- Use `@egraph.method` for methods
- Leverage type annotations for better IDE support

### Rust-Python Integration
- Use PyO3's `#[pyclass]` and `#[pymethods]` macros
- Handle errors with appropriate Python exceptions
- Convert between Rust and Python types in `conversions.rs`

## Documentation

Documentation is built with Sphinx:
- Source files in `docs/`
- Build with `make docs`
- Output in `docs/_build/html/`
- Hosted on ReadTheDocs

## Testing Strategy

1. **Unit tests**: Test individual functions and classes
2. **Integration tests**: Test complete workflows
3. **Snapshot tests**: Use inline snapshots for complex outputs
4. **Benchmarks**: Performance testing with pytest-benchmark
5. **Type checking**: Validate type stubs and annotations

## Performance Considerations

- The library uses Rust for performance-critical operations
- Benchmarking is done via CodSpeed for continuous performance monitoring
- Profile with release builds (`cargo build --release`) when needed

## Continuous Integration

GitHub Actions workflows in `.github/workflows/`:
- `CI.yml` - Main testing, type checking, benchmarks, and docs
- `version.yml` - Version management
- `update-changelog.yml` - Changelog automation

Tests run on Python 3.10, 3.11, 3.12, and 3.13.
