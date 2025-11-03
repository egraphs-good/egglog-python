# Agent Instructions for egglog-python

This file provides instructions for AI coding agents (including GitHub Copilot) working on this repository.

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
- **Rust toolchain** - See rust-toolchain.toml for version
- **Python** - See .python-version for version

### Common Commands

```bash
# Install dependencies
uv sync --all-extras

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
- Use pytest with snapshot testing (syrupy)
- Benchmarks use pytest-benchmark and CodSpeed
- Run tests with: `uv run pytest --benchmark-disable -vvv`

## Rust Code Standards

### General Guidelines
- **Edition**: Rust 2024 (experimental)
- **FFI**: Uses PyO3 for Python bindings
- **Main library**: Uses egglog from git (saulshanabrook/egg-smol, clone-cost branch)

### Rust File Organization
- `src/lib.rs` - Main library entry point
- `src/egraph.rs` - E-graph implementation
- `src/conversions.rs` - Type conversions between Python and Rust
- `src/py_object_sort.rs` - Python object handling
- `src/extract.rs` - Extraction functionality
- `src/error.rs` - Error handling
- `src/serialize.rs` - Serialization support
- `src/termdag.rs` - Term DAG operations
- `src/utils.rs` - Utility functions

### Python File Organization

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

## Code Style Preferences

1. **Imports**: Follow Ruff's import sorting
2. **Naming**: 
   - Python: snake_case for functions and variables, PascalCase for classes
   - Rust: Follow standard Rust conventions
3. **Comments**: Use clear, explanatory comments for complex logic
4. **Documentation**: Keep docs synchronized with code changes

## Contributing Guidelines

When making changes:
1. Update or add tests in `python/tests/` for Python changes
2. Run the full test suite before committing
3. Ensure type checking passes with `make mypy`
4. Build documentation if changing public APIs
5. Follow existing code patterns and style
6. Keep changes minimal and focused
7. Add a changelog entry in `docs/changelog.md` under the UNRELEASED section

## Common Patterns

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

## Documentation

Documentation is built with Sphinx:
- Source files in `docs/`
- Build with `make docs`
- Output in `docs/_build/html/`
- Hosted on ReadTheDocs

## Testing Strategy

1. **Unit tests**: Test individual functions and classes
2. **Integration tests**: Test complete workflows
3. **Snapshot tests**: Use syrupy for snapshot testing of complex outputs
4. **Benchmarks**: Performance testing with pytest-benchmark and pytest-codspeed
5. **Parallel testing**: Use pytest-xdist for faster test runs
6. **Type checking**: Validate type stubs and annotations

## Performance Considerations

- The library uses Rust for performance-critical operations
- Benchmarking is done via CodSpeed for continuous performance monitoring
- Profile with release builds (`cargo build --release`) when needed
