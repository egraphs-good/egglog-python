#!/bin/bash
set -e

echo "🚀 Setting up egglog-python development environment..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "📦 Syncing dependencies..."
if uv sync --all-extras --locked 2>&1 | grep -q "Resolved"; then
    echo "✅ Dependencies synced successfully"
else
    echo "✅ Dependencies already up to date"
fi

echo ""
echo "🔍 Running quick validation checks..."
echo ""

# Run ruff check (non-blocking)
echo "  → Checking code style with ruff..."
if uv run ruff check . --quiet 2>&1; then
    echo "    ✅ Ruff checks passed"
else
    echo "    ⚠️  Ruff found some issues (run 'uv run ruff check --fix .' to auto-fix)"
fi

# Run quick type check (non-blocking)
echo "  → Type checking with mypy..."
if make mypy 2>&1 | tail -n 1 | grep -q "Success"; then
    echo "    ✅ Type checks passed"
else
    echo "    ⚠️  Type check issues found (run 'make mypy' for details)"
fi

echo ""
echo "✨ Environment ready! Quick reference:"
echo ""
echo "  Run tests:        uv run pytest --benchmark-disable -vvv"
echo "  After Rust edit:  uv sync --reinstall-package egglog --all-extras"
echo "  Format code:      uv run ruff format ."
echo "  Fix linting:      uv run ruff check --fix ."
echo "  Type check:       make mypy"
echo ""
echo "📚 See docs/reference/contributing.md for complete development guide"
echo ""
