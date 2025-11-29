#!/bin/bash
# Project Check Script
# Runs uv check and other validation commands

set -e  # Exit on error

echo "=========================================="
echo "Project Check Script"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "   Install from: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✓ uv is installed: $(uv --version)"
echo ""

# Sync dependencies
echo "📦 Syncing dependencies..."
uv sync
echo ""

# Verify uv can run Python
echo "🔍 Verifying Python environment..."
uv run python --version
echo ""

# Check if main.py is valid
echo "🔍 Validating main.py..."
uv run python -m py_compile main.py
echo "✓ main.py syntax is valid"
echo ""

# Check if source files are valid
echo "🔍 Validating source files..."
for file in src/*.py; do
    if [ -f "$file" ]; then
        uv run python -m py_compile "$file"
        echo "✓ $(basename $file) syntax is valid"
    fi
done
echo ""

echo "=========================================="
echo "✅ Project check completed successfully!"
echo "=========================================="

