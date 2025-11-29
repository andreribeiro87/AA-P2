#!/bin/bash
# Quick Test Script for All MWC Algorithms
# Tests all algorithms on a smaller set of graphs for quick validation

# set -e  # Allow script to continue even if one algorithm fails

SEED=112974
GRAPHS_DIR="experiments/graphs"
RESULTS_DIR="experiments/results_quick"

echo "=========================================="
echo "Quick Test - All MWC Algorithms"
echo "=========================================="
echo ""

# Check uv
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    exit 1
fi

# Generate small test graphs if needed (4-15 vertices)
if [ ! -d "$GRAPHS_DIR" ] || [ -z "$(ls -A $GRAPHS_DIR 2>/dev/null)" ]; then
    echo "📦 Generating test graphs (4-15 vertices)..."
    uv run python main.py generate \
        --seed "$SEED" \
        --min-vertices 4 \
        --max-vertices 15 \
        --densities 25.0 50.0 \
        --output-dir "$GRAPHS_DIR"
    echo "✓ Graphs generated"
    echo ""
fi

# Test randomized algorithms (use seed for reproducibility)
declare -a RANDOM_ALGORITHMS=(
    "random_construction"
    "random_greedy_hybrid"
    "iterative_random_search"
    "monte_carlo"
    "las_vegas"
)

# Test reduction/exact algorithms (deterministic, NOT random)
declare -a REDUCTION_ALGORITHMS=(
    "mwc_redu"
    "max_clique_weight"
    "max_clique_dyn_weight"
)

# NEW: Exact Branch & Bound algorithms
declare -a EXACT_ALGORITHMS=(
    "wlmc"
    "tsm_mwc"
)

# NEW: Heuristic/Local Search algorithms
declare -a HEURISTIC_ALGORITHMS=(
    "fast_wclq"
    "scc_walk"
    "mwc_peel"
)

# Test baseline algorithms (exhaustive + greedy)
echo "📊 Running baseline algorithms..."
OUTPUT="${RESULTS_DIR}/baseline"
mkdir -p "$OUTPUT"
uv run python main.py benchmark \
    --graphs-dir "$GRAPHS_DIR" \
    --output-dir "$OUTPUT" \
    --exhaustive "4..15" \
    --heuristic "4..15" \
    --verbose 2>&1 | tee "${OUTPUT}/run.log" || {
    echo "⚠️  Warning: baseline had errors (check ${OUTPUT}/run.log)"
}
echo "✓ Baseline completed"
echo ""

# Test randomized algorithms
echo "📊 Testing randomized algorithms..."
for algo in "${RANDOM_ALGORITHMS[@]}"; do
    echo "🧪 Testing: ${algo}..."
    
    OUTPUT="${RESULTS_DIR}/${algo}"
    mkdir -p "$OUTPUT"

    uv run python main.py benchmark \
        --graphs-dir "$GRAPHS_DIR" \
        --output-dir "$OUTPUT" \
        --exhaustive none \
        --heuristic none \
        --random \
        --random-range "4..100" \
        --random-strategy "$algo" \
        --random-seed "$SEED" \
        --random-time-limit 10.0 \
        --verbose 2>&1 | tee "${OUTPUT}/run.log" || {
        echo "⚠️  Warning: ${algo} had errors (check ${OUTPUT}/run.log)"
    }
    
    echo "✓ ${algo} completed"
    echo ""
done

# Test reduction/exact algorithms
echo "📊 Testing reduction algorithms..."
for algo in "${REDUCTION_ALGORITHMS[@]}"; do
    echo "🧪 Testing: ${algo}..."
    
    OUTPUT="${RESULTS_DIR}/${algo}"
    mkdir -p "$OUTPUT"

    uv run python main.py benchmark \
        --graphs-dir "$GRAPHS_DIR" \
        --output-dir "$OUTPUT" \
        --exhaustive none \
        --heuristic none \
        --reduction \
        --reduction-range "4..100" \
        --reduction-strategy "$algo" \
        --verbose 2>&1 | tee "${OUTPUT}/run.log" || {
        echo "⚠️  Warning: ${algo} had errors (check ${OUTPUT}/run.log)"
    }
    
    echo "✓ ${algo} completed"
    echo ""
done

# Test exact BnB algorithms (WLMC, TSM-MWC)
echo "📊 Testing exact Branch & Bound algorithms..."
for algo in "${EXACT_ALGORITHMS[@]}"; do
    echo "🧪 Testing: ${algo}..."
    
    OUTPUT="${RESULTS_DIR}/${algo}"
    mkdir -p "$OUTPUT"

    uv run python main.py benchmark \
        --graphs-dir "$GRAPHS_DIR" \
        --output-dir "$OUTPUT" \
        --exhaustive none \
        --heuristic none \
        --random \
        --random-range "4..100" \
        --random-strategy "$algo" \
        --random-time-limit 30.0 \
        --verbose 2>&1 | tee "${OUTPUT}/run.log" || {
        echo "⚠️  Warning: ${algo} had errors (check ${OUTPUT}/run.log)"
    }
    
    echo "✓ ${algo} completed"
    echo ""
done

# Test heuristic/local search algorithms (FastWClq, SCCWalk, MWCPeel)
echo "📊 Testing heuristic/local search algorithms..."
for algo in "${HEURISTIC_ALGORITHMS[@]}"; do
    echo "🧪 Testing: ${algo}..."
    
    OUTPUT="${RESULTS_DIR}/${algo}"
    mkdir -p "$OUTPUT"

    uv run python main.py benchmark \
        --graphs-dir "$GRAPHS_DIR" \
        --output-dir "$OUTPUT" \
        --exhaustive none \
        --heuristic none \
        --random \
        --random-range "4..100" \
        --random-strategy "$algo" \
        --random-seed "$SEED" \
        --random-time-limit 10.0 \
        --verbose 2>&1 | tee "${OUTPUT}/run.log" || {
        echo "⚠️  Warning: ${algo} had errors (check ${OUTPUT}/run.log)"
    }
    
    echo "✓ ${algo} completed"
    echo ""
done

echo "=========================================="
echo "✅ Quick test completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "📈 Generate visualizations with:"
echo "   uv run python main.py visualize ${RESULTS_DIR}"
echo "=========================================="
