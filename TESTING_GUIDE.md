# Testing Guide for Randomized Algorithms

This guide explains how to test all the randomized algorithms implemented in this project.

## Prerequisites

- Python ≥ 3.13
- `uv` package manager (install from https://github.com/astral-sh/uv)

## Quick Start

### 1. Check Project Setup

First, verify everything is set up correctly:

```bash
./check_project.sh
```

This will:
- Verify `uv` is installed
- Sync all dependencies
- Run `uv check` to validate the project
- Compile-check all Python files

### 2. Quick Test (Recommended First)

Run a quick test on smaller graphs (4-15 vertices) to verify all algorithms work:

```bash
./quick_test.sh
```

This will:
- Generate small test graphs if needed
- Test all 8 randomized algorithms (ONLY random, no exhaustive/heuristic)
- Use a 10-second time limit per algorithm
- Save results to `experiments/results_quick/`

**Time:** ~5-10 minutes

### 3. Comprehensive Test Suite

Run the full test suite on graphs of varying sizes:

```bash
./run_all_random_tests.sh
```

This will:
- Generate graphs in two ranges:
  - **Small graphs (4-20 vertices):** For baseline comparison
  - **Large graphs (20-50 vertices):** For randomized algorithm testing
- Run baseline algorithms separately (exhaustive on small, greedy on all)
- Test all 8 randomized algorithms (ONLY random, no exhaustive/heuristic) on both small and large graphs
- Use appropriate time limits (30-60 seconds depending on algorithm)
- Save results to `experiments/results_randomized/`

**Time:** ~30-60 minutes (depending on your system)

## Algorithms Tested

The scripts test all these randomized algorithms:

1. **random_construction** - Basic random construction
2. **random_greedy_hybrid** - Hybrid of random and greedy
3. **iterative_random_search** - Iterative random search
4. **monte_carlo** - Monte Carlo sampling
5. **las_vegas** - Las Vegas algorithm (always correct)
6. **mwc_redu** - Graph reduction preprocessing
7. **max_clique_weight** - Branch-and-bound with weighted coloring (static)
8. **max_clique_dyn_weight** - Branch-and-bound with weighted coloring (dynamic)

## Configuration

### Modify Graph Sizes

Edit `run_all_random_tests.sh` to change graph ranges:

```bash
# Small graphs (for exhaustive comparison)
--min-vertices 4 --max-vertices 20

# Large graphs (for randomized algorithms)
--min-vertices 20 --max-vertices 50
```

### Modify Time Limits

Adjust time limits per algorithm in `run_all_random_tests.sh`:

```bash
# For faster algorithms
extra_args="--random-time-limit 30.0"

# For slower algorithms (like max_clique_weight)
extra_args="--random-time-limit 60.0"
```

### Modify Algorithm Parameters

You can customize algorithm-specific parameters in the test script. For example:

```bash
# For monte_carlo
extra_args="--random-max-iterations 2000 --random-time-limit 30.0"

# For random_greedy_hybrid
extra_args="--random-num-starts 20 --random-top-k 5 --random-factor 0.6"
```

## Understanding Results

Results are saved in JSON and CSV format in the output directories:

```
experiments/results_randomized/
├── exhaustive_baseline/
│   ├── benchmark_results.json
│   └── benchmark_results.csv
├── greedy_baseline_small/
├── greedy_baseline_large/
├── random_construction_small/
├── random_construction_large/
├── monte_carlo_small/
├── monte_carlo_large/
└── ...
```

### Key Metrics

Each result includes:
- **Clique size**: Number of vertices in the found clique
- **Total weight**: Sum of weights in the clique
- **Execution time**: Time taken in seconds
- **Basic operations**: Number of operations performed
- **Configurations tested**: Number of candidate solutions tested
- **Precision**: Percentage compared to optimal (if exhaustive available)

### Visualizing Results

To visualize results from a specific test run:

```bash
uv run python main.py visualize \
    experiments/results_randomized/monte_carlo_small/benchmark_results.json \
    --output-dir experiments/plots/monte_carlo_small
```

## Important Notes

### Running Only Randomized Algorithms

You can now run ONLY randomized algorithms without requiring exhaustive or heuristic flags:

```bash
# Run only random algorithms
uv run python main.py benchmark \
    --exhaustive none \
    --heuristic none \
    --random \
    --random-range "4..50" \
    --random-strategy monte_carlo
```

### Exhaustive Search Limitation

- **Exhaustive search is limited to ≤20 vertices** as it becomes impractical for larger graphs
- The test scripts automatically limit exhaustive search to this range when used as baseline
- Randomized algorithms can be tested on larger graphs (20-50+ vertices) independently

### Performance Expectations

- **Faster algorithms** (random_construction, monte_carlo): ~5-30 seconds per graph
- **Medium algorithms** (random_greedy_hybrid, iterative_random_search): ~10-60 seconds
- **Slower algorithms** (max_clique_weight, max_clique_dyn_weight): ~30-300 seconds

### Memory Usage

Large graphs (50+ vertices) may require significant memory, especially for:
- `max_clique_weight` variants (branch-and-bound)
- Algorithms with extensive duplicate checking

## Troubleshooting

### Scripts Not Executable

```bash
chmod +x check_project.sh quick_test.sh run_all_random_tests.sh
```

### Out of Memory

Reduce graph sizes in the test script:
- Change `--max-vertices 50` to `--max-vertices 30`
- Reduce number of graphs generated

### Timeouts

Increase time limits in the script for slower algorithms:
- Change `--random-time-limit 30.0` to `--random-time-limit 120.0`

### Missing Graphs

Graphs will be automatically generated if missing. To manually generate:

```bash
uv run python main.py generate \
    --seed 112974 \
    --min-vertices 4 \
    --max-vertices 20 \
    --output-dir experiments/graphs
```

## Manual Testing

To test a single algorithm manually:

```bash
# Test monte_carlo on a specific graph
uv run python main.py solve \
    experiments/graphs/graph_n10_d50.graphml \
    --mode random \
    --random-strategy monte_carlo \
    --num-samples 1000 \
    --seed 112974
```

To benchmark a single randomized algorithm (ONLY random, no exhaustive/heuristic):

```bash
uv run python main.py benchmark \
    --graphs-dir experiments/graphs \
    --output-dir experiments/results \
    --exhaustive none \
    --heuristic none \
    --random \
    --random-range "4..50" \
    --random-strategy monte_carlo \
    --random-seed 112974 \
    --random-time-limit 30.0
```

**Key changes:**
- `--exhaustive none` - Skip exhaustive search
- `--heuristic none` - Skip greedy heuristic  
- `--random-range "4..50"` - Specify vertex range for random algorithms
- `--random` - Enable random algorithms

## Next Steps

After running tests:

1. **Analyze results** in the CSV files
2. **Compare algorithms** using the precision metrics
3. **Visualize** using the visualization command
4. **Generate plots** for your report

For more details on individual algorithms, see `RANDOMIZED_ALGORITHMS_EXAMPLES.md`.

