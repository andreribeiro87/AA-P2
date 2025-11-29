# Testing Scripts Quick Reference

## Available Scripts

### 1. `check_project.sh`

Validates the project setup and dependencies.

```bash
./check_project.sh
```

**What it does:**

- Checks if `uv` is installed
- Syncs all dependencies
- Validates Python environment
- Compiles all Python files to check syntax

**Time:** ~5 seconds

---

### 2. `quick_test.sh`

Quick test of all MWC algorithms on small graphs (4-15 vertices).

```bash
./quick_test.sh
```

**What it does:**

- Generates small test graphs (4-15 vertices, 2 densities) if needed
- Tests baseline algorithms (exhaustive + greedy)
- Tests all 5 randomized algorithms
- Tests all 3 reduction algorithms
- Tests all 2 exact Branch & Bound algorithms (WLMC, TSM-MWC)
- Tests all 3 heuristic/local search algorithms (FastWClq, SCCWalk, MWCPeel)
- Uses 10-30 second time limits per algorithm
- Saves results to `experiments/results_quick/`

**Algorithms tested:**

- **Baseline:** exhaustive, greedy
- **Randomized:** random_construction, random_greedy_hybrid, iterative_random_search, monte_carlo, las_vegas
- **Reduction:** mwc_redu, max_clique_weight, max_clique_dyn_weight
- **Exact BnB:** wlmc, tsm_mwc
- **Heuristic:** fast_wclq, scc_walk, mwc_peel

**Time:** ~10-15 minutes

**Best for:** Quick validation that all algorithms work

---

### 3. `run_all_random_tests.sh`

Comprehensive test suite for all algorithms on multiple graph sizes.

```bash
./run_all_random_tests.sh [--generate-graphs|-g]
```

**Options:**

- `--generate-graphs, -g`: Generate new test graphs before testing

**What it does:**

- Generates graphs in two ranges:
  - **Small (4-20 vertices):** For baseline comparison with exhaustive search
  - **Large (20-200 vertices):** For scalability testing
- Runs exhaustive search on small graphs (baseline for comparison)
- Runs greedy heuristic on all graphs (baseline for comparison)
- Tests all algorithm categories on both small and large graphs
- Uses appropriate time limits (30-60 seconds)
- Saves results to `experiments/results_randomized/`
- Creates detailed log file with timestamp

**Time:** ~60-120 minutes

**Best for:** Full experimental analysis for your report

---

### 4. `run_complete_experiments.sh` ⭐ NEW

Complete experiments script that runs all algorithms and generates visualizations.

```bash
./run_complete_experiments.sh [OPTIONS]
```

**Options:**

- `--generate-graphs, -g`: Generate new test graphs
- `--quick, -q`: Run quick tests only (smaller graphs, fewer iterations)
- `--skip-baseline`: Skip exhaustive and greedy baseline tests
- `--skip-visualize`: Skip visualization generation
- `--help, -h`: Show help message

**What it does:**

1. Checks prerequisites (uv, Python, project structure)
2. Generates test graphs (if needed or requested)
3. Runs baseline algorithms (exhaustive + greedy)
4. Tests all 5 randomized algorithms
5. Tests all 3 reduction algorithms
6. Tests all 2 exact Branch & Bound algorithms
7. Tests all 3 heuristic/local search algorithms
8. **Generates all visualizations automatically**
9. Produces summary report

**Time:**

- Quick mode: ~15-20 minutes
- Full mode: ~60-90 minutes

**Best for:** Complete end-to-end experiments for the assignment report

---

## Quick Start

1. **Validate setup:**

   ```bash
   ./check_project.sh
   ```

2. **Quick test (validation):**

   ```bash
   ./quick_test.sh
   ```

3. **Complete experiments with visualizations:**

   ```bash
   ./run_complete_experiments.sh --generate-graphs
   ```

4. **Quick experiments (faster):**
   ```bash
   ./run_complete_experiments.sh --quick --generate-graphs
   ```

---

## Results Structure

### Quick Test Results

```
experiments/results_quick/
├── baseline/
│   ├── benchmark_results.json
│   └── benchmark_results.csv
├── random_construction/
├── random_greedy_hybrid/
├── iterative_random_search/
├── monte_carlo/
├── las_vegas/
├── mwc_redu/
├── max_clique_weight/
├── max_clique_dyn_weight/
├── wlmc/
├── tsm_mwc/
├── fast_wclq/
├── scc_walk/
└── mwc_peel/
```

### Complete Experiments Results

```
experiments/results_complete/
├── baseline/
├── [all algorithm directories...]
experiments/plots/
├── execution_time.png
├── operations_count.png
├── configurations_tested.png
├── heuristic_precision.png
└── ...
```

### Full Test Results

```
experiments/results_randomized/
├── exhaustive_baseline/
├── greedy_baseline_small/
├── greedy_baseline_large/
├── random_construction_small/
├── random_construction_large/
├── monte_carlo_small/
├── monte_carlo_large/
└── ...
```

---

## Algorithms Tested

All scripts test these 8 randomized algorithms:

1. `random_construction` - Basic random construction
2. `random_greedy_hybrid` - Hybrid random/greedy
3. `iterative_random_search` - Iterative random search
4. `monte_carlo` - Monte Carlo sampling
5. `las_vegas` - Las Vegas (always correct)
6. `mwc_redu` - Graph reduction preprocessing
7. `max_clique_weight` - Branch-and-bound (static bounds)
8. `max_clique_dyn_weight` - Branch-and-bound (dynamic bounds)

---

## Important Notes

### Graph Size Limits

- **Exhaustive search:** Automatically limited to ≤20 vertices
- **Randomized algorithms:** Tested on 20-50 vertices
- You can modify these limits in the scripts

### Time Limits

- Quick test: 10 seconds per algorithm
- Full test: 30-60 seconds per algorithm (depending on algorithm)
- Can be adjusted in the scripts

### Log Files

Full test suite creates timestamped log files:

```
experiments/test_run_YYYYMMDD_HHMMSS.log
```

---

## Troubleshooting

### Scripts not executable

```bash
chmod +x *.sh
```

### Out of memory

Reduce max vertices in the scripts:

- Change `--max-vertices 50` to `--max-vertices 30`

### Too slow

- Use `quick_test.sh` instead
- Reduce graph sizes
- Increase time limits to avoid timeouts

### Missing graphs

Graphs are auto-generated, but you can manually generate:

```bash
uv run python main.py generate \
    --seed 112974 \
    --min-vertices 4 \
    --max-vertices 20 \
    --output-dir experiments/graphs
```

---

## Next Steps After Testing

1. **View results:**

   ```bash
   cat experiments/results_quick/monte_carlo/benchmark_results.csv
   ```

2. **Visualize:**

   ```bash
   uv run python main.py visualize \
       experiments/results_quick/monte_carlo/benchmark_results.json
   ```

3. **Compare algorithms:**
   - Check precision percentages in CSV files
   - Compare execution times
   - Analyze operations counts

For detailed information, see `TESTING_GUIDE.md`.
