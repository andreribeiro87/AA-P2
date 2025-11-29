# Randomized Algorithms Usage Examples

This guide provides examples for running the randomized algorithms implemented for the Maximum Weight Clique problem.

## Table of Contents
1. [Basic Usage - Single Graph](#basic-usage---single-graph)
2. [Randomized Algorithm Strategies](#randomized-algorithm-strategies)
3. [Parameter Tuning](#parameter-tuning)
4. [Benchmarking](#benchmarking)
5. [Comparison with Other Algorithms](#comparison-with-other-algorithms)

---

## Basic Usage - Single Graph

### Run Only Randomized Algorithm

```bash
# Run default random_greedy_hybrid strategy on a single graph
python main.py solve experiments/graphs/graph_n10_d50.graphml --mode random

# Run random_construction strategy
python main.py solve experiments/graphs/graph_n10_d50.graphml --mode random --random-strategy random_construction

# Run iterative_random_search strategy
python main.py solve experiments/graphs/graph_n10_d50.graphml --mode random --random-strategy iterative_random_search
```

### Run All Algorithms (Exact + Greedy + Random)

```bash
# Compare all three approaches
python main.py solve experiments/graphs/graph_n10_d50.graphml --mode all

# With visualization
python main.py solve experiments/graphs/graph_n10_d50.graphml --mode all --visualize
```

---

## Randomized Algorithm Strategies

### 1. Random Construction (`random_construction`)

This strategy randomly constructs cliques by starting from random vertices and adding compatible vertices.

```bash
# Basic usage
python main.py solve graph.graphml --mode random --random-strategy random_construction

# With iteration limit (stop after 1000 iterations)
python main.py solve graph.graphml --mode random --random-strategy random_construction --max-iterations 1000

# With time limit (stop after 5 seconds)
python main.py solve graph.graphml --mode random --random-strategy random_construction --time-limit 5.0

# With both limits and seed for reproducibility
python main.py solve graph.graphml --mode random --random-strategy random_construction \
    --max-iterations 500 --time-limit 10.0 --seed 42
```

### 2. Random Greedy Hybrid (`random_greedy_hybrid`)

Combines randomness with greedy selection. This is the **default strategy**.

```bash
# Basic usage (default: 10 starts, top_k=3, randomness_factor=0.5)
python main.py solve graph.graphml --mode random

# Increase number of random starts (more exploration)
python main.py solve graph.graphml --mode random --num-starts 50

# More greedy (less random) - randomness_factor closer to 0
python main.py solve graph.graphml --mode random --randomness-factor 0.2

# More random (less greedy) - randomness_factor closer to 1
python main.py solve graph.graphml --mode random --randomness-factor 0.8

# Consider top 5 candidates instead of top 3
python main.py solve graph.graphml --mode random --top-k 5

# Full example with all parameters
python main.py solve graph.graphml --mode random \
    --random-strategy random_greedy_hybrid \
    --num-starts 20 \
    --top-k 5 \
    --randomness-factor 0.6 \
    --seed 12345
```

### 3. Iterative Random Search (`iterative_random_search`)

Tests random subsets of different sizes, starting from larger sizes.

```bash
# Basic usage
python main.py solve graph.graphml --mode random --random-strategy iterative_random_search

# With iteration limit
python main.py solve graph.graphml --mode random --random-strategy iterative_random_search \
    --max-iterations 2000

# With time limit
python main.py solve graph.graphml --mode random --random-strategy iterative_random_search \
    --time-limit 15.0

# With seed for reproducibility
python main.py solve graph.graphml --mode random --random-strategy iterative_random_search \
    --seed 42
```

---

## Parameter Tuning

### Reproducibility with Seeds

```bash
# Use seed for reproducible results
python main.py solve graph.graphml --mode random --seed 112974

# Same seed = same results
python main.py solve graph.graphml --mode random --seed 112974
```

### Time Limits

```bash
# Quick test (1 second)
python main.py solve graph.graphml --mode random --time-limit 1.0

# Medium test (10 seconds)
python main.py solve graph.graphml --mode random --time-limit 10.0

# Long test (60 seconds)
python main.py solve graph.graphml --mode random --time-limit 60.0
```

### Iteration Limits

```bash
# Quick test (100 iterations)
python main.py solve graph.graphml --mode random --max-iterations 100

# Medium test (1000 iterations)
python main.py solve graph.graphml --mode random --max-iterations 1000

# Extensive test (10000 iterations)
python main.py solve graph.graphml --mode random --max-iterations 10000
```

---

## Benchmarking

### Benchmark with Randomized Algorithms

```bash
# Run randomized algorithm on all graphs
python main.py benchmark --random

# Run randomized algorithm with specific strategy
python main.py benchmark --random --random-strategy random_construction

# Run randomized algorithm with custom parameters
python main.py benchmark --random \
    --random-strategy random_greedy_hybrid \
    --random-num-starts 20 \
    --random-top-k 5 \
    --random-factor 0.6

# Run randomized algorithm with time limit
python main.py benchmark --random \
    --random-strategy random_construction \
    --random-time-limit 5.0

# Run randomized algorithm with iteration limit
python main.py benchmark --random \
    --random-strategy iterative_random_search \
    --random-max-iterations 1000

# Run all algorithms (exact + greedy + random) and generate plots
python main.py benchmark --random --plot
```

### Benchmark on Specific Graph Ranges

```bash
# Exhaustive on small graphs, Greedy on medium, Random on all
python main.py benchmark \
    --exhaustive 4..15 \
    --heuristic 4..50 \
    --random \
    --random-strategy random_greedy_hybrid

# All algorithms on same range
python main.py benchmark \
    --exhaustive 4..12 \
    --heuristic 4..12 \
    --random \
    --random-strategy random_greedy_hybrid
```

### Benchmark with Reproducibility

```bash
# Use seed for reproducible benchmarks
python main.py benchmark --random --random-seed 112974 --plot
```

---

## Comparison with Other Algorithms

### Compare Random vs Exact

```bash
# Run both and compare
python main.py solve graph.graphml --mode all

# Output will show:
# - Exact solution (optimal)
# - Greedy solution
# - Random solution
# - Precision comparisons
```

### Compare Different Random Strategies

```bash
# Test random_construction
python main.py solve graph.graphml --mode random --random-strategy random_construction --seed 42

# Test random_greedy_hybrid
python main.py solve graph.graphml --mode random --random-strategy random_greedy_hybrid --seed 42

# Test iterative_random_search
python main.py solve graph.graphml --mode random --random-strategy iterative_random_search --seed 42
```

---

## Complete Example Workflow

### 1. Generate Test Graphs

```bash
python main.py generate --seed 112974 --min-vertices 4 --max-vertices 20
```

### 2. Test Randomized Algorithm on Single Graph

```bash
python main.py solve experiments/graphs/graph_n10_d50.graphml \
    --mode random \
    --random-strategy random_greedy_hybrid \
    --num-starts 20 \
    --top-k 5 \
    --randomness-factor 0.6 \
    --seed 42 \
    --visualize
```

### 3. Benchmark All Algorithms

```bash
python main.py benchmark \
    --exhaustive 4..15 \
    --heuristic 4..30 \
    --random \
    --random-strategy random_greedy_hybrid \
    --random-num-starts 20 \
    --random-seed 42 \
    --plot
```

### 4. Visualize Results

```bash
python main.py visualize experiments/results/benchmark_results.json
```

---

## Parameter Recommendations

### For Small Graphs (n < 15)
```bash
python main.py solve graph.graphml --mode random \
    --random-strategy random_greedy_hybrid \
    --num-starts 10 \
    --top-k 3 \
    --randomness-factor 0.5
```

### For Medium Graphs (15 ≤ n < 50)
```bash
python main.py solve graph.graphml --mode random \
    --random-strategy random_greedy_hybrid \
    --num-starts 20 \
    --top-k 5 \
    --randomness-factor 0.6 \
    --time-limit 10.0
```

### For Large Graphs (n ≥ 50)
```bash
python main.py solve graph.graphml --mode random \
    --random-strategy random_greedy_hybrid \
    --num-starts 50 \
    --top-k 10 \
    --randomness-factor 0.7 \
    --time-limit 30.0
```

### For Quick Testing
```bash
python main.py solve graph.graphml --mode random \
    --random-strategy random_construction \
    --max-iterations 100 \
    --time-limit 2.0
```

---

## Understanding Output

When running randomized algorithms, you'll see:

```
RANDOMIZED ALGORITHM: RANDOM_GREEDY_HYBRID
============================================================
Clique: {0, 2, 5, 7}
Clique size: 4
Total weight: 245.67
Basic operations: 1,234
Configurations tested: 20
Unique configurations: 18
Duplicates: 2
Stopping reason: exhausted
```

- **Clique**: The found maximum weight clique
- **Total weight**: Sum of vertex weights in the clique
- **Basic operations**: Number of edge checks performed
- **Configurations tested**: Total number of configurations tried
- **Unique configurations**: Number of unique configurations (no duplicates)
- **Duplicates**: Number of duplicate configurations skipped
- **Stopping reason**: Why the algorithm stopped (max_iterations, time_limit, no_improvement, exhausted)

---

## Tips

1. **Start with defaults**: Try `--mode random` first to see how it performs
2. **Use seeds for comparison**: Always use `--seed` when comparing strategies
3. **Time limits for large graphs**: Use `--time-limit` instead of `--max-iterations` for large graphs
4. **More starts = better quality**: Increase `--num-starts` for better solutions (but slower)
5. **Balance randomness**: Adjust `--randomness-factor` based on graph structure:
   - Dense graphs: lower factor (more greedy)
   - Sparse graphs: higher factor (more random)

