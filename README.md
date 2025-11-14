# Maximum Weight Clique Solver

**Projeto 1 de Algoritmos Avançados (2025/2026)**

## Report

The complete project report is available in PDF format:
- **Report PDF**: [`docs/report/112974_AA_Project_1.pdf`](docs/report/112974_AA_Project_1.pdf)

The report includes:
- Problem formulation and theoretical background
- Detailed algorithm descriptions and complexity analysis
- Experimental methodology and results
- Performance analysis and comparisons
- Conclusions and future work

**GitHub Repository**: [https://github.com/andreribeiro87/AA-P1](https://github.com/andreribeiro87/AA-P1)

## Problem Description

**Problem 12 - Maximum Weight Clique**: Find a maximum weight clique for a given undirected graph G(V, E), whose vertices carry positive weights, with n vertices and m edges.

- A **clique** of G is a subset of vertices, all adjacent to each other (forming a complete subgraph)
- The **weight of a clique** is the sum of its vertices' weights
- A **maximum weight clique** is a clique whose total weight is as large as possible

This is an NP-hard optimization problem with applications in social network analysis, bioinformatics, and pattern recognition.

## Algorithms Implemented

### 1. Exhaustive Search (Exact Algorithm)

The exhaustive search algorithm finds the optimal solution by testing all possible vertex subsets:

1. Generate all possible subsets of vertices (2^n configurations)
2. For each subset, check if it forms a clique (all vertices are pairwise adjacent)
3. Calculate the total weight of valid cliques
4. Return the clique with maximum weight

**Time Complexity**: O(2^n × n²) where n is the number of vertices
- 2^n subsets to test
- O(n²) to verify if each subset is a clique

**Space Complexity**: O(n) for storing the current best clique

### 2. Greedy Heuristic (Approximation Algorithm)

The greedy heuristic provides fast approximate solutions:

**Strategy**: Multi-start greedy construction
1. Try starting from each vertex in the graph
2. For each starting vertex:
   - Initialize clique with that vertex
   - Iteratively add the highest-weight compatible vertex
   - A vertex is compatible if it's adjacent to all current clique members
   - Stop when no compatible vertices remain
3. Return the best clique found across all starting points

**Rationale**: 
- Single-start greedy can get stuck in local optima depending on the starting vertex
- Multi-start approach explores multiple construction paths
- Starting from different vertices helps find better solutions in dense graphs
- Still maintains polynomial time complexity

**Time Complexity**: O(n³) where n is the number of vertices
- O(n) starting points
- For each start: O(n²) to build clique (n iterations × n adjacency checks)

**Space Complexity**: O(n) for storing cliques

**Expected Precision**: 70-100% depending on graph structure and density

## Project Structure

```
projeto1/
├── src/
│   ├── graph_generator.py    # Random graph generation with 2D point vertices
│   ├── algorithms.py         # Exhaustive search and greedy heuristic implementations
│   ├── benchmark.py          # Performance measurement and analysis
│   └── visualizer.py         # Graph and results visualization
├── main.py                   # CLI interface
├── experiments/
│   ├── graphs/              # Generated graph instances (.graphml)
│   ├── results/             # Benchmark results (.csv, .json)
│   └── plots/               # Performance visualization plots (.png)
├── pyproject.toml           # Project dependencies
└── README.md                # This file
```

## Dependencies

- Python ≥ 3.13
- NetworkX ≥ 3.5 (graph data structures and algorithms)
- Matplotlib (visualization)
- NumPy (numerical computations for plots)
- Typer (command-line interface framework)
- Pandas (data analysis, if needed)

Install dependencies:
```bash
pip install networkx matplotlib numpy typer
```

Or using uv (recommended):
```bash
uv sync
```

This will install all dependencies specified in `pyproject.toml`.

## Usage

### 1. Generate Random Graphs

Generate graphs with specified parameters:

```bash
python main.py generate --seed 12345 --min-vertices 4 --max-vertices 12
```

Options:
- `--seed`: Random seed for reproducibility (default: 112974)
- `--min-vertices`: Minimum number of vertices (default: 4)
- `--max-vertices`: Maximum number of vertices (default: 12)
- `--densities`: Edge density percentages (default: 12.5 25 50 75)
- `--output-dir`: Output directory (default: experiments/graphs)

For each vertex count, generates 4 graphs with different edge densities (12.5%, 25%, 50%, 75%).

### 2. Solve a Single Graph

Find the maximum weight clique for a specific graph:

```bash
python main.py solve experiments/graphs/graph_n6_d50.graphml --visualize
```

Options:
- `--visualize`: Generate a visualization of the solution
- `--no-show`: Save visualization to file without displaying
- `--mode`: Algorithm mode to run - `both`, `exhaustive`, or `heuristic` (default: `both`)

### 3. Run Benchmarks

Benchmark all generated graphs and collect performance metrics:

```bash
python main.py benchmark --plot
```

**Separate Algorithm Ranges (Advanced):**

Since exhaustive search becomes impractical for large graphs while the greedy heuristic can handle much larger instances, you can benchmark them separately with different vertex ranges:

```bash
# Exhaustive search on small graphs (4-15 vertices)
# Greedy heuristic on larger graphs (4-50 vertices)
python main.py benchmark --exhaustive 4..15 --heuristic 4..50 --plot
```

This is useful to:
- See where exhaustive search becomes too slow
- Evaluate how the greedy heuristic performs on much larger graphs
- Generate plots comparing both algorithms across their practical ranges

Options:
- `--graphs-dir`: Directory containing graphs (default: experiments/graphs)
- `--output-dir`: Output directory for results (default: experiments/results)
- `--exhaustive`: Vertex range for exhaustive search (e.g., `4..15` or `all` for all graphs)
- `--heuristic`: Vertex range for greedy heuristic (e.g., `4..100` or `all` for all graphs)
- `--plot`: Generate performance plots after benchmarking
- `--verbose`: Print detailed progress (default: True)

By default, both flags are set to `all`, meaning both algorithms run on all available graphs.

Outputs:
- `benchmark_results.csv`: Results in tabular format
- `benchmark_results.json`: Results in JSON format
- Performance plots (if `--plot` is used)
- When using separate ranges, results are saved in subdirectories (`exhaustive/` and `heuristic/`)

### 4. Visualize Results

Generate plots from existing benchmark results:

```bash
python main.py visualize experiments/results/benchmark_results.json
```

Or use the default path:
```bash
python main.py visualize
```

Options:
- `results`: Path to results JSON file (default: experiments/results/benchmark_results.json)
- `--output-dir`: Output directory for plots (default: experiments/plots)

Generated plots:
- `execution_time.png`: Algorithm execution time vs. graph size (two subplots: exhaustive and greedy)
- `operations_count.png`: Basic operations count vs. graph size (exhaustive search)
- `configurations_tested.png`: Number of tested configurations vs. graph size (exhaustive search)
- `heuristic_precision.png`: Greedy heuristic precision vs. graph size

## Experimental Methodology

### Graph Generation
- Vertices are 2D points with integer coordinates between 1 and 500
- Minimum distance between vertices enforced (≥10 units)
- Each vertex assigned a random weight between 1.0 and 100.0
- Edges randomly selected to achieve target density

### Metrics Collected
1. **Execution Time**: Wall-clock time using `time.perf_counter()`
2. **Basic Operations**: Number of edge adjacency checks
3. **Configurations Tested**: Number of vertex subsets examined
4. **Heuristic Precision**: `(greedy_weight / optimal_weight) × 100%`

### Analysis Goals
- Empirically verify theoretical time complexity
- Determine practical limits of exhaustive search
- Evaluate greedy heuristic quality vs. speed tradeoff
- Analyze impact of graph density on performance

## Example Workflow

Complete workflow for computational experiments:

```bash
# 1. Generate graphs (seed = your student number)
# Generate up to 30 vertices to test heuristic on large graphs
python main.py generate --seed 112974 --min-vertices 4 --max-vertices 30

# 2. Run benchmarks with separate ranges for each algorithm
# Exhaustive: 4-15 vertices (practical limit)
# Heuristic: 4-30 vertices (show it can handle larger graphs)
python main.py benchmark --exhaustive 4..15 --heuristic 4..30 --plot

# 3. Solve and visualize a specific instance
python main.py solve experiments/graphs/graph_n8_d50.graphml --visualize

# 4. Generate additional plots from results
python main.py visualize
```

**Alternative: Traditional Workflow (both algorithms on same range)**

```bash
# Generate smaller set of graphs
python main.py generate --seed 112974 --min-vertices 4 --max-vertices 14

# Run both algorithms on all graphs
python main.py benchmark --plot
```

## Computational Complexity Analysis

### Exhaustive Search
- **Theoretical**: O(2^n × n²)
- **Expected behavior**: Exponential growth
- **Practical limit**: ~15-20 vertices (depending on hardware)

### Greedy Heuristic
- **Theoretical**: O(n³)
- **Expected behavior**: Polynomial growth
- **Practical limit**: Thousands of vertices

### Empirical Verification
Run benchmarks to observe:
- Execution time doubling with each additional vertex (exhaustive)
- Polynomial scaling of greedy heuristic
- Exponential growth in configurations tested (2^n)

## Author

André 112974 - Universidade de Aveiro (2025/2026)
