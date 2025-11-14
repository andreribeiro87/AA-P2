"""
Benchmarking infrastructure for Maximum Weight Clique algorithms.

Measures execution time, operations count, and heuristic precision.
"""

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import networkx as nx

from src.algorithms import MaxWeightCliqueSolver, compare_solutions


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single graph instance."""

    # Graph properties
    n_vertices: int
    n_edges: int
    edge_density_percent: float
    
    # Exhaustive search metrics (optional if only running heuristic)
    exact_clique_size: int | None = None
    exact_weight: float | None = None
    exact_time_seconds: float | None = None
    exact_operations: int | None = None
    exact_configurations: int | None = None
    
    # Greedy heuristic metrics (optional if only running exhaustive)
    greedy_clique_size: int | None = None
    greedy_weight: float | None = None
    greedy_time_seconds: float | None = None
    greedy_operations: int | None = None
    greedy_configurations: int | None = None
    
    # Comparison metrics (optional)
    precision_percent: float | None = None
    speedup_factor: float | None = None


class BenchmarkRunner:
    """Run benchmarks on graph instances and collect performance data."""

    def __init__(self, verbose: bool = True):
        """
        Initialize benchmark runner.

        Args:
            verbose: If True, print progress information
        """
        self.verbose = verbose

    def benchmark_graph(
        self, 
        graph: nx.Graph, 
        mode: str = "both"
    ) -> BenchmarkResult:
        """
        Run algorithm(s) on a graph and collect metrics.

        Args:
            graph: NetworkX graph to benchmark
            mode: "both", "exhaustive", or "heuristic"

        Returns:
            BenchmarkResult with metrics
        """
        solver = MaxWeightCliqueSolver(graph)
        
        # Calculate graph properties
        n_vertices = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        max_edges = n_vertices * (n_vertices - 1) // 2
        edge_density = (n_edges / max_edges * 100.0) if max_edges > 0 else 0.0

        # Make the edge density one of 12.5, 25.0, 50.0, 75.0, grab the closest one
        closest_density = min([12.5, 25.0, 50.0, 75.0], key=lambda x: abs(x - edge_density))

        if self.verbose:
            mode_str = {"both": "both algorithms", "exhaustive": "exhaustive search", "heuristic": "greedy heuristic"}[mode]
            print(f"Benchmarking graph ({mode_str}): {n_vertices} vertices, {n_edges} edges ({closest_density:.1f}% density)")

        # Initialize result
        result = BenchmarkResult(
            n_vertices=n_vertices,
            n_edges=n_edges,
            edge_density_percent=closest_density,
        )

        # Benchmark exhaustive search
        exact_result = None
        exact_time = None
        if mode in ["both", "exhaustive"]:
            if self.verbose:
                print("  Running exhaustive search...", end=" ", flush=True)
            start_time = time.perf_counter()
            exact_result = solver.exhaustive_search()
            exact_time = time.perf_counter() - start_time
            if self.verbose:
                print(f"✓ ({exact_time:.4f}s)")
            
            result.exact_clique_size = len(exact_result.clique)
            result.exact_weight = exact_result.total_weight
            result.exact_time_seconds = exact_time
            result.exact_operations = exact_result.basic_operations
            result.exact_configurations = exact_result.configurations_tested

        # Benchmark greedy heuristic
        greedy_result = None
        greedy_time = None
        if mode in ["both", "heuristic"]:
            if self.verbose:
                print("  Running greedy heuristic...", end=" ", flush=True)
            start_time = time.perf_counter()
            greedy_result = solver.greedy_heuristic()
            greedy_time = time.perf_counter() - start_time
            if self.verbose:
                print(f"✓ ({greedy_time:.4f}s)")
            
            result.greedy_clique_size = len(greedy_result.clique)
            result.greedy_weight = greedy_result.total_weight
            result.greedy_time_seconds = greedy_time
            result.greedy_operations = greedy_result.basic_operations
            result.greedy_configurations = greedy_result.configurations_tested

        # Calculate comparison metrics if both ran
        if mode == "both" and exact_result and greedy_result:
            comparison = compare_solutions(exact_result, greedy_result)
            result.precision_percent = comparison["precision_percent"]
            result.speedup_factor = (exact_time or 0) / (greedy_time or 1) if greedy_time and greedy_time > 0 else 0.0

        return result

    def benchmark_graph_file(self, filepath: Path, mode: str = "both") -> BenchmarkResult:
        """
        Load a graph from file and benchmark it.

        Args:
            filepath: Path to GraphML file
            mode: "both", "exhaustive", or "heuristic"

        Returns:
            BenchmarkResult with metrics
        """
        graph = nx.read_graphml(filepath)
        # Convert node labels to integers
        graph = nx.convert_node_labels_to_integers(graph)
        # Convert weight to float (GraphML saves as strings)
        for node in graph.nodes():
            graph.nodes[node]["weight"] = float(graph.nodes[node]["weight"])
        
        return self.benchmark_graph(graph, mode=mode)

    def benchmark_series(
        self,
        graph_files: list[Path],
        output_dir: Path = Path("experiments/results"),
        mode: str = "both",
    ) -> list[BenchmarkResult]:
        """
        Benchmark a series of graphs and save results.

        Args:
            graph_files: list of paths to graph files
            output_dir: Directory to save results
            mode: "both", "exhaustive", or "heuristic"

        Returns:
            list of BenchmarkResult objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[BenchmarkResult] = []

        mode_name = {"both": "BOTH ALGORITHMS", "exhaustive": "EXHAUSTIVE SEARCH", "heuristic": "GREEDY HEURISTIC"}[mode]
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {len(graph_files)} GRAPHS - {mode_name}")
        print(f"{'='*60}\n")

        for i, filepath in enumerate(graph_files, 1):
            if self.verbose:
                print(f"[{i}/{len(graph_files)}] {filepath.name}")
            
            try:
                result = self.benchmark_graph_file(filepath, mode=mode)
                results.append(result)
                
                if self.verbose:
                    if mode == "both" and result.precision_percent is not None:
                        print(f"  Precision: {result.precision_percent:.1f}%, "
                              f"Speedup: {result.speedup_factor:.1f}x\n")
                    elif mode == "exhaustive" and result.exact_time_seconds is not None:
                        print(f"  Time: {result.exact_time_seconds:.4f}s\n")
                    elif mode == "heuristic" and result.greedy_time_seconds is not None:
                        print(f"  Time: {result.greedy_time_seconds:.4f}s\n")
            
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                continue

        # Save results
        self._save_results(results, output_dir)

        return results

    def _save_results(self, results: list[BenchmarkResult], output_dir: Path) -> None:
        """
        Save benchmark results to CSV and JSON files.

        Args:
            results: list of benchmark results
            output_dir: Directory to save files
        """
        # Save as CSV
        csv_path = output_dir / "benchmark_results.csv"
        with open(csv_path, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for result in results:
                    writer.writerow(asdict(result))
        
        print(f"✓ Results saved to {csv_path}")

        # Save as JSON
        json_path = output_dir / "benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        print(f"✓ Results saved to {json_path}")

    def print_summary(self, results: list[BenchmarkResult]) -> None:
        """
        Print a summary of benchmark results.

        Args:
            results: list of benchmark results
        """
        if not results:
            print("No results to summarize.")
            return

        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}\n")

        print(f"Total graphs tested: {len(results)}")
        
        avg_precision = sum(r.precision_percent for r in results if r.precision_percent is not None) / len(results)
        print(f"Average greedy precision: {avg_precision:.2f}%")
        
        avg_speedup = sum(r.speedup_factor for r in results if r.speedup_factor is not None) / len(results)
        print(f"Average speedup: {avg_speedup:.2f}x")

        # Find largest graph tested
        largest = max(results, key=lambda r: r.n_vertices)
        print(f"\nLargest graph: {largest.n_vertices} vertices")
        print(f"  Exact time: {largest.exact_time_seconds or 0:.4f}s")
        print(f"  Greedy time: {largest.greedy_time_seconds or 0:.4f}s")
        print(f"  Precision: {largest.precision_percent or 0:.2f}%")

        # Performance by density
        print("\nPerformance by edge density:")
        densities: dict[float, list[BenchmarkResult]] = {}
        for result in results:
            density = result.edge_density_percent
            if density not in densities:
                densities[density] = []
            densities[density].append(result)
        
        for density in sorted(densities.keys()):
            results_at_density = densities[density]
            avg_prec = sum(r.precision_percent or 0 for r in results_at_density) / len(results_at_density)
            print(f"  {density:.1f}%: {avg_prec:.2f}% precision ({len(results_at_density)} graphs)")


def main() -> None:
    """Run benchmarks on generated graphs."""
    # Find all generated graphs
    graphs_dir = Path("experiments/graphs")
    
    if not graphs_dir.exists():
        print(f"Error: {graphs_dir} does not exist.")
        print("Run graph_generator.py first to generate graphs.")
        return

    graph_files = sorted(graphs_dir.glob("*.graphml"))
    
    if not graph_files:
        print(f"No graph files found in {graphs_dir}")
        return

    print(f"Found {len(graph_files)} graph files")

    # Run benchmarks
    runner = BenchmarkRunner(verbose=True)
    results = runner.benchmark_series(graph_files)

    # Print summary
    runner.print_summary(results)


if __name__ == "__main__":
    main()

