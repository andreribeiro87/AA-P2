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

from src.algorithms import (
    AlgorithmResult,
    MaxWeightCliqueSolver,
    RandomizedAlgorithmResult,
    StoppingReason,
    compare_solutions,
)


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
    exact_clique: str | None = None  # Comma-separated vertex IDs

    # Greedy heuristic metrics (optional if only running exhaustive)
    greedy_clique_size: int | None = None
    greedy_weight: float | None = None
    greedy_time_seconds: float | None = None
    greedy_operations: int | None = None
    greedy_configurations: int | None = None
    greedy_clique: str | None = None  # Comma-separated vertex IDs

    # Comparison metrics (optional)
    precision_percent: float | None = None
    speedup_factor: float | None = None

    # Randomized algorithm metrics (optional)
    random_clique_size: int | None = None
    random_weight: float | None = None
    random_time_seconds: float | None = None
    random_operations: int | None = None
    random_configurations: int | None = None
    random_unique_configurations: int | None = None
    random_duplicates: int | None = None
    random_stopping_reason: str | None = None
    random_precision_percent: float | None = None
    random_clique: str | None = None  # Comma-separated vertex IDs


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
        mode: str = "both",
        random_strategy: str | None = None,
        random_params: dict | None = None,
        reduction_strategy: str | None = None,
        reduction_params: dict | None = None,
    ) -> BenchmarkResult:
        """
        Run algorithm(s) on a graph and collect metrics.

        Args:
            graph: NetworkX graph to benchmark
            mode: "both", "exhaustive", "heuristic", "random", or "all"
            random_strategy: "random_construction", "random_greedy_hybrid", or "iterative_random_search"
            random_params: Dictionary of parameters for randomized algorithm

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
        closest_density = min(
            [12.5, 25.0, 50.0, 75.0], key=lambda x: abs(x - edge_density)
        )

        if self.verbose:
            mode_str_map = {
                "both": "both algorithms",
                "exhaustive": "exhaustive search",
                "heuristic": "greedy heuristic",
                "random": "randomized algorithm",
                "reduction": "reduction/exact algorithm",
                "all": "all algorithms",
            }
            mode_str = mode_str_map.get(mode, mode)
            print(
                f"Benchmarking graph ({mode_str}): {n_vertices} vertices, {n_edges} edges ({closest_density:.1f}% density)"
            )

        # Initialize result
        result = BenchmarkResult(
            n_vertices=n_vertices,
            n_edges=n_edges,
            edge_density_percent=closest_density,
        )

        # Benchmark exhaustive search
        exact_result = None
        exact_time = None
        if mode in ["both", "exhaustive", "all"]:
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
            result.exact_clique = ",".join(str(v) for v in sorted(exact_result.clique))

        # Benchmark greedy heuristic
        greedy_result = None
        greedy_time = None
        if mode in ["both", "heuristic", "all"]:
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
            result.greedy_clique = ",".join(
                str(v) for v in sorted(greedy_result.clique)
            )

        # Benchmark randomized algorithm
        random_result = None
        random_time = None
        if mode in ["random", "all"]:
            if random_strategy is None:
                random_strategy = "random_greedy_hybrid"  # Default strategy

            if random_params is None:
                random_params = {}

            if self.verbose:
                print(f"  Running {random_strategy}...", end=" ", flush=True)

            start_time = time.perf_counter()

            if random_strategy == "random_construction":
                # Filter parameters that random_construction accepts
                filtered_params = {
                    k: v
                    for k, v in random_params.items()
                    if k in ["seed", "max_iterations"]
                }
                random_result = solver.random_construction(**filtered_params)
            elif random_strategy == "random_greedy_hybrid":
                # Filter parameters that random_greedy_hybrid accepts (no time_limit)
                filtered_params = {
                    k: v
                    for k, v in random_params.items()
                    if k in ["num_starts", "top_k", "randomness_factor", "seed"]
                }
                random_result = solver.random_greedy_hybrid(**filtered_params)
            elif random_strategy == "iterative_random_search":
                # Filter parameters that iterative_random_search accepts
                filtered_params = {
                    k: v
                    for k, v in random_params.items()
                    if k in ["seed", "max_iterations"]
                }
                random_result = solver.iterative_random_search(**filtered_params)
            elif random_strategy == "monte_carlo":
                random_result = solver.monte_carlo(**random_params)
            elif random_strategy == "las_vegas":
                random_result = solver.las_vegas(**random_params)
            # New algorithms from literature
            elif random_strategy == "wlmc":
                # WLMC - exact BnB, doesn't use seed
                filtered_params = {
                    k: v
                    for k, v in random_params.items()
                    if k in ["time_limit", "use_preprocessing"]
                }
                algo_result = solver.wlmc(**filtered_params)
                # Convert to RandomizedAlgorithmResult for consistency
                random_result = RandomizedAlgorithmResult(
                    clique=algo_result.clique,
                    total_weight=algo_result.total_weight,
                    basic_operations=algo_result.basic_operations,
                    configurations_tested=algo_result.configurations_tested,
                    unique_configurations_tested=algo_result.configurations_tested,
                    duplicate_count=0,
                    stopping_reason=StoppingReason.EXHAUSTED,
                )
            elif random_strategy == "tsm_mwc":
                # TSM-MWC - exact BnB, doesn't use seed
                filtered_params = {
                    k: v for k, v in random_params.items() if k in ["time_limit"]
                }
                algo_result = solver.tsm_mwc(**filtered_params)
                # Convert to RandomizedAlgorithmResult for consistency
                random_result = RandomizedAlgorithmResult(
                    clique=algo_result.clique,
                    total_weight=algo_result.total_weight,
                    basic_operations=algo_result.basic_operations,
                    configurations_tested=algo_result.configurations_tested,
                    unique_configurations_tested=algo_result.configurations_tested,
                    duplicate_count=0,
                    stopping_reason=StoppingReason.EXHAUSTED,
                )
            elif random_strategy == "fast_wclq":
                # FastWClq - semi-exact heuristic with graph reduction
                filtered_params = {
                    k: v
                    for k, v in random_params.items()
                    if k in ["seed", "max_iterations", "time_limit", "bms_k"]
                }
                random_result = solver.fast_wclq(**filtered_params)
            elif random_strategy == "scc_walk":
                # SCCWalk - local search with Strong Configuration Checking
                filtered_params = {
                    k: v
                    for k, v in random_params.items()
                    if k
                    in [
                        "seed",
                        "max_iterations",
                        "time_limit",
                        "max_unimprove_steps",
                        "walk_perturbation_strength",
                    ]
                }
                random_result = solver.scc_walk(**filtered_params)
            elif random_strategy == "mwc_peel":
                # MWCPeel - hybrid reduction with peeling
                filtered_params = {
                    k: v
                    for k, v in random_params.items()
                    if k in ["seed", "max_iterations", "time_limit", "peel_fraction"]
                }
                random_result = solver.mwc_peel(**filtered_params)
            else:
                raise ValueError(f"Unknown random strategy: {random_strategy}")

            random_time = time.perf_counter() - start_time

            if self.verbose:
                print(f"✓ ({random_time:.4f}s)")

            result.random_clique_size = len(random_result.clique)
            result.random_weight = random_result.total_weight
            result.random_time_seconds = random_time
            result.random_operations = random_result.basic_operations
            result.random_configurations = random_result.configurations_tested
            result.random_unique_configurations = (
                random_result.unique_configurations_tested
            )
            result.random_duplicates = random_result.duplicate_count
            result.random_stopping_reason = (
                random_result.stopping_reason.value
                if random_result.stopping_reason
                else None
            )
            result.random_clique = ",".join(
                str(v) for v in sorted(random_result.clique)
            )

            # Calculate precision compared to exact result if available
            if exact_result and exact_result.total_weight > 0:
                result.random_precision_percent = (
                    random_result.total_weight / exact_result.total_weight * 100.0
                )
            elif greedy_result and greedy_result.total_weight > 0:
                result.random_precision_percent = (
                    random_result.total_weight / greedy_result.total_weight * 100.0
                )

        # Benchmark reduction/exact algorithms (deterministic, NOT random)
        reduction_result = None
        reduction_time = None
        if mode == "reduction":
            if reduction_strategy is None:
                reduction_strategy = "mwc_redu"  # Default strategy

            if reduction_params is None:
                reduction_params = {}

            if self.verbose:
                print(f"  Running {reduction_strategy}...", end=" ", flush=True)

            start_time = time.perf_counter()

            if reduction_strategy == "mwc_redu":
                algo_result = solver.mwc_redu(**reduction_params)
                # mwc_redu can return either AlgorithmResult or RandomizedAlgorithmResult
                if isinstance(algo_result, RandomizedAlgorithmResult):
                    reduction_result = algo_result
                else:
                    # Convert to RandomizedAlgorithmResult for consistency
                    reduction_result = RandomizedAlgorithmResult(
                        clique=algo_result.clique,
                        total_weight=algo_result.total_weight,
                        basic_operations=algo_result.basic_operations,
                        configurations_tested=algo_result.configurations_tested,
                        unique_configurations_tested=algo_result.configurations_tested,
                        duplicate_count=0,
                        stopping_reason=StoppingReason.EXHAUSTED,
                    )
            elif reduction_strategy == "max_clique_weight":
                algo_result = solver.max_clique_weight(**reduction_params)
                # Convert to RandomizedAlgorithmResult for consistency
                reduction_result = RandomizedAlgorithmResult(
                    clique=algo_result.clique,
                    total_weight=algo_result.total_weight,
                    basic_operations=algo_result.basic_operations,
                    configurations_tested=algo_result.configurations_tested,
                    unique_configurations_tested=algo_result.configurations_tested,
                    duplicate_count=0,
                    stopping_reason=StoppingReason.EXHAUSTED,
                )
            elif reduction_strategy == "max_clique_dyn_weight":
                algo_result = solver.max_clique_dyn_weight(**reduction_params)
                # Convert to RandomizedAlgorithmResult for consistency
                reduction_result = RandomizedAlgorithmResult(
                    clique=algo_result.clique,
                    total_weight=algo_result.total_weight,
                    basic_operations=algo_result.basic_operations,
                    configurations_tested=algo_result.configurations_tested,
                    unique_configurations_tested=algo_result.configurations_tested,
                    duplicate_count=0,
                    stopping_reason=StoppingReason.EXHAUSTED,
                )
            else:
                raise ValueError(f"Unknown reduction strategy: {reduction_strategy}")

            reduction_time = time.perf_counter() - start_time

            if self.verbose:
                print(f"✓ ({reduction_time:.4f}s)")

            # Use random_* fields for reduction algorithms too (for consistency in BenchmarkResult)
            result.random_clique_size = len(reduction_result.clique)
            result.random_weight = reduction_result.total_weight
            result.random_time_seconds = reduction_time
            result.random_operations = reduction_result.basic_operations
            result.random_configurations = reduction_result.configurations_tested
            result.random_unique_configurations = (
                reduction_result.unique_configurations_tested
            )
            result.random_duplicates = reduction_result.duplicate_count
            result.random_stopping_reason = (
                reduction_result.stopping_reason.value
                if reduction_result.stopping_reason
                else None
            )
            result.random_clique = ",".join(
                str(v) for v in sorted(reduction_result.clique)
            )

            # Calculate precision compared to exact result if available
            if exact_result and exact_result.total_weight > 0:
                result.random_precision_percent = (
                    reduction_result.total_weight / exact_result.total_weight * 100.0
                )
            elif greedy_result and greedy_result.total_weight > 0:
                result.random_precision_percent = (
                    reduction_result.total_weight / greedy_result.total_weight * 100.0
                )

        # Calculate comparison metrics if both ran
        if mode == "both" and exact_result and greedy_result:
            comparison = compare_solutions(exact_result, greedy_result)
            result.precision_percent = comparison["precision_percent"]
            result.speedup_factor = (
                (exact_time or 0) / (greedy_time or 1)
                if greedy_time and greedy_time > 0
                else 0.0
            )

        return result

    def benchmark_graph_file(
        self,
        filepath: Path,
        mode: str = "both",
        random_strategy: str | None = None,
        random_params: dict | None = None,
        reduction_strategy: str | None = None,
        reduction_params: dict | None = None,
    ) -> BenchmarkResult:
        """
        Load a graph from file and benchmark it.

        Loads graph on-demand (one at a time) for memory efficiency.

        Args:
            filepath: Path to graph file (.graphml, .clq, .dimacs, etc.)
            mode: "both", "exhaustive", "heuristic", "random", or "all"
            random_strategy: Randomized algorithm strategy
            random_params: Parameters for randomized algorithm

        Returns:
            BenchmarkResult with metrics
        """
        # Use graph loader to support multiple formats and handle weights
        from src.graph_loader import BenchmarkGraphLoader

        loader = BenchmarkGraphLoader()
        graph = loader.load_graph(filepath)

        return self.benchmark_graph(
            graph,
            mode=mode,
            random_strategy=random_strategy,
            random_params=random_params,
            reduction_strategy=reduction_strategy,
            reduction_params=reduction_params,
        )

    def benchmark_series(
        self,
        graph_files: list[Path],
        output_dir: Path = Path("experiments/results"),
        mode: str = "both",
        random_strategy: str | None = None,
        random_params: dict | None = None,
        reduction_strategy: str | None = None,
        reduction_params: dict | None = None,
    ) -> list[BenchmarkResult]:
        """
        Benchmark a series of graphs and save results.

        Args:
            graph_files: list of paths to graph files
            output_dir: Directory to save results
            mode: "both", "exhaustive", "heuristic", "random", or "all"
            random_strategy: Randomized algorithm strategy
            random_params: Parameters for randomized algorithm

        Returns:
            list of BenchmarkResult objects
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[BenchmarkResult] = []

        mode_name_map = {
            "both": "BOTH ALGORITHMS",
            "exhaustive": "EXHAUSTIVE SEARCH",
            "heuristic": "GREEDY HEURISTIC",
            "random": "RANDOMIZED ALGORITHM",
            "reduction": "REDUCTION/EXACT ALGORITHM",
            "all": "ALL ALGORITHMS",
        }
        mode_name = mode_name_map.get(mode, mode.upper())
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING {len(graph_files)} GRAPHS - {mode_name}")
        print(f"{'=' * 60}\n")

        for i, filepath in enumerate(graph_files, 1):
            if self.verbose:
                print(f"[{i}/{len(graph_files)}] {filepath.name}")

            try:
                result = self.benchmark_graph_file(
                    filepath,
                    mode=mode,
                    random_strategy=random_strategy,
                    random_params=random_params,
                    reduction_strategy=reduction_strategy,
                    reduction_params=reduction_params,
                )
                results.append(result)

                if self.verbose:
                    if mode == "both" and result.precision_percent is not None:
                        print(
                            f"  Precision: {result.precision_percent:.1f}%, "
                            f"Speedup: {result.speedup_factor:.1f}x\n"
                        )
                    elif mode == "exhaustive" and result.exact_time_seconds is not None:
                        print(f"  Time: {result.exact_time_seconds:.4f}s\n")
                    elif mode == "heuristic" and result.greedy_time_seconds is not None:
                        print(f"  Time: {result.greedy_time_seconds:.4f}s\n")
                    elif mode == "random" and result.random_time_seconds is not None:
                        precision_str = (
                            f", Precision: {result.random_precision_percent:.1f}%"
                            if result.random_precision_percent
                            else ""
                        )
                        print(
                            f"  Time: {result.random_time_seconds:.4f}s{precision_str}\n"
                        )
                    elif mode == "reduction" and result.random_time_seconds is not None:
                        # Use random_time_seconds for reduction algorithms too (it's the algorithm time)
                        precision_str = (
                            f", Precision: {result.random_precision_percent:.1f}%"
                            if result.random_precision_percent
                            else ""
                        )
                        print(
                            f"  Time: {result.random_time_seconds:.4f}s{precision_str}\n"
                        )
                    elif mode == "all":
                        info_parts = []
                        if result.exact_time_seconds is not None:
                            info_parts.append(
                                f"Exact: {result.exact_time_seconds:.4f}s"
                            )
                        if result.greedy_time_seconds is not None:
                            info_parts.append(
                                f"Greedy: {result.greedy_time_seconds:.4f}s"
                            )
                        if result.random_time_seconds is not None:
                            info_parts.append(
                                f"Random: {result.random_time_seconds:.4f}s"
                            )
                        if info_parts:
                            print(f"  {' | '.join(info_parts)}\n")

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
        with open(csv_path, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for result in results:
                    writer.writerow(asdict(result))

        print(f"✓ Results saved to {csv_path}")

        # Save as JSON
        json_path = output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
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

        print(f"\n{'=' * 60}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 60}\n")

        print(f"Total graphs tested: {len(results)}")

        avg_precision = sum(
            r.precision_percent for r in results if r.precision_percent is not None
        ) / len(results)
        print(f"Average greedy precision: {avg_precision:.2f}%")

        avg_speedup = sum(
            r.speedup_factor for r in results if r.speedup_factor is not None
        ) / len(results)
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
            avg_prec = sum(r.precision_percent or 0 for r in results_at_density) / len(
                results_at_density
            )
            print(
                f"  {density:.1f}%: {avg_prec:.2f}% precision ({len(results_at_density)} graphs)"
            )


# Algorithm categories for organization
ALGORITHM_CATEGORIES = {
    "exact": ["exhaustive"],
    "greedy": ["greedy"],
    "randomized": [
        "random_construction",
        "random_greedy_hybrid",
        "iterative_random_search",
        "monte_carlo",
        "las_vegas",
    ],
    "reduction": ["mwc_redu", "max_clique_weight", "max_clique_dyn_weight"],
    "branch_bound": ["wlmc", "tsm_mwc"],
    "heuristic": ["fast_wclq", "scc_walk", "mwc_peel"],
}

ALL_ALGORITHMS = [
    "exhaustive",
    "greedy",
    "random_construction",
    "random_greedy_hybrid",
    "iterative_random_search",
    "monte_carlo",
    "las_vegas",
    "mwc_redu",
    "max_clique_weight",
    "max_clique_dyn_weight",
    "wlmc",
    "tsm_mwc",
    "fast_wclq",
    "scc_walk",
    "mwc_peel",
]


def get_algorithm_config(
    algorithm: str, common_params: dict | None = None
) -> tuple[str, str | None, dict]:
    """
    Get the mode, strategy, and parameters for an algorithm.

    Args:
        algorithm: Algorithm name
        common_params: Common parameters like time_limit, max_iterations, seed

    Returns:
        Tuple of (mode, strategy, params)
    """
    if common_params is None:
        common_params = {}

    # Exact algorithms
    if algorithm == "exhaustive":
        return ("exhaustive", None, {})

    # Greedy heuristic
    if algorithm == "greedy":
        return ("heuristic", None, {})

    # Randomized algorithms
    if algorithm in [
        "random_construction",
        "random_greedy_hybrid",
        "iterative_random_search",
        "monte_carlo",
        "las_vegas",
    ]:
        params = dict(common_params)
        if algorithm == "random_greedy_hybrid":
            params["num_starts"] = params.get("num_starts", 10)
            params["top_k"] = params.get("top_k", 3)
            params["randomness_factor"] = params.get("randomness_factor", 0.5)
        elif algorithm == "monte_carlo":
            params["num_samples"] = params.get("num_samples", 1000)
        elif algorithm == "las_vegas":
            params["max_attempts"] = params.get(
                "max_attempts", common_params.get("max_iterations", 10000)
            )
        return ("random", algorithm, params)

    # Reduction algorithms (don't accept time_limit, max_iterations, seed directly)
    if algorithm in ["mwc_redu", "max_clique_weight", "max_clique_dyn_weight"]:
        # These algorithms don't accept time_limit, max_iterations, or seed directly
        # Filter out unsupported parameters
        params = {}
        if algorithm == "mwc_redu":
            params["reduction_rules"] = common_params.get(
                "reduction_rules", ["domination", "isolation", "degree"]
            )
            params["solver_method"] = common_params.get("solver_method", "greedy")
            params["aggressive"] = common_params.get("aggressive", False)
        elif algorithm == "max_clique_weight":
            params["variant"] = common_params.get("variant", "static")
            params["color_ordering"] = common_params.get(
                "color_ordering", "weight_desc"
            )
            params["use_reduction"] = common_params.get("use_reduction", False)
        elif algorithm == "max_clique_dyn_weight":
            params["color_ordering"] = common_params.get(
                "color_ordering", "weight_desc"
            )
            params["use_reduction"] = common_params.get("use_reduction", False)
        return ("reduction", algorithm, params)

    # Exact Branch & Bound algorithms
    if algorithm in ["wlmc", "tsm_mwc"]:
        params = dict(common_params)
        return ("random", algorithm, params)

    # Heuristic algorithms
    if algorithm in ["fast_wclq", "scc_walk", "mwc_peel"]:
        params = dict(common_params)
        return ("random", algorithm, params)

    # Default fallback
    return ("heuristic", None, {})


def convert_result_to_dict(result: BenchmarkResult, algorithm: str) -> dict | None:
    """
    Convert a BenchmarkResult to simple dict format for visualization.

    Args:
        result: BenchmarkResult object
        algorithm: Name of the algorithm used

    Returns:
        Dict with algorithm, n_vertices, time_seconds, operations, weight, density
    """
    item = asdict(result)

    n_vertices = item.get("n_vertices", 0)
    density = item.get("edge_density_percent", 0)

    # Determine which fields to use based on algorithm type
    if algorithm == "exhaustive":
        time_s = item.get("exact_time_seconds")
        operations = item.get("exact_operations")
        weight = item.get("exact_weight")
    elif algorithm == "greedy":
        time_s = item.get("greedy_time_seconds")
        operations = item.get("greedy_operations")
        weight = item.get("greedy_weight")
    else:
        time_s = item.get("random_time_seconds")
        operations = item.get("random_operations")
        weight = item.get("random_weight")

    if time_s is None:
        return None

    return {
        "algorithm": algorithm,
        "n_vertices": n_vertices,
        "time_seconds": time_s,
        "operations": operations or 0,
        "weight": weight or 0,
        "density": density,
    }


def benchmark_directory(
    graphs_dir: Path,
    output_dir: Path = Path("experiments/results_custom"),
    algorithms: list[str] | None = None,
    recursive: bool = False,
    max_graphs: int | None = None,
    min_vertices: int | None = None,
    max_vertices: int | None = None,
    time_limit: float = 60.0,
    max_iterations: int = 5000,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, list[BenchmarkResult]]:
    """
    Benchmark all graphs in a directory with specified algorithms.

    Supports multiple graph formats:
    - GraphML (.graphml)
    - Sedgewick & Wayne TXT format (.txt with edge list)
    - Adjacency matrix format (.txt with NxN matrix)
    - DIMACS format (.clq, .dimacs)

    Args:
        graphs_dir: Directory containing graph files
        output_dir: Directory to save results
        algorithms: List of algorithm names to run (default: ["greedy"])
        recursive: Search subdirectories
        max_graphs: Maximum number of graphs to process
        min_vertices: Minimum number of vertices (filter graphs)
        max_vertices: Maximum number of vertices (filter graphs)
        time_limit: Time limit per graph for each algorithm
        max_iterations: Max iterations for randomized algorithms
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Dictionary mapping algorithm name to list of BenchmarkResult
    """
    from src.graph_loader import BenchmarkGraphLoader

    if algorithms is None:
        algorithms = ["greedy"]

    # Validate algorithms
    invalid_algs = [a for a in algorithms if a not in ALL_ALGORITHMS]
    if invalid_algs:
        raise ValueError(
            f"Unknown algorithms: {invalid_algs}. Available: {ALL_ALGORITHMS}"
        )

    # Load graph files
    loader = BenchmarkGraphLoader()
    graph_files = loader.list_graph_files(graphs_dir, pattern="*", recursive=recursive)

    if not graph_files:
        raise FileNotFoundError(
            f"No graph files found in '{graphs_dir}'. "
            "Supported formats: .graphml, .txt (SW/adjacency matrix), .clq, .dimacs"
        )

    # Filter by vertex count if specified
    if min_vertices is not None or max_vertices is not None:
        filtered_files = []
        for gf in graph_files:
            try:
                G = loader.load_graph(gf)
                n = G.number_of_nodes()
                if min_vertices is not None and n < min_vertices:
                    continue
                if max_vertices is not None and n > max_vertices:
                    continue
                filtered_files.append(gf)
            except Exception:
                # Include files we can't load (let them fail later with error message)
                filtered_files.append(gf)

        if verbose and len(filtered_files) != len(graph_files):
            v_range = f"{min_vertices or '*'}..{max_vertices or '*'}"
            print(
                f"Filtered to {len(filtered_files)} of {len(graph_files)} graphs ({v_range} vertices)"
            )
        graph_files = filtered_files

    if not graph_files:
        raise FileNotFoundError(
            f"No graph files found matching vertex range. "
            f"Range: {min_vertices or '*'}..{max_vertices or '*'}"
        )

    # Limit graphs if specified
    if max_graphs is not None and len(graph_files) > max_graphs:
        if verbose:
            print(f"Limiting to first {max_graphs} of {len(graph_files)} graphs")
        graph_files = graph_files[:max_graphs]

    if verbose:
        print(f"Found {len(graph_files)} graph files in {graphs_dir}")
        print(f"Running {len(algorithms)} algorithm(s): {', '.join(algorithms)}")

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common parameters
    common_params = {
        "time_limit": time_limit,
        "max_iterations": max_iterations,
    }
    if seed is not None:
        common_params["seed"] = seed

    runner = BenchmarkRunner(verbose=verbose)
    all_results: dict[str, list[BenchmarkResult]] = {}

    for alg in algorithms:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running {alg.upper()}")
            print(f"{'=' * 60}")

        alg_output_dir = output_dir / alg
        alg_output_dir.mkdir(parents=True, exist_ok=True)

        # Get algorithm configuration
        mode, strategy, params = get_algorithm_config(alg, common_params)

        try:
            results = runner.benchmark_series(
                graph_files,
                output_dir=alg_output_dir,
                mode=mode,
                random_strategy=strategy if mode == "random" else None,
                random_params=params if mode == "random" else None,
                reduction_strategy=strategy if mode == "reduction" else None,
                reduction_params=params if mode == "reduction" else None,
            )

            all_results[alg] = results

            if verbose:
                if results:
                    print(f"✓ {alg}: Processed {len(results)} graphs")
                else:
                    print(f"⚠ {alg}: No results generated")

        except Exception as e:
            if verbose:
                print(f"✗ {alg}: Error - {e}")
            all_results[alg] = []

    # Print summary
    if verbose:
        print(f"\n{'=' * 60}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 60}")

        for alg, results in all_results.items():
            if results:
                print(f"  {alg}: {len(results)} graphs processed")
            else:
                print(f"  {alg}: No results")

        print(f"\nResults saved to: {output_dir}")

    return all_results


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
