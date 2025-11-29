"""
Maximum Weight Clique Solver - Main Entry Point.

Command-line interface for graph generation, algorithm execution, and benchmarking.
"""

import json
from pathlib import Path
from typing import List

import networkx as nx
import typer

from src.graph_generator import GraphGenerator
from src.graph_loader import BenchmarkGraphLoader
from src.algorithms import (
    MaxWeightCliqueSolver,
    RandomizedAlgorithmResult,
    StoppingReason,
)
from src.visualizer import GraphVisualizer, ResultsVisualizer

app = typer.Typer(
    name="projeto1",
    help="Maximum Weight Clique Solver",
    no_args_is_help=True,
)


@app.command("generate")
def cmd_generate_graphs(
    seed: int = typer.Option(112974, help="Random seed"),
    min_vertices: int = typer.Option(4, help="Minimum number of vertices"),
    max_vertices: int = typer.Option(12, help="Maximum number of vertices"),
    densities: List[float] = typer.Option(
        [12.5, 25.0, 50.0, 75.0], help="Edge density percentages"
    ),
    output_dir: str = typer.Option("experiments/graphs", help="Output directory"),
) -> None:
    """Generate random graphs for experiments."""
    typer.echo(f"Generating graphs with seed={seed}")
    typer.echo(f"Vertices: {min_vertices} to {max_vertices}")
    typer.echo(f"Densities: {densities}%")

    generator = GraphGenerator(seed=seed)
    files = generator.generate_graph_series(
        min_vertices=min_vertices,
        max_vertices=max_vertices,
        densities=densities,
        output_dir=Path(output_dir),
    )

    typer.echo(f"\n✓ Generated {len(files)} graphs successfully!")


@app.command("solve")
def cmd_solve(
    graph: str = typer.Argument(
        ..., help="Path to graph file (.graphml, .clq, .dimacs)"
    ),
    visualize: bool = typer.Option(False, help="Visualize the solution"),
    no_show: bool = typer.Option(False, help="Don't display plot (only save to file)"),
    mode: str = typer.Option(
        "both",
        help="Mode to run the algorithm (both, exhaustive, heuristic, random, all)",
    ),
    random_strategy: str = typer.Option(
        "random_greedy_hybrid",
        help="Randomized algorithm strategy (random_construction, random_greedy_hybrid, iterative_random_search, monte_carlo, las_vegas, mwc_redu, max_clique_weight, max_clique_dyn_weight, wlmc, fast_wclq, scc_walk, mwc_peel, tsm_mwc)",
    ),
    max_iterations: int = typer.Option(
        None, help="Maximum iterations for randomized algorithms"
    ),
    time_limit: float = typer.Option(
        None, help="Time limit in seconds for randomized algorithms"
    ),
    num_starts: int = typer.Option(
        10, help="Number of starts for random_greedy_hybrid"
    ),
    top_k: int = typer.Option(3, help="Top-k candidates for random_greedy_hybrid"),
    randomness_factor: float = typer.Option(
        0.5, help="Randomness factor for random_greedy_hybrid (0-1)"
    ),
    seed: int = typer.Option(None, help="Random seed for reproducibility"),
    # Monte Carlo parameters
    num_samples: int = typer.Option(1000, help="Number of samples for monte_carlo"),
    sample_size_strategy: str = typer.Option(
        "proportional",
        help="Sample size strategy for monte_carlo (fixed, proportional, adaptive)",
    ),
    probability_threshold: float = typer.Option(
        0.1, help="Probability threshold for monte_carlo"
    ),
    # Las Vegas parameters
    max_attempts: int = typer.Option(10000, help="Maximum attempts for las_vegas"),
    construction_strategy: str = typer.Option(
        "iterative_construction",
        help="Construction strategy for las_vegas (random_walk, iterative_construction)",
    ),
    # MWCRedu parameters
    reduction_rules: str = typer.Option(
        "domination,isolation,degree",
        help="Comma-separated reduction rules for mwc_redu",
    ),
    solver_method: str = typer.Option(
        "greedy",
        help="Solver method for mwc_redu (exhaustive, greedy, monte_carlo, las_vegas)",
    ),
    aggressive: bool = typer.Option(
        False, help="Use aggressive reductions for mwc_redu"
    ),
    # MaxCliqueWeight parameters
    variant: str = typer.Option(
        "static", help="Variant for max_clique_weight (static, dynamic)"
    ),
    color_ordering: str = typer.Option(
        "weight_desc",
        help="Color ordering for max_clique_weight (weight_desc, degree_desc)",
    ),
    use_reduction: bool = typer.Option(
        False, help="Use graph reduction for max_clique_weight"
    ),
    # FastWClq parameters
    bms_k: int = typer.Option(5, help="BMS selection k for fast_wclq"),
    # SCCWalk parameters
    max_unimprove_steps: int = typer.Option(
        100, help="Steps without improvement before walk perturbation"
    ),
    walk_perturbation_strength: float = typer.Option(
        0.3, help="Fraction of clique to perturb in walk"
    ),
    # MWCPeel parameters
    peel_fraction: float = typer.Option(
        0.1, help="Fraction of vertices to peel each iteration"
    ),
) -> None:
    """Solve maximum weight clique for a single graph."""
    graph_path = Path(graph)

    if not graph_path.exists():
        typer.echo(f"Error: Graph file '{graph_path}' not found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading graph from {graph_path}...")

    # Use graph loader to support multiple formats
    loader = BenchmarkGraphLoader()
    try:
        G = loader.load_graph(graph_path)
    except Exception as e:
        typer.echo(f"Error loading graph: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Graph: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")

    solver = MaxWeightCliqueSolver(G)
    exact_result = None
    greedy_result = None
    random_result = None

    if mode in ["exhaustive", "both", "all"]:
        # Run exhaustive search
        typer.echo("\n" + "=" * 60)
        typer.echo("EXHAUSTIVE SEARCH")
        typer.echo("=" * 60)
        exact_result = solver.exhaustive_search()
        typer.echo(f"Clique: {sorted(exact_result.clique)}")
        typer.echo(f"Clique size: {len(exact_result.clique)}")
        typer.echo(f"Total weight: {exact_result.total_weight:.2f}")
        typer.echo(f"Basic operations: {exact_result.basic_operations:,}")
        typer.echo(f"Configurations tested: {exact_result.configurations_tested:,}")

    if mode in ["heuristic", "both", "all"]:
        # Run greedy heuristic
        typer.echo("\n" + "=" * 60)
        typer.echo("GREEDY HEURISTIC")
        typer.echo("=" * 60)
        greedy_result = solver.greedy_heuristic()
        typer.echo(f"Clique: {sorted(greedy_result.clique)}")
        typer.echo(f"Clique size: {len(greedy_result.clique)}")
        typer.echo(f"Total weight: {greedy_result.total_weight:.2f}")
        typer.echo(f"Basic operations: {greedy_result.basic_operations:,}")
        typer.echo(f"Configurations tested: {greedy_result.configurations_tested:,}")

    if mode in ["random", "all"]:
        # Run randomized algorithm
        typer.echo("\n" + "=" * 60)
        typer.echo(f"RANDOMIZED ALGORITHM: {random_strategy.upper()}")
        typer.echo("=" * 60)

        # Prepare parameters
        random_params: dict = {}
        if seed is not None:
            random_params["seed"] = seed

        if random_strategy == "random_construction":
            if max_iterations is not None:
                random_params["max_iterations"] = max_iterations
            if time_limit is not None:
                random_params["time_limit"] = time_limit
            random_result = solver.random_construction(**random_params)
        elif random_strategy == "random_greedy_hybrid":
            random_params["num_starts"] = num_starts
            random_params["top_k"] = top_k
            random_params["randomness_factor"] = randomness_factor
            random_result = solver.random_greedy_hybrid(**random_params)
        elif random_strategy == "iterative_random_search":
            if max_iterations is not None:
                random_params["max_iterations"] = max_iterations
            if time_limit is not None:
                random_params["time_limit"] = time_limit
            random_result = solver.iterative_random_search(**random_params)
        elif random_strategy == "monte_carlo":
            random_params["num_samples"] = (
                num_samples if max_iterations is None else max_iterations
            )
            random_params["sample_size_strategy"] = sample_size_strategy
            random_params["probability_threshold"] = probability_threshold
            if time_limit is not None:
                random_params["time_limit"] = time_limit
            random_result = solver.monte_carlo(**random_params)
        elif random_strategy == "las_vegas":
            random_params["max_attempts"] = (
                max_attempts if max_iterations is None else max_iterations
            )
            random_params["construction_strategy"] = construction_strategy
            if time_limit is not None:
                random_params["time_limit"] = time_limit
            random_result = solver.las_vegas(**random_params)
        elif random_strategy == "mwc_redu":
            rules_list = [r.strip() for r in reduction_rules.split(",")]
            random_params["reduction_rules"] = rules_list
            random_params["solver_method"] = solver_method
            random_params["aggressive"] = aggressive
            # Add solver-specific parameters
            solver_params: dict = {}
            if solver_method == "monte_carlo":
                solver_params["num_samples"] = num_samples
            if seed is not None:
                solver_params["seed"] = seed
            random_params["solver_params"] = solver_params
            result = solver.mwc_redu(**random_params)
            # mwc_redu can return either AlgorithmResult or RandomizedAlgorithmResult
            if isinstance(result, RandomizedAlgorithmResult):
                random_result = result
            else:
                # Convert to RandomizedAlgorithmResult for consistency
                random_result = RandomizedAlgorithmResult(
                    clique=result.clique,
                    total_weight=result.total_weight,
                    basic_operations=result.basic_operations,
                    configurations_tested=result.configurations_tested,
                    unique_configurations_tested=result.configurations_tested,
                    duplicate_count=0,
                    stopping_reason=StoppingReason.EXHAUSTED,
                )
        elif random_strategy == "max_clique_weight":
            random_params["variant"] = variant
            random_params["color_ordering"] = color_ordering
            random_params["use_reduction"] = use_reduction
            if reduction_rules:
                rules_list = [r.strip() for r in reduction_rules.split(",")]
                random_params["reduction_rules"] = rules_list
            result = solver.max_clique_weight(**random_params)
            # Convert to RandomizedAlgorithmResult for consistency
            random_result = RandomizedAlgorithmResult(
                clique=result.clique,
                total_weight=result.total_weight,
                basic_operations=result.basic_operations,
                configurations_tested=result.configurations_tested,
                unique_configurations_tested=result.configurations_tested,
                duplicate_count=0,
                stopping_reason=StoppingReason.EXHAUSTED,
            )
        elif random_strategy == "max_clique_dyn_weight":
            random_params["color_ordering"] = color_ordering
            random_params["use_reduction"] = use_reduction
            if reduction_rules:
                rules_list = [r.strip() for r in reduction_rules.split(",")]
                random_params["reduction_rules"] = rules_list
            result = solver.max_clique_dyn_weight(**random_params)
            # Convert to RandomizedAlgorithmResult for consistency
            random_result = RandomizedAlgorithmResult(
                clique=result.clique,
                total_weight=result.total_weight,
                basic_operations=result.basic_operations,
                configurations_tested=result.configurations_tested,
                unique_configurations_tested=result.configurations_tested,
                duplicate_count=0,
                stopping_reason=StoppingReason.EXHAUSTED,
            )
        elif random_strategy == "wlmc":
            # WLMC - Weighted Large Maximum Clique (exact BnB) - doesn't use seed
            wlmc_params: dict = {}
            if time_limit is not None:
                wlmc_params["time_limit"] = time_limit
            wlmc_params["use_preprocessing"] = True
            result = solver.wlmc(**wlmc_params)
            # Convert to RandomizedAlgorithmResult for consistency
            random_result = RandomizedAlgorithmResult(
                clique=result.clique,
                total_weight=result.total_weight,
                basic_operations=result.basic_operations,
                configurations_tested=result.configurations_tested,
                unique_configurations_tested=result.configurations_tested,
                duplicate_count=0,
                stopping_reason=StoppingReason.EXHAUSTED,
            )
        elif random_strategy == "fast_wclq":
            # FastWClq - Semi-exact heuristic with graph reduction
            if max_iterations is not None:
                random_params["max_iterations"] = max_iterations
            if time_limit is not None:
                random_params["time_limit"] = time_limit
            random_params["bms_k"] = bms_k
            random_result = solver.fast_wclq(**random_params)
        elif random_strategy == "scc_walk":
            # SCCWalk - Local search with Strong Configuration Checking
            if max_iterations is not None:
                random_params["max_iterations"] = max_iterations
            if time_limit is not None:
                random_params["time_limit"] = time_limit
            random_params["max_unimprove_steps"] = max_unimprove_steps
            random_params["walk_perturbation_strength"] = walk_perturbation_strength
            random_result = solver.scc_walk(**random_params)
        elif random_strategy == "mwc_peel":
            # MWCPeel - Hybrid reduction with peeling
            if max_iterations is not None:
                random_params["max_iterations"] = max_iterations
            if time_limit is not None:
                random_params["time_limit"] = time_limit
            random_params["peel_fraction"] = peel_fraction
            random_result = solver.mwc_peel(**random_params)
        elif random_strategy == "tsm_mwc":
            # TSM-MWC - Two-Stage MaxSAT (exact BnB) - doesn't use seed
            tsm_params: dict = {}
            if time_limit is not None:
                tsm_params["time_limit"] = time_limit
            result = solver.tsm_mwc(**tsm_params)
            # Convert to RandomizedAlgorithmResult for consistency
            random_result = RandomizedAlgorithmResult(
                clique=result.clique,
                total_weight=result.total_weight,
                basic_operations=result.basic_operations,
                configurations_tested=result.configurations_tested,
                unique_configurations_tested=result.configurations_tested,
                duplicate_count=0,
                stopping_reason=StoppingReason.EXHAUSTED,
            )
        else:
            typer.echo(f"Error: Unknown random strategy '{random_strategy}'", err=True)
            raise typer.Exit(1)

        typer.echo(f"Clique: {sorted(random_result.clique)}")
        typer.echo(f"Clique size: {len(random_result.clique)}")
        typer.echo(f"Total weight: {random_result.total_weight:.2f}")
        typer.echo(f"Basic operations: {random_result.basic_operations:,}")
        typer.echo(f"Configurations tested: {random_result.configurations_tested:,}")
        typer.echo(
            f"Unique configurations: {random_result.unique_configurations_tested:,}"
        )
        typer.echo(f"Duplicates: {random_result.duplicate_count:,}")
        if random_result.stopping_reason:
            typer.echo(f"Stopping reason: {random_result.stopping_reason.value}")

    # Comparison
    typer.echo("\n" + "=" * 60)
    typer.echo("COMPARISON")
    typer.echo("=" * 60)

    if exact_result and greedy_result and exact_result.total_weight > 0:
        precision = greedy_result.total_weight / exact_result.total_weight * 100.0
        typer.echo(f"Greedy precision: {precision:.2f}%")
        typer.echo(
            f"Operations ratio: {greedy_result.basic_operations / exact_result.basic_operations:.2%}"
            if exact_result.basic_operations > 0
            else "0.00%"
        )

    if random_result:
        if exact_result and exact_result.total_weight > 0:
            random_precision = (
                random_result.total_weight / exact_result.total_weight * 100.0
            )
            typer.echo(f"Random precision: {random_precision:.2f}%")
        elif greedy_result and greedy_result.total_weight > 0:
            random_precision = (
                random_result.total_weight / greedy_result.total_weight * 100.0
            )
            typer.echo(f"Random vs Greedy precision: {random_precision:.2f}%")

    # Visualize if requested
    if visualize:
        output_path = Path("experiments/plots") / f"{graph_path.stem}_solution.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Choose which clique to visualize
        clique_to_show = None
        if exact_result:
            clique_to_show = exact_result.clique
        elif random_result:
            clique_to_show = random_result.clique
        elif greedy_result:
            clique_to_show = greedy_result.clique

        if clique_to_show:
            GraphVisualizer.visualize_graph_with_clique(
                G,
                clique_to_show,
                title=f"Maximum Weight Clique - {graph_path.stem}",
                output_path=output_path,
                show=not no_show,
            )


@app.command("benchmark")
def cmd_benchmark(
    graphs_dir: str = typer.Option(
        "experiments/graphs", help="Directory containing graph files"
    ),
    output_dir: str = typer.Option("experiments/results", help="Output directory"),
    algorithm: str = typer.Option(
        "greedy",
        help="Algorithm to run (or 'all' for all algorithms)",
    ),
    recursive: bool = typer.Option(False, help="Search subdirectories"),
    max_graphs: int = typer.Option(None, help="Maximum number of graphs to process"),
    min_vertices: int = typer.Option(None, help="Minimum vertices to include"),
    max_vertices: int = typer.Option(None, help="Maximum vertices to include"),
    time_limit: float = typer.Option(60.0, help="Time limit (seconds) per graph"),
    max_iterations: int = typer.Option(
        5000, help="Max iterations for randomized algorithms"
    ),
    seed: int = typer.Option(None, help="Random seed"),
    verbose: bool = typer.Option(True, help="Print progress"),
    plot: bool = typer.Option(False, help="Generate plots after benchmarking"),
) -> None:
    """
    Run benchmarks on graphs from a directory.

    Supports multiple graph formats: GraphML, SW TXT, adjacency matrix, DIMACS.

    Examples:
        python main.py benchmark --algorithm greedy
        python main.py benchmark --graphs-dir experiments/sw --algorithm all
        python main.py benchmark --graphs-dir my_graphs/ --algorithm fast_wclq --recursive
        python main.py benchmark --algorithm exhaustive --min-vertices 4 --max-vertices 20
    """
    from src.benchmark import (
        benchmark_directory,
        ALL_ALGORITHMS,
        convert_result_to_dict,
    )

    # Determine algorithms to run
    if algorithm == "all":
        algorithms = ALL_ALGORITHMS
    else:
        algorithms = [algorithm]

    try:
        results = benchmark_directory(
            graphs_dir=Path(graphs_dir),
            output_dir=Path(output_dir),
            algorithms=algorithms,
            recursive=recursive,
            max_graphs=max_graphs,
            min_vertices=min_vertices,
            max_vertices=max_vertices,
            time_limit=time_limit,
            max_iterations=max_iterations,
            seed=seed,
            verbose=verbose,
        )
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Generate plots if requested
    if plot and results:
        typer.echo("\nGenerating visualizations...")
        try:
            all_viz_results = []
            for alg, alg_results in results.items():
                for r in alg_results:
                    converted = convert_result_to_dict(r, alg)
                    if converted:
                        all_viz_results.append(converted)

            if all_viz_results:
                output_path = Path(output_dir)
                ResultsVisualizer.generate_all_visualizations(
                    all_viz_results, output_path / "plots"
                )
                typer.echo(f"✓ Plots saved to: {output_path / 'plots'}")
        except Exception as e:
            typer.echo(f"Warning: Could not generate plots: {e}", err=True)


@app.command("load-benchmarks")
def cmd_load_benchmarks(
    input_dir: str = typer.Argument(..., help="Directory containing benchmark graphs"),
    output_dir: str = typer.Option(
        "experiments/benchmarks", help="Output directory for converted graphs"
    ),
    convert: bool = typer.Option(True, help="Convert graphs to GraphML format"),
    recursive: bool = typer.Option(False, help="Search directories recursively"),
) -> None:
    """Load and optionally convert benchmark graphs from web repositories."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        typer.echo(f"Error: Input directory '{input_path}' not found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading graphs from {input_path}...")

    loader = BenchmarkGraphLoader()
    graph_data = loader.load_directory(input_path, pattern="*", recursive=recursive)

    typer.echo(f"Found {len(graph_data)} graph files")

    if convert:
        output_path.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Converting to GraphML format in {output_path}...")

        converted_count = 0
        for input_file, graph in graph_data:
            output_file = output_path / f"{input_file.stem}.graphml"
            try:
                nx.write_graphml(graph, output_file)
                converted_count += 1
                if converted_count % 10 == 0:
                    typer.echo(
                        f"  Converted {converted_count}/{len(graph_data)} graphs..."
                    )
            except Exception as e:
                typer.echo(
                    f"  Warning: Could not convert {input_file.name}: {e}", err=True
                )

        typer.echo(f"\n✓ Converted {converted_count} graphs to {output_path}")
    else:
        typer.echo("\nGraphs loaded successfully (not converted)")
        for filepath, graph in graph_data[:10]:  # Show first 10
            typer.echo(
                f"  {filepath.name}: {graph.number_of_nodes()} vertices, "
                f"{graph.number_of_edges()} edges"
            )


@app.command("visualize")
def cmd_visualize(
    results: List[str] = typer.Argument(
        ["experiments/results/benchmark_results.json"],
        help="Path(s) to results JSON file(s) or directory containing benchmark_results.json files",
    ),
    output_dir: str = typer.Option("experiments/plots", help="Output directory"),
) -> None:
    """Visualize benchmark results from one or more JSON files."""
    from src.benchmark import BenchmarkResult

    results_files: list[Path] = []

    # Expand paths - handle both files and directories
    for result_path_str in results:
        result_path = Path(result_path_str)

        if not result_path.exists():
            typer.echo(
                f"Warning: Path '{result_path}' does not exist. Skipping.", err=True
            )
            continue

        if result_path.is_file():
            # It's a file - add it if it's a benchmark results file
            if result_path.name == "benchmark_results.json":
                results_files.append(result_path)
            else:
                typer.echo(
                    f"Warning: '{result_path}' is not a benchmark_results.json file. Skipping.",
                    err=True,
                )
        elif result_path.is_dir():
            # It's a directory - search for benchmark_results.json files recursively
            found_files = list(result_path.rglob("benchmark_results.json"))
            if found_files:
                results_files.extend(found_files)
            else:
                typer.echo(
                    f"Warning: No benchmark_results.json files found in '{result_path}'. Skipping.",
                    err=True,
                )

    if not results_files:
        typer.echo("Error: No valid benchmark results files found.", err=True)
        typer.echo(
            "Run 'python main.py benchmark' first to generate results.", err=True
        )
        raise typer.Exit(1)

    typer.echo(f"Loading results from {len(results_files)} file(s)...")

    # Load all results from all files
    all_results: list[BenchmarkResult] = []
    for results_file in results_files:
        try:
            with open(results_file) as f:
                results_data = json.load(f)

            # Handle both single result object and list of results
            if isinstance(results_data, dict):
                results_data = [results_data]
            elif not isinstance(results_data, list):
                typer.echo(
                    f"Warning: Unexpected data format in '{results_file}'. Skipping.",
                    err=True,
                )
                continue

            file_results = [BenchmarkResult(**r) for r in results_data]
            all_results.extend(file_results)
            typer.echo(f"  Loaded {len(file_results)} results from {results_file}")
        except Exception as e:
            typer.echo(
                f"Warning: Failed to load '{results_file}': {e}. Skipping.", err=True
            )

    if not all_results:
        typer.echo("Error: No valid results loaded from any file.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nTotal: {len(all_results)} results loaded")

    # Generate plots
    output_dir_path = Path(output_dir)
    ResultsVisualizer.plot_all_metrics(all_results, output_dir_path)


@app.command("visualize-individual")
def cmd_visualize_individual(
    results_dir: str = typer.Argument(
        "experiments/results_complete",
        help="Directory containing algorithm result files (JSON format)",
    ),
    output_dir: str = typer.Option("experiments/plots", help="Output directory"),
    algorithms: List[str] = typer.Option(
        None, help="Specific algorithms to visualize (default: all found in results)"
    ),
    comparison_pairs: List[str] = typer.Option(
        None, help="Pairwise comparisons in format 'alg1:alg2' (default: auto-generate)"
    ),
) -> None:
    """
    Generate individual algorithm charts and pairwise comparisons.

    This command creates:
    1. Individual charts for each algorithm (time and operations vs graph size)
    2. Pairwise comparison charts between algorithms
    3. Category summary charts

    Reads BenchmarkResult JSON files and converts them to the visualization format.
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        typer.echo(
            f"Error: Results directory '{results_path}' does not exist.", err=True
        )
        raise typer.Exit(1)

    # Load all algorithm results
    all_results: list[dict] = []

    # Look for JSON files in the directory structure
    json_files = list(results_path.rglob("**/benchmark_results.json"))

    if not json_files:
        # Try looking for any JSON files
        json_files = list(results_path.glob("**/*.json"))

    if not json_files:
        typer.echo(f"Error: No JSON files found in '{results_path}'", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading results from {len(json_files)} JSON file(s)...")

    for json_file in json_files:
        try:
            # Extract algorithm name from folder path
            # e.g., experiments/results_complete/fast_wclq/benchmark_results.json -> fast_wclq
            # or experiments/results_complete/baseline/exhaustive/benchmark_results.json -> exhaustive
            parent_folder = json_file.parent.name
            grandparent_folder = (
                json_file.parent.parent.name
                if json_file.parent.parent != results_path
                else None
            )

            # Determine algorithm name
            if parent_folder in ["exhaustive", "heuristic"]:
                algorithm_name = (
                    parent_folder if parent_folder == "exhaustive" else "greedy"
                )
            elif grandparent_folder == "baseline":
                algorithm_name = (
                    "exhaustive" if parent_folder == "exhaustive" else "greedy"
                )
            else:
                algorithm_name = parent_folder

            with open(json_file) as f:
                data = json.load(f)

            # Handle list of results
            if isinstance(data, list):
                for item in data:
                    converted = _convert_benchmark_result(item, algorithm_name)
                    if converted:
                        all_results.append(converted)
            elif isinstance(data, dict):
                converted = _convert_benchmark_result(data, algorithm_name)
                if converted:
                    all_results.append(converted)

            typer.echo(
                f"  Loaded {json_file.relative_to(results_path)} ({algorithm_name})"
            )
        except Exception as e:
            typer.echo(f"  Warning: Failed to load {json_file.name}: {e}", err=True)

    if not all_results:
        typer.echo("Error: No valid results loaded.", err=True)
        raise typer.Exit(1)

    # Filter by algorithms if specified
    if algorithms:
        all_results = [r for r in all_results if r.get("algorithm") in algorithms]
        if not all_results:
            typer.echo(
                f"Error: No results found for specified algorithms: {algorithms}",
                err=True,
            )
            raise typer.Exit(1)

    # Find unique algorithms
    found_algorithms = sorted(set(r.get("algorithm", "unknown") for r in all_results))
    typer.echo(
        f"\nFound {len(all_results)} results for {len(found_algorithms)} algorithms:"
    )
    for alg in found_algorithms:
        count = sum(1 for r in all_results if r.get("algorithm") == alg)
        typer.echo(f"  - {alg}: {count} data points")

    output_path = Path(output_dir)

    # Generate all visualizations
    typer.echo(f"\nGenerating visualizations to {output_path}...")

    # Parse comparison pairs if provided
    pairs = None
    if comparison_pairs:
        pairs = []
        for pair in comparison_pairs:
            if ":" in pair:
                alg1, alg2 = pair.split(":", 1)
                pairs.append((alg1, alg2))
            else:
                typer.echo(
                    f"  Warning: Invalid pair format '{pair}', use 'alg1:alg2'",
                    err=True,
                )

    ResultsVisualizer.generate_all_visualizations(all_results, output_path)

    typer.echo(f"\n✓ All visualizations saved to {output_path}")


def _convert_benchmark_result(item: dict, algorithm_name: str) -> dict | None:
    """
    Convert a BenchmarkResult dict to the simple visualization format.

    Args:
        item: Dictionary with BenchmarkResult fields
        algorithm_name: Name of the algorithm

    Returns:
        Simplified dict with algorithm, n_vertices, time_seconds, operations, weight, density
    """
    n_vertices = item.get("n_vertices", 0)
    density = item.get("edge_density_percent", 0)

    # Determine which fields to use based on algorithm type
    if algorithm_name == "exhaustive":
        time_s = item.get("exact_time_seconds")
        operations = item.get("exact_operations")
        weight = item.get("exact_weight")
    elif algorithm_name == "greedy":
        time_s = item.get("greedy_time_seconds")
        operations = item.get("greedy_operations")
        weight = item.get("greedy_weight")
    else:
        # Randomized/reduction/heuristic algorithms use random_* fields
        time_s = item.get("random_time_seconds")
        operations = item.get("random_operations")
        weight = item.get("random_weight")

    # Skip if no valid time data
    if time_s is None:
        return None

    return {
        "algorithm": algorithm_name,
        "n_vertices": n_vertices,
        "time_seconds": time_s,
        "operations": operations or 0,
        "weight": weight or 0,
        "density": density,
    }


@app.command("compare-quality")
def cmd_compare_quality(
    results_dir: str = typer.Argument(
        "experiments/results_complete/correctness",
        help="Directory containing correctness test results",
    ),
    output_dir: str = typer.Option(
        "experiments/plots/quality", help="Output directory for quality plots"
    ),
    algorithms: List[str] = typer.Option(
        None, help="Specific algorithms to compare (default: all except exhaustive)"
    ),
    show: bool = typer.Option(False, help="Display plots interactively"),
) -> None:
    """
    Compare solution quality of algorithms against exhaustive search.

    This command analyzes correctness test results and generates:
    1. Precision plots (algorithm weight / exhaustive weight) for each algorithm
    2. Summary table showing average precision per algorithm
    3. Quality comparison charts by graph size and density

    Uses the correctness test results where all algorithms run on the same graphs.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        typer.echo(
            f"Error: Results directory '{results_path}' does not exist.", err=True
        )
        raise typer.Exit(1)

    # Load exhaustive results first (ground truth)
    exhaustive_data = {}
    exhaustive_files = list(
        results_path.rglob("**/exhaustive/**/benchmark_results.json")
    )
    if not exhaustive_files:
        exhaustive_files = list(
            results_path.rglob("**/exhaustive/benchmark_results.json")
        )

    if not exhaustive_files:
        typer.echo(
            "Error: No exhaustive search results found. Run correctness tests first.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Loading exhaustive results from {len(exhaustive_files)} file(s)...")
    for json_file in exhaustive_files:
        with open(json_file) as f:
            data = json.load(f)
        for item in data:
            if item.get("exact_weight") is not None:
                key = (item["n_vertices"], item["edge_density_percent"])
                exhaustive_data[key] = {
                    "weight": item["exact_weight"],
                    "time": item.get("exact_time_seconds", 0),
                    "operations": item.get("exact_operations", 0),
                }

    typer.echo(f"  Found {len(exhaustive_data)} exhaustive results")

    # Load all other algorithm results
    algorithm_results = {}
    json_files = list(results_path.rglob("**/benchmark_results.json"))

    for json_file in json_files:
        # Extract algorithm name
        parent = json_file.parent.name
        if parent == "exhaustive":
            continue  # Skip exhaustive, we already have it

        algorithm_name = parent
        if algorithm_name not in algorithm_results:
            algorithm_results[algorithm_name] = []

        with open(json_file) as f:
            data = json.load(f)

        for item in data:
            # Determine weight field
            if item.get("greedy_weight") is not None:
                weight = item["greedy_weight"]
                time_s = item.get("greedy_time_seconds", 0)
            elif item.get("random_weight") is not None:
                weight = item["random_weight"]
                time_s = item.get("random_time_seconds", 0)
            elif item.get("exact_weight") is not None:
                weight = item["exact_weight"]
                time_s = item.get("exact_time_seconds", 0)
            else:
                continue

            key = (item["n_vertices"], item["edge_density_percent"])
            if key in exhaustive_data:
                exact_weight = exhaustive_data[key]["weight"]
                precision = (weight / exact_weight * 100) if exact_weight > 0 else 0
                algorithm_results[algorithm_name].append(
                    {
                        "n_vertices": item["n_vertices"],
                        "density": item["edge_density_percent"],
                        "weight": weight,
                        "exact_weight": exact_weight,
                        "precision": precision,
                        "time": time_s,
                    }
                )

    # Filter algorithms if specified
    if algorithms:
        algorithm_results = {
            k: v for k, v in algorithm_results.items() if k in algorithms
        }

    if not algorithm_results:
        typer.echo("Error: No algorithm results found to compare.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nLoaded results for {len(algorithm_results)} algorithms:")
    for alg, results in algorithm_results.items():
        avg_precision = (
            sum(r["precision"] for r in results) / len(results) if results else 0
        )
        typer.echo(
            f"  - {alg}: {len(results)} results, avg precision: {avg_precision:.2f}%"
        )

    # Generate plots
    typer.echo(f"\nGenerating quality comparison plots to {output_path}...")

    # 1. Precision summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    alg_names = sorted(algorithm_results.keys())
    avg_precisions = []
    min_precisions = []
    colors = []

    for alg in alg_names:
        results = algorithm_results[alg]
        precisions = [r["precision"] for r in results]
        avg_precisions.append(sum(precisions) / len(precisions) if precisions else 0)
        min_precisions.append(min(precisions) if precisions else 0)
        # Color based on average precision
        avg = avg_precisions[-1]
        if avg >= 99:
            colors.append("#2ECC71")  # Green - optimal
        elif avg >= 95:
            colors.append("#3498DB")  # Blue - excellent
        elif avg >= 90:
            colors.append("#F39C12")  # Orange - good
        else:
            colors.append("#E74C3C")  # Red - needs improvement

    x = np.arange(len(alg_names))
    bars = ax.bar(x, avg_precisions, color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(
        y=100, color="green", linestyle="--", linewidth=2, label="Optimal (100%)"
    )
    ax.axhline(y=95, color="blue", linestyle=":", linewidth=1, label="95% threshold")

    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Average Precision (%)", fontsize=12)
    ax.set_title(
        "Solution Quality: Average Precision vs Exhaustive Search", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(alg_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val, minv in zip(bars, avg_precisions, min_precisions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 5,
            f"(min: {minv:.1f}%)",
            ha="center",
            va="top",
            fontsize=7,
            color="gray",
        )

    plt.tight_layout()
    plt.savefig(output_path / "precision_summary.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    typer.echo(f"  ✓ Saved precision_summary.png")

    # 2. Precision vs graph size for each algorithm
    fig, ax = plt.subplots(figsize=(14, 8))

    for alg in alg_names:
        results = algorithm_results[alg]
        if not results:
            continue
        # Group by n_vertices and average
        by_size = {}
        for r in results:
            n = r["n_vertices"]
            if n not in by_size:
                by_size[n] = []
            by_size[n].append(r["precision"])

        sizes = sorted(by_size.keys())
        avg_precs = [sum(by_size[s]) / len(by_size[s]) for s in sizes]
        ax.plot(sizes, avg_precs, marker="o", label=alg, linewidth=2, markersize=4)

    ax.axhline(y=100, color="green", linestyle="--", linewidth=2, alpha=0.5)
    ax.set_xlabel("Number of Vertices", fontsize=12)
    ax.set_ylabel("Average Precision (%)", fontsize=12)
    ax.set_title("Solution Quality vs Graph Size", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path / "precision_vs_size.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    typer.echo(f"  ✓ Saved precision_vs_size.png")

    # 3. Precision by density
    densities = sorted(
        set(r["density"] for results in algorithm_results.values() for r in results)
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, density in enumerate(densities[:4]):
        ax = axes[idx]
        for alg in alg_names:
            results = [r for r in algorithm_results[alg] if r["density"] == density]
            if not results:
                continue
            sizes = [r["n_vertices"] for r in results]
            precs = [r["precision"] for r in results]
            ax.scatter(sizes, precs, label=alg, alpha=0.7, s=30)

        ax.axhline(y=100, color="green", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Vertices", fontsize=10)
        ax.set_ylabel("Precision (%)", fontsize=10)
        ax.set_title(f"Density: {density}%", fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower left")

    plt.suptitle("Solution Quality by Graph Density", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / "precision_by_density.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    typer.echo(f"  ✓ Saved precision_by_density.png")

    # 4. Individual algorithm vs exhaustive scatter plots
    for alg in alg_names:
        results = algorithm_results[alg]
        if not results:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))

        exact_weights = [r["exact_weight"] for r in results]
        alg_weights = [r["weight"] for r in results]

        # Color by density
        colors_scatter = [r["density"] for r in results]
        scatter = ax.scatter(
            exact_weights,
            alg_weights,
            c=colors_scatter,
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Perfect line
        max_w = max(max(exact_weights), max(alg_weights)) * 1.1
        ax.plot([0, max_w], [0, max_w], "g--", linewidth=2, label="Perfect (100%)")
        ax.plot([0, max_w], [0, max_w * 0.95], "b:", linewidth=1, label="95%")
        ax.plot([0, max_w], [0, max_w * 0.90], "r:", linewidth=1, label="90%")

        ax.set_xlabel("Exhaustive Weight (Optimal)", fontsize=12)
        ax.set_ylabel(f"{alg} Weight", fontsize=12)
        ax.set_title(f"Solution Quality: {alg} vs Exhaustive", fontsize=14)
        ax.legend(loc="lower right")
        ax.set_xlim(0, max_w)
        ax.set_ylim(0, max_w)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Edge Density (%)")

        # Add precision annotation
        avg_prec = sum(r["precision"] for r in results) / len(results)
        ax.text(
            0.05,
            0.95,
            f"Avg Precision: {avg_prec:.2f}%",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(
            output_path / f"quality_{alg}_vs_exhaustive.png",
            dpi=150,
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close()
        typer.echo(f"  ✓ Saved quality_{alg}_vs_exhaustive.png")

    # 5. Summary table
    summary_file = output_path / "quality_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SOLUTION QUALITY SUMMARY - Comparison vs Exhaustive Search\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"{'Algorithm':<30} {'Avg %':>10} {'Min %':>10} {'Max %':>10} {'Optimal':>10}\n"
        )
        f.write("-" * 80 + "\n")

        for alg in sorted(alg_names):
            results = algorithm_results[alg]
            if not results:
                continue
            precisions = [r["precision"] for r in results]
            avg_p = sum(precisions) / len(precisions)
            min_p = min(precisions)
            max_p = max(precisions)
            optimal = sum(1 for p in precisions if p >= 99.99)

            f.write(
                f"{alg:<30} {avg_p:>10.2f} {min_p:>10.2f} {max_p:>10.2f} {optimal:>10}/{len(precisions)}\n"
            )

        f.write("-" * 80 + "\n")
        f.write("\nOptimal = solutions within 0.01% of exhaustive search\n")

    typer.echo(f"  ✓ Saved quality_summary.txt")

    # Print summary to console
    typer.echo("\n" + "=" * 60)
    typer.echo("QUALITY SUMMARY")
    typer.echo("=" * 60)
    typer.echo(f"{'Algorithm':<25} {'Avg Precision':>15} {'Min':>10} {'Optimal':>12}")
    typer.echo("-" * 60)

    for alg in sorted(
        alg_names,
        key=lambda a: -sum(r["precision"] for r in algorithm_results[a])
        / len(algorithm_results[a])
        if algorithm_results[a]
        else 0,
    ):
        results = algorithm_results[alg]
        if not results:
            continue
        precisions = [r["precision"] for r in results]
        avg_p = sum(precisions) / len(precisions)
        min_p = min(precisions)
        optimal = sum(1 for p in precisions if p >= 99.99)
        typer.echo(
            f"{alg:<25} {avg_p:>14.2f}% {min_p:>9.2f}% {optimal:>5}/{len(precisions)}"
        )

    typer.echo("=" * 60)
    typer.echo(f"\n✓ All quality plots saved to {output_path}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
