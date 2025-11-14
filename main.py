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
from src.algorithms import MaxWeightCliqueSolver
from src.benchmark import BenchmarkRunner
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
    densities: List[float] = typer.Option([12.5, 25.0, 50.0, 75.0], help="Edge density percentages"),
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
    graph: str = typer.Argument(..., help="Path to graph file (.graphml)"),
    visualize: bool = typer.Option(False, help="Visualize the solution"),
    no_show: bool = typer.Option(False, help="Don't display plot (only save to file)"),
    mode: str = typer.Option("both", help="Mode to run the algorithm (both, exhaustive, heuristic)"),
) -> None:
    """Solve maximum weight clique for a single graph."""
    graph_path = Path(graph)
    
    if not graph_path.exists():
        typer.echo(f"Error: Graph file '{graph_path}' not found.", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Loading graph from {graph_path}...")
    G = nx.read_graphml(graph_path)
    G = nx.convert_node_labels_to_integers(G)
    
    # Convert weight to float
    for node in G.nodes():
        G.nodes[node]["weight"] = float(G.nodes[node]["weight"])
    
    typer.echo(f"Graph: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
    
    solver = MaxWeightCliqueSolver(G)
    exact_result = None
    greedy_result = None

    if mode == "exhaustive" or mode == "both":
    # Run exhaustive search
        typer.echo("\n" + "="*60)
        typer.echo("EXHAUSTIVE SEARCH")
        typer.echo("="*60)
        exact_result = solver.exhaustive_search()
        typer.echo(f"Clique: {sorted(exact_result.clique)}")
        typer.echo(f"Clique size: {len(exact_result.clique)}")
        typer.echo(f"Total weight: {exact_result.total_weight:.2f}")
        typer.echo(f"Basic operations: {exact_result.basic_operations:,}")
        typer.echo(f"Configurations tested: {exact_result.configurations_tested:,}")
        
    if mode == "heuristic" or mode == "both":
    # Run greedy heuristic
        typer.echo("\n" + "="*60)
        typer.echo("GREEDY HEURISTIC")
        typer.echo("="*60)
        greedy_result = solver.greedy_heuristic()
        typer.echo(f"Clique: {sorted(greedy_result.clique)}")
        typer.echo(f"Clique size: {len(greedy_result.clique)}")
        typer.echo(f"Total weight: {greedy_result.total_weight:.2f}")
        typer.echo(f"Basic operations: {greedy_result.basic_operations:,}")
        typer.echo(f"Configurations tested: {greedy_result.configurations_tested:,}")
        
    # Comparison
    if exact_result and exact_result.total_weight > 0:
        precision = (greedy_result.total_weight / exact_result.total_weight * 100.0)
    else:
        precision = 100.0
    
    typer.echo("\n" + "="*60)
    typer.echo("COMPARISON")
    typer.echo("="*60)
    typer.echo(f"Greedy precision: {precision:.2f}%")
    typer.echo(f"Operations ratio: {greedy_result.basic_operations / exact_result.basic_operations:.2%}" if exact_result and exact_result.basic_operations > 0 else "0.00%")
    
    # Visualize if requested
    if visualize:
        output_path = Path("experiments/plots") / f"{graph_path.stem}_solution.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        GraphVisualizer.visualize_graph_with_clique(
            G, exact_result.clique if exact_result else greedy_result.clique,
            title=f"Maximum Weight Clique - {graph_path.stem}",
            output_path=output_path,
            show=not no_show,
        )


def _parse_range(range_str: str) -> tuple[int, int]:
    """Parse a range string like '4..20' into (min, max) tuple."""
    parts = range_str.split("..")
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}. Expected format: 'min..max'")
    return int(parts[0]), int(parts[1])


def _filter_graphs_by_vertices(graph_files: list[Path], min_v: int, max_v: int) -> list[Path]:
    """Filter graph files by vertex count range."""
    filtered = []
    for gf in graph_files:
        # Extract n from filename like "graph_n10_d50.graphml"
        name = gf.stem
        if "_n" in name:
            n_str = name.split("_n")[1].split("_")[0]
            n = int(n_str)
            if min_v <= n <= max_v:
                filtered.append(gf)
    return sorted(filtered)


@app.command("benchmark")
def cmd_benchmark(
    graphs_dir: str = typer.Option("experiments/graphs", help="Directory containing graphs"),
    output_dir: str = typer.Option("experiments/results", help="Output directory"),
    exhaustive: str = typer.Option("all", help="Vertex range for exhaustive search (e.g., '4..15' or 'all' for all graphs)"),
    heuristic: str = typer.Option("all", help="Vertex range for greedy heuristic (e.g., '4..100' or 'all' for all graphs)"),
    plot: bool = typer.Option(False, help="Generate plots after benchmarking"),
    verbose: bool = typer.Option(True, help="Print detailed progress"),
) -> None:
    """Run benchmarks on generated graphs."""
    graphs_dir_path = Path(graphs_dir)
    
    if not graphs_dir_path.exists():
        typer.echo(f"Error: Graphs directory '{graphs_dir_path}' not found.", err=True)
        typer.echo("Run 'python main.py generate' first to create graphs.", err=True)
        raise typer.Exit(1)
    
    all_graph_files = sorted(graphs_dir_path.glob("*.graphml"))
    
    if not all_graph_files:
        typer.echo(f"No graph files found in {graphs_dir_path}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Found {len(all_graph_files)} graph files")
    
    # Determine available vertex range from files
    vertex_counts = set()
    for gf in all_graph_files:
        name = gf.stem
        if "_n" in name:
            n_str = name.split("_n")[1].split("_")[0]
            vertex_counts.add(int(n_str))
    
    if not vertex_counts:
        typer.echo("Error: Could not determine vertex counts from graph filenames.", err=True)
        raise typer.Exit(1)
    
    min_vertices = min(vertex_counts)
    max_vertices = max(vertex_counts)
    default_range = f"{min_vertices}..{max_vertices}"
    
    # Parse vertex ranges
    exhaustive_range = exhaustive if exhaustive != "all" else default_range
    heuristic_range = heuristic if heuristic != "all" else default_range
    
    runner = BenchmarkRunner(verbose=verbose)
    all_results = []
    
    # Determine mode based on ranges
    if exhaustive_range == heuristic_range:
        # Same range for both - run together
        ex_min, ex_max = _parse_range(exhaustive_range)
        graph_files = _filter_graphs_by_vertices(all_graph_files, ex_min, ex_max)
        typer.echo(f"Running both algorithms on graphs with {ex_min}..{ex_max} vertices ({len(graph_files)} graphs)")
        
        results = runner.benchmark_series(
            graph_files,
            output_dir=Path(output_dir),
            mode="both",
        )
        all_results.extend(results)
    else:
        # Different ranges - run separately
        ex_min, ex_max = _parse_range(exhaustive_range)
        he_min, he_max = _parse_range(heuristic_range)
        
        ex_files = _filter_graphs_by_vertices(all_graph_files, ex_min, ex_max)
        he_files = _filter_graphs_by_vertices(all_graph_files, he_min, he_max)
        
        typer.echo(f"Exhaustive search: {ex_min}..{ex_max} vertices ({len(ex_files)} graphs)")
        typer.echo(f"Greedy heuristic: {he_min}..{he_max} vertices ({len(he_files)} graphs)")
        
        # Run exhaustive search
        if ex_files:
            ex_results = runner.benchmark_series(
                ex_files,
                output_dir=Path(output_dir) / "exhaustive",
                mode="exhaustive",
            )
            all_results.extend(ex_results)
        
        # Run greedy heuristic
        if he_files:
            he_results = runner.benchmark_series(
                he_files,
                output_dir=Path(output_dir) / "heuristic",
                mode="heuristic",
            )
            all_results.extend(he_results)
    
    # Print summary
    if all_results:
        runner.print_summary(all_results)
    
    # Generate plots if requested
    if plot and all_results:
        ResultsVisualizer.plot_all_metrics(
            all_results,
            output_dir=Path(output_dir).parent / "plots",
        )


@app.command("visualize")
def cmd_visualize(
    results: str = typer.Argument("experiments/results/benchmark_results.json", help="Path to results JSON file"),
    output_dir: str = typer.Option("experiments/plots", help="Output directory"),
) -> None:
    """Visualize benchmark results."""
    results_file = Path(results)
    
    if not results_file.exists():
        typer.echo(f"Error: Results file '{results_file}' not found.", err=True)
        typer.echo("Run 'python main.py benchmark' first to generate results.", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Loading results from {results_file}...")
    
    with open(results_file) as f:
        results_data = json.load(f)
    
    # Import BenchmarkResult to reconstruct objects
    from src.benchmark import BenchmarkResult
    results_objects = [BenchmarkResult(**r) for r in results_data]
    
    typer.echo(f"Loaded {len(results_objects)} results")
    
    # Generate plots
    output_dir_path = Path(output_dir)
    ResultsVisualizer.plot_all_metrics(results_objects, output_dir_path)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
