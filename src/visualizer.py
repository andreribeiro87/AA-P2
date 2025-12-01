"""
Visualization module for graphs and benchmark results.

Creates visual representations of graphs, cliques, and performance metrics.
Generates individual algorithm charts and pairwise comparison charts.
"""

from collections import Counter
from pathlib import Path
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.benchmark import BenchmarkResult


# Algorithm display names and categories
ALGORITHM_INFO = {
    "exhaustive": {
        "name": "Exhaustive Search",
        "category": "exact",
        "color": "#E74C3C",
    },
    "greedy": {"name": "Greedy Heuristic", "category": "heuristic", "color": "#3498DB"},
    "random_construction": {
        "name": "Random Construction",
        "category": "randomized",
        "color": "#2ECC71",
    },
    "random_greedy_hybrid": {
        "name": "Random Greedy Hybrid",
        "category": "randomized",
        "color": "#9B59B6",
    },
    "iterative_random_search": {
        "name": "Iterative Random Search",
        "category": "randomized",
        "color": "#F39C12",
    },
    "monte_carlo": {
        "name": "Monte Carlo",
        "category": "randomized",
        "color": "#1ABC9C",
    },
    "las_vegas": {"name": "Las Vegas", "category": "randomized", "color": "#E91E63"},
    "mwc_redu": {"name": "MWC-Redu", "category": "reduction", "color": "#00BCD4"},
    "max_clique_weight": {
        "name": "Max-Clique-Weight",
        "category": "reduction",
        "color": "#FF5722",
    },
    "max_clique_dyn_weight": {
        "name": "Max-Clique-Dyn-Weight",
        "category": "reduction",
        "color": "#795548",
    },
    "wlmc": {"name": "WLMC", "category": "exact_bnb", "color": "#673AB7"},
    "tsm_mwc": {"name": "TSM-MWC", "category": "exact_bnb", "color": "#009688"},
    "fast_wclq": {"name": "FastWClq", "category": "heuristic", "color": "#FF9800"},
    "scc_walk": {"name": "SCC-Walk", "category": "heuristic", "color": "#607D8B"},
    "mwc_peel": {"name": "MWC-Peel", "category": "heuristic", "color": "#8BC34A"},
}


class GraphVisualizer:
    """Visualize graphs and maximum weight cliques."""

    @staticmethod
    def visualize_graph_with_clique(
        graph: nx.Graph,
        clique: set[int],
        title: str = "Graph with Maximum Weight Clique",
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Visualize a graph highlighting a clique.

        Args:
            graph: NetworkX graph to visualize
            clique: set of vertices in the clique
            title: Plot title
            output_path: Path to save figure (if None, don't save)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get positions from node attributes (x, y coordinates)
        pos = {
            node: (graph.nodes[node]["x"], graph.nodes[node]["y"])
            for node in graph.nodes()
        }

        # Separate nodes into clique and non-clique
        clique_nodes = list(clique)
        other_nodes = [n for n in graph.nodes() if n not in clique]

        # Draw edges
        # Clique internal edges (bold red)
        clique_edges = [(u, v) for u, v in graph.edges() if u in clique and v in clique]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=clique_edges,
            width=2.5,
            alpha=0.8,
            edge_color="red",
            ax=ax,
        )

        # Other edges (thin gray)
        other_edges = [(u, v) for u, v in graph.edges() if (u, v) not in clique_edges]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=other_edges,
            width=1.0,
            alpha=0.3,
            edge_color="gray",
            ax=ax,
        )

        # Draw nodes
        # Clique nodes (red)
        if clique_nodes:
            clique_weights = [graph.nodes[n]["weight"] for n in clique_nodes]
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=clique_nodes,
                node_color="red",
                node_size=[w * 10 for w in clique_weights],
                alpha=0.9,
                ax=ax,
            )

        # Other nodes (light blue)
        if other_nodes:
            other_weights = [graph.nodes[n]["weight"] for n in other_nodes]
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=other_nodes,
                node_color="lightblue",
                node_size=[w * 10 for w in other_weights],
                alpha=0.7,
                ax=ax,
            )

        # Draw labels
        labels = {
            node: f"{node}\n({graph.nodes[node]['weight']:.1f})"
            for node in graph.nodes()
        }
        nx.draw_networkx_labels(
            graph, pos, labels, font_size=8, font_weight="bold", ax=ax
        )

        # Add legend
        clique_weight = sum(graph.nodes[n]["weight"] for n in clique)
        legend_elements = [
            mpatches.Patch(color="red", label=f"Clique (weight={clique_weight:.2f})"),
            mpatches.Patch(color="lightblue", label="Other vertices"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Graph visualization saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()


class AlgorithmData:
    """Container for algorithm benchmark data."""

    def __init__(self, name: str, display_name: str, color: str, category: str):
        self.name = name
        self.display_name = display_name
        self.color = color
        self.category = category
        self.vertices: list[int] = []
        self.times: list[float] = []
        self.operations: list[int] = []
        self.weights: list[float] = []
        self.densities: list[float] = []

    def add_data(
        self, n_vertices: int, time_s: float, ops: int, weight: float, density: float
    ):
        """Add a data point."""
        self.vertices.append(n_vertices)
        self.times.append(time_s)
        self.operations.append(ops)
        self.weights.append(weight)
        self.densities.append(density)

    def has_data(self) -> bool:
        """Check if this algorithm has any data."""
        return len(self.vertices) > 0

    def get_sorted_data(self) -> tuple:
        """Get data sorted by number of vertices."""
        if not self.has_data():
            return [], [], [], [], []

        sorted_indices = sorted(
            range(len(self.vertices)), key=lambda i: self.vertices[i]
        )
        return (
            [self.vertices[i] for i in sorted_indices],
            [self.times[i] for i in sorted_indices],
            [self.operations[i] for i in sorted_indices],
            [self.weights[i] for i in sorted_indices],
            [self.densities[i] for i in sorted_indices],
        )


class ResultsVisualizer:
    """
    Visualize benchmark results with individual algorithm charts and pairwise comparisons.

    This class provides:
    1. Individual algorithm charts: Time and Operations vs Graph Size
    2. Pairwise comparison charts: Algorithm A vs Algorithm B
    3. Category summary charts: Group algorithms by type
    """

    @staticmethod
    def _extract_algorithm_data(results: list[dict]) -> dict[str, AlgorithmData]:
        """
        Extract algorithm data from results.

        Args:
            results: List of dicts with keys: algorithm, n_vertices, time_seconds, operations, weight, density

        Returns:
            Dictionary mapping algorithm name to AlgorithmData
        """
        algorithms: dict[str, AlgorithmData] = {}

        for r in results:
            alg_name = r.get("algorithm", "unknown")
            if alg_name not in algorithms:
                info = ALGORITHM_INFO.get(
                    alg_name,
                    {
                        "name": alg_name.replace("_", " ").title(),
                        "category": "other",
                        "color": "#888888",
                    },
                )
                algorithms[alg_name] = AlgorithmData(
                    name=alg_name,
                    display_name=info["name"],
                    color=info["color"],
                    category=info["category"],
                )

            algorithms[alg_name].add_data(
                n_vertices=r.get("n_vertices", 0),
                time_s=r.get("time_seconds", 0.0),
                ops=r.get("operations", 0),
                weight=r.get("weight", 0.0),
                density=r.get("density", 0.0),
            )

        return algorithms

    @staticmethod
    def plot_individual_algorithm_time(
        results: list[dict],
        algorithm: str,
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Create individual time chart for a single algorithm.

        Args:
            results: List of benchmark results
            algorithm: Algorithm name to plot
            output_path: Path to save figure
            show: Whether to display the plot
        """
        alg_results = [r for r in results if r.get("algorithm") == algorithm]

        if not alg_results:
            print(f"No data for algorithm: {algorithm}")
            return

        info = ALGORITHM_INFO.get(
            algorithm, {"name": algorithm.replace("_", " ").title(), "color": "#3498DB"}
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by density for multiple lines
        densities: dict[float, list[tuple]] = {}
        for r in alg_results:
            d = r.get("density", 0.0)
            if d not in densities:
                densities[d] = []
            densities[d].append((r.get("n_vertices", 0), r.get("time_seconds", 0)))

        # Sort densities and assign colors
        sorted_densities = sorted(densities.keys())
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_densities)))

        for density, color in zip(sorted_densities, colors):
            data = sorted(densities[density], key=lambda x: x[0])
            vertices, times = zip(*data) if data else ([], [])
            ax.plot(
                vertices,
                times,
                "o-",
                color=color,
                linewidth=2,
                markersize=8,
                label=f"Density {density:.0f}%",
            )

        ax.set_xlabel("Number of Vertices", fontsize=12)
        ax.set_ylabel("Execution Time (seconds)", fontsize=12)
        ax.set_title(
            f"{info['name']} - Execution Time vs Graph Size",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Use log scale if range is large
        times_all = [r.get("time_seconds", 0) for r in alg_results]
        if max(times_all) / (min(times_all) + 1e-10) > 100:
            ax.set_yscale("log")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ {info['name']} time chart saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_individual_algorithm_operations(
        results: list[dict],
        algorithm: str,
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Create individual operations chart for a single algorithm.

        Args:
            results: List of benchmark results
            algorithm: Algorithm name to plot
            output_path: Path to save figure
            show: Whether to display the plot
        """
        alg_results = [r for r in results if r.get("algorithm") == algorithm]

        if not alg_results:
            print(f"No data for algorithm: {algorithm}")
            return

        info = ALGORITHM_INFO.get(
            algorithm, {"name": algorithm.replace("_", " ").title(), "color": "#3498DB"}
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by density for multiple lines
        densities: dict[float, list[tuple]] = {}
        for r in alg_results:
            d = r.get("density", 0.0)
            if d not in densities:
                densities[d] = []
            densities[d].append((r.get("n_vertices", 0), r.get("operations", 0)))

        # Sort densities and assign colors
        sorted_densities = sorted(densities.keys())
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(sorted_densities)))

        for density, color in zip(sorted_densities, colors):
            data = sorted(densities[density], key=lambda x: x[0])
            vertices, ops = zip(*data) if data else ([], [])
            ax.plot(
                vertices,
                ops,
                "s-",
                color=color,
                linewidth=2,
                markersize=8,
                label=f"Density {density:.0f}%",
            )

        ax.set_xlabel("Number of Vertices", fontsize=12)
        ax.set_ylabel("Basic Operations", fontsize=12)
        ax.set_title(
            f"{info['name']} - Operations vs Graph Size", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Use log scale if range is large
        ops_all = [r.get("operations", 0) for r in alg_results]
        if ops_all and max(ops_all) / (min(ops_all) + 1) > 100:
            ax.set_yscale("log")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ {info['name']} operations chart saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_pairwise_time_comparison(
        results: list[dict],
        algorithm1: str,
        algorithm2: str,
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Create pairwise time comparison chart between two algorithms.

        Args:
            results: List of benchmark results
            algorithm1: First algorithm name
            algorithm2: Second algorithm name
            output_path: Path to save figure
            show: Whether to display the plot
        """
        alg1_results = {
            (r.get("n_vertices"), r.get("density")): r
            for r in results
            if r.get("algorithm") == algorithm1
        }
        alg2_results = {
            (r.get("n_vertices"), r.get("density")): r
            for r in results
            if r.get("algorithm") == algorithm2
        }

        # Find common graph instances
        common_keys = set(alg1_results.keys()) & set(alg2_results.keys())

        if not common_keys:
            print(f"No common data between {algorithm1} and {algorithm2}")
            return

        info1 = ALGORITHM_INFO.get(
            algorithm1,
            {"name": algorithm1.replace("_", " ").title(), "color": "#E74C3C"},
        )
        info2 = ALGORITHM_INFO.get(
            algorithm2,
            {"name": algorithm2.replace("_", " ").title(), "color": "#3498DB"},
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by density
        densities: dict[float, list[tuple]] = {}
        for key in common_keys:
            n, d = key
            if d not in densities:
                densities[d] = []
            densities[d].append(
                (
                    n,
                    alg1_results[key].get("time_seconds", 0),
                    alg2_results[key].get("time_seconds", 0),
                )
            )

        width = 0.35
        x_positions = []
        x_labels = []
        alg1_times = []
        alg2_times = []

        for density in sorted(densities.keys()):
            data = sorted(densities[density], key=lambda x: x[0])
            for n, t1, t2 in data:
                x_positions.append(len(x_labels))
                x_labels.append(f"n={n}\nd={density:.0f}%")
                alg1_times.append(t1)
                alg2_times.append(t2)

        x = np.arange(len(x_labels))

        bars1 = ax.bar(
            x - width / 2,
            alg1_times,
            width,
            label=info1["name"],
            color=info1["color"],
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            alg2_times,
            width,
            label=info2["name"],
            color=info2["color"],
            alpha=0.8,
        )

        ax.set_xlabel("Graph Instance", fontsize=12)
        ax.set_ylabel("Execution Time (seconds)", fontsize=12)
        ax.set_title(
            f"Time Comparison: {info1['name']} vs {info2['name']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Use log scale if needed
        all_times = alg1_times + alg2_times
        if all_times and max(all_times) / (min(all_times) + 1e-10) > 100:
            ax.set_yscale("log")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Pairwise time comparison saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_pairwise_operations_comparison(
        results: list[dict],
        algorithm1: str,
        algorithm2: str,
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Create pairwise operations comparison chart between two algorithms.

        Args:
            results: List of benchmark results
            algorithm1: First algorithm name
            algorithm2: Second algorithm name
            output_path: Path to save figure
            show: Whether to display the plot
        """
        alg1_results = {
            (r.get("n_vertices"), r.get("density")): r
            for r in results
            if r.get("algorithm") == algorithm1
        }
        alg2_results = {
            (r.get("n_vertices"), r.get("density")): r
            for r in results
            if r.get("algorithm") == algorithm2
        }

        # Find common graph instances
        common_keys = set(alg1_results.keys()) & set(alg2_results.keys())

        if not common_keys:
            print(f"No common data between {algorithm1} and {algorithm2}")
            return

        info1 = ALGORITHM_INFO.get(
            algorithm1,
            {"name": algorithm1.replace("_", " ").title(), "color": "#E74C3C"},
        )
        info2 = ALGORITHM_INFO.get(
            algorithm2,
            {"name": algorithm2.replace("_", " ").title(), "color": "#3498DB"},
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by density
        densities: dict[float, list[tuple]] = {}
        for key in common_keys:
            n, d = key
            if d not in densities:
                densities[d] = []
            densities[d].append(
                (
                    n,
                    alg1_results[key].get("operations", 0),
                    alg2_results[key].get("operations", 0),
                )
            )

        width = 0.35
        x_labels = []
        alg1_ops = []
        alg2_ops = []

        for density in sorted(densities.keys()):
            data = sorted(densities[density], key=lambda x: x[0])
            for n, o1, o2 in data:
                x_labels.append(f"n={n}\nd={density:.0f}%")
                alg1_ops.append(o1)
                alg2_ops.append(o2)

        x = np.arange(len(x_labels))

        bars1 = ax.bar(
            x - width / 2,
            alg1_ops,
            width,
            label=info1["name"],
            color=info1["color"],
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            alg2_ops,
            width,
            label=info2["name"],
            color=info2["color"],
            alpha=0.8,
        )

        ax.set_xlabel("Graph Instance", fontsize=12)
        ax.set_ylabel("Basic Operations", fontsize=12)
        ax.set_title(
            f"Operations Comparison: {info1['name']} vs {info2['name']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Use log scale if needed
        all_ops = alg1_ops + alg2_ops
        if all_ops and max(all_ops) / (min(all_ops) + 1) > 100:
            ax.set_yscale("log")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Pairwise operations comparison saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_pairwise_line_comparison(
        results: list[dict],
        algorithm1: str,
        algorithm2: str,
        metric: str = "time",
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Create pairwise line chart comparison between two algorithms.

        Args:
            results: List of benchmark results
            algorithm1: First algorithm name
            algorithm2: Second algorithm name
            metric: "time" or "operations"
            output_path: Path to save figure
            show: Whether to display the plot
        """
        alg1_results = [r for r in results if r.get("algorithm") == algorithm1]
        alg2_results = [r for r in results if r.get("algorithm") == algorithm2]

        if not alg1_results or not alg2_results:
            print(f"Missing data for comparison")
            return

        info1 = ALGORITHM_INFO.get(
            algorithm1,
            {"name": algorithm1.replace("_", " ").title(), "color": "#E74C3C"},
        )
        info2 = ALGORITHM_INFO.get(
            algorithm2,
            {"name": algorithm2.replace("_", " ").title(), "color": "#3498DB"},
        )

        metric_key = "time_seconds" if metric == "time" else "operations"
        ylabel = "Execution Time (seconds)" if metric == "time" else "Basic Operations"
        title_metric = "Time" if metric == "time" else "Operations"

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot algorithm 1
        data1 = sorted(
            [(r.get("n_vertices", 0), r.get(metric_key, 0)) for r in alg1_results]
        )
        v1, m1 = zip(*data1) if data1 else ([], [])
        ax.plot(
            v1,
            m1,
            "o-",
            color=info1["color"],
            linewidth=2,
            markersize=8,
            label=info1["name"],
        )

        # Plot algorithm 2
        data2 = sorted(
            [(r.get("n_vertices", 0), r.get(metric_key, 0)) for r in alg2_results]
        )
        v2, m2 = zip(*data2) if data2 else ([], [])
        ax.plot(
            v2,
            m2,
            "s-",
            color=info2["color"],
            linewidth=2,
            markersize=8,
            label=info2["name"],
        )

        ax.set_xlabel("Number of Vertices", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"{title_metric} Comparison: {info1['name']} vs {info2['name']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Use log scale if needed
        all_values = list(m1) + list(m2)
        if all_values and max(all_values) / (min(all_values) + 1e-10) > 100:
            ax.set_yscale("log")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Pairwise {metric} line comparison saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_all_individual_charts(
        results: list[dict],
        output_dir: Path = Path("experiments/plots/individual"),
    ) -> None:
        """
        Generate individual charts for all algorithms in the results.

        Args:
            results: List of benchmark results
            output_dir: Directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get unique algorithms
        algorithms = set(r.get("algorithm") for r in results)

        print(f"\n{'=' * 60}")
        print(f"GENERATING INDIVIDUAL ALGORITHM CHARTS")
        print(f"Found {len(algorithms)} algorithms")
        print(f"{'=' * 60}\n")

        for alg in sorted(algorithms):
            # Time chart
            ResultsVisualizer.plot_individual_algorithm_time(
                results, alg, output_path=output_dir / f"{alg}_time.png", show=False
            )

            # Operations chart
            ResultsVisualizer.plot_individual_algorithm_operations(
                results,
                alg,
                output_path=output_dir / f"{alg}_operations.png",
                show=False,
            )

        print(f"\n✓ All individual charts saved to {output_dir}")

    @staticmethod
    def plot_all_pairwise_comparisons(
        results: list[dict],
        output_dir: Path = Path("experiments/plots/pairwise"),
        comparison_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Generate pairwise comparison charts between algorithms.

        Args:
            results: List of benchmark results
            output_dir: Directory to save plots
            comparison_pairs: List of (alg1, alg2) tuples to compare.
                            If None, compares all pairs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get unique algorithms
        algorithms = sorted(set(r.get("algorithm") for r in results))

        if comparison_pairs is None:
            # Generate meaningful pairs based on categories
            comparison_pairs = ResultsVisualizer._generate_comparison_pairs(algorithms)

        print(f"\n{'=' * 60}")
        print(f"GENERATING PAIRWISE COMPARISON CHARTS")
        print(f"Comparing {len(comparison_pairs)} algorithm pairs")
        print(f"{'=' * 60}\n")

        for alg1, alg2 in comparison_pairs:
            if alg1 not in algorithms or alg2 not in algorithms:
                continue

            # Time comparison (bar chart)
            ResultsVisualizer.plot_pairwise_time_comparison(
                results,
                alg1,
                alg2,
                output_path=output_dir / f"{alg1}_vs_{alg2}_time_bar.png",
                show=False,
            )

            # Operations comparison (bar chart)
            ResultsVisualizer.plot_pairwise_operations_comparison(
                results,
                alg1,
                alg2,
                output_path=output_dir / f"{alg1}_vs_{alg2}_operations_bar.png",
                show=False,
            )

            # Time comparison (line chart)
            ResultsVisualizer.plot_pairwise_line_comparison(
                results,
                alg1,
                alg2,
                metric="time",
                output_path=output_dir / f"{alg1}_vs_{alg2}_time_line.png",
                show=False,
            )

            # Operations comparison (line chart)
            ResultsVisualizer.plot_pairwise_line_comparison(
                results,
                alg1,
                alg2,
                metric="operations",
                output_path=output_dir / f"{alg1}_vs_{alg2}_operations_line.png",
                show=False,
            )

        print(f"\n✓ All pairwise comparison charts saved to {output_dir}")

    @staticmethod
    def _generate_comparison_pairs(algorithms: list[str]) -> list[tuple[str, str]]:
        """
        Generate meaningful comparison pairs based on algorithm categories.

        Compares:
        - Exact vs Exact
        - Heuristic vs Heuristic
        - Each heuristic vs an exact (for precision reference)
        """
        pairs = []

        # Define categories
        categories = {
            "exact": ["exhaustive", "wlmc", "tsm_mwc"],
            "reduction": ["mwc_redu", "max_clique_weight", "max_clique_dyn_weight"],
            "heuristic": ["greedy", "fast_wclq", "scc_walk", "mwc_peel"],
            "randomized": [
                "random_construction",
                "random_greedy_hybrid",
                "iterative_random_search",
                "monte_carlo",
                "las_vegas",
            ],
        }

        # Filter to only available algorithms
        available = {
            cat: [a for a in algs if a in algorithms]
            for cat, algs in categories.items()
        }

        # Key comparisons
        key_comparisons = [
            # Baseline comparisons
            ("exhaustive", "greedy"),
            ("exhaustive", "mwc_redu"),
            ("greedy", "fast_wclq"),
            ("greedy", "mwc_peel"),
            # Exact algorithm comparisons
            ("wlmc", "tsm_mwc"),
            ("exhaustive", "wlmc"),
            # Heuristic comparisons
            ("fast_wclq", "scc_walk"),
            ("fast_wclq", "mwc_peel"),
            ("scc_walk", "mwc_peel"),
            # Reduction comparisons
            ("mwc_redu", "max_clique_weight"),
            ("max_clique_weight", "max_clique_dyn_weight"),
            # Randomized comparisons
            ("random_construction", "random_greedy_hybrid"),
            ("iterative_random_search", "monte_carlo"),
            ("monte_carlo", "las_vegas"),
            # Cross-category comparisons
            ("greedy", "random_greedy_hybrid"),
            ("mwc_redu", "greedy"),
        ]

        # Add pairs that exist in the data
        for alg1, alg2 in key_comparisons:
            if alg1 in algorithms and alg2 in algorithms:
                pairs.append((alg1, alg2))

        return pairs

    @staticmethod
    def plot_category_summary(
        results: list[dict],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Create summary chart comparing algorithm categories.

        Args:
            results: List of benchmark results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        if not results:
            print("No data for category summary")
            return

        # Group by category
        categories = {
            "Exact": ["exhaustive", "wlmc", "tsm_mwc"],
            "Reduction": ["mwc_redu", "max_clique_weight", "max_clique_dyn_weight"],
            "Heuristic": ["greedy", "fast_wclq", "scc_walk", "mwc_peel"],
            "Randomized": [
                "random_construction",
                "random_greedy_hybrid",
                "iterative_random_search",
                "monte_carlo",
                "las_vegas",
            ],
        }

        category_colors = {
            "Exact": "#E74C3C",
            "Reduction": "#00BCD4",
            "Heuristic": "#3498DB",
            "Randomized": "#2ECC71",
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Aggregate data by category
        cat_times: dict[str, list[float]] = {cat: [] for cat in categories}
        cat_ops: dict[str, list[float]] = {cat: [] for cat in categories}

        for r in results:
            alg = r.get("algorithm", "")
            for cat, algs in categories.items():
                if alg in algs:
                    cat_times[cat].append(r.get("time_seconds", 0))
                    cat_ops[cat].append(r.get("operations", 0))
                    break

        # Plot average time by category
        ax1 = axes[0]
        cats_with_data = [c for c in categories if cat_times[c]]
        avg_times = [
            sum(cat_times[c]) / len(cat_times[c]) if cat_times[c] else 0
            for c in cats_with_data
        ]
        colors = [category_colors[c] for c in cats_with_data]

        ax1.bar(cats_with_data, avg_times, color=colors, alpha=0.8)
        ax1.set_ylabel("Average Execution Time (s)", fontsize=12)
        ax1.set_title("Average Time by Category", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")
        if avg_times and max(avg_times) / (min(avg_times) + 1e-10) > 100:
            ax1.set_yscale("log")

        # Plot average operations by category
        ax2 = axes[1]
        avg_ops = [
            sum(cat_ops[c]) / len(cat_ops[c]) if cat_ops[c] else 0
            for c in cats_with_data
        ]

        ax2.bar(cats_with_data, avg_ops, color=colors, alpha=0.8)
        ax2.set_ylabel("Average Operations", fontsize=12)
        ax2.set_title("Average Operations by Category", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        if avg_ops and max(avg_ops) / (min(avg_ops) + 1) > 100:
            ax2.set_yscale("log")

        plt.suptitle(
            "Algorithm Category Performance Summary", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Category summary saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_all_algorithms_comparison(
        results: list[dict],
        metric: str = "time",
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Create a single chart comparing all algorithms.

        Args:
            results: List of benchmark results
            metric: "time" or "operations"
            output_path: Path to save figure
            show: Whether to display the plot
        """
        if not results:
            print("No data for all algorithms comparison")
            return

        algorithms = ResultsVisualizer._extract_algorithm_data(results)

        metric_key = "time_seconds" if metric == "time" else "operations"
        ylabel = "Execution Time (seconds)" if metric == "time" else "Basic Operations"
        title_metric = "Time" if metric == "time" else "Operations"

        fig, ax = plt.subplots(figsize=(14, 8))

        for alg_name, alg_data in sorted(algorithms.items()):
            if not alg_data.has_data():
                continue

            data = sorted(
                [
                    (r.get("n_vertices", 0), r.get(metric_key, 0))
                    for r in results
                    if r.get("algorithm") == alg_name
                ]
            )
            if not data:
                continue

            vertices, values = zip(*data)
            ax.plot(
                vertices,
                values,
                "o-",
                color=alg_data.color,
                linewidth=2,
                markersize=6,
                label=alg_data.display_name,
                alpha=0.8,
            )

        ax.set_xlabel("Number of Vertices", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"All Algorithms - {title_metric} Comparison",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Use log scale if needed
        all_values = [r.get(metric_key, 0) for r in results]
        if all_values and max(all_values) / (min(all_values) + 1e-10) > 100:
            ax.set_yscale("log")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ All algorithms {metric} comparison saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def generate_all_visualizations(
        results: list[dict],
        output_dir: Path = Path("experiments/plots"),
    ) -> None:
        """
        Generate all visualization types: individual, pairwise, and summary charts.

        Args:
            results: List of benchmark results
            output_dir: Base directory for all plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("GENERATING ALL VISUALIZATIONS")
        print(f"{'=' * 60}\n")

        # 1. Individual algorithm charts
        ResultsVisualizer.plot_all_individual_charts(results, output_dir / "individual")

        # 2. Pairwise comparison charts
        ResultsVisualizer.plot_all_pairwise_comparisons(
            results, output_dir / "pairwise"
        )

        # 3. Summary charts
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        ResultsVisualizer.plot_category_summary(
            results, output_path=summary_dir / "category_summary.png", show=False
        )

        ResultsVisualizer.plot_all_algorithms_comparison(
            results,
            metric="time",
            output_path=summary_dir / "all_algorithms_time.png",
            show=False,
        )

        ResultsVisualizer.plot_all_algorithms_comparison(
            results,
            metric="operations",
            output_path=summary_dir / "all_algorithms_operations.png",
            show=False,
        )

        print(f"\n{'=' * 60}")
        print(f"✓ ALL VISUALIZATIONS COMPLETE")
        print(f"  - Individual charts: {output_dir}/individual/")
        print(f"  - Pairwise charts: {output_dir}/pairwise/")
        print(f"  - Summary charts: {output_dir}/summary/")
        print(f"{'=' * 60}\n")

    # ================================================================================
    # LEGACY METHODS - Kept for backward compatibility with BenchmarkResult
    # ================================================================================

    @staticmethod
    def plot_execution_time(
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot execution time vs number of vertices (legacy BenchmarkResult format).

        Args:
            results: list of benchmark results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        # Group by density
        densities: dict[float, list[BenchmarkResult]] = {}
        for result in results:
            density = result.edge_density_percent
            if density not in densities:
                densities[density] = []
            densities[density].append(result)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot exact algorithm
        for density in sorted(densities.keys()):
            results_at_density = sorted(densities[density], key=lambda r: r.n_vertices)
            # Filter out None values
            data_points = [
                (r.n_vertices, r.exact_time_seconds)
                for r in results_at_density
                if r.exact_time_seconds is not None
            ]
            if data_points:
                vertices, times = zip(*data_points)
                ax1.plot(vertices, times, marker="o", label=f"{density:.1f}% density")

        ax1.set_xlabel("Number of Vertices", fontsize=12)
        ax1.set_ylabel("Execution Time (seconds)", fontsize=12)
        ax1.set_title(
            "Exhaustive Search - Execution Time", fontsize=14, fontweight="bold"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # ax1.set_yscale('log')

        # Plot greedy algorithm
        for density in sorted(densities.keys()):
            results_at_density = sorted(densities[density], key=lambda r: r.n_vertices)
            # Filter out None values
            data_points = [
                (r.n_vertices, r.greedy_time_seconds)
                for r in results_at_density
                if r.greedy_time_seconds is not None
            ]
            if data_points:
                vertices, times = zip(*data_points)
                ax2.plot(vertices, times, marker="s", label=f"{density:.1f}% density")

        ax2.set_xlabel("Number of Vertices", fontsize=12)
        ax2.set_ylabel("Execution Time (seconds)", fontsize=12)
        ax2.set_title(
            "Greedy Heuristic - Execution Time", fontsize=14, fontweight="bold"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # ax2.set_yscale('log')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Execution time plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_operations_count(
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot basic operations count vs number of vertices.

        Args:
            results: list of benchmark results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        # Group by density
        densities: dict[float, list[BenchmarkResult]] = {}
        for result in results:
            density = result.edge_density_percent
            if density not in densities:
                densities[density] = []
            densities[density].append(result)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot for each density
        for density in sorted(densities.keys()):
            results_at_density = sorted(densities[density], key=lambda r: r.n_vertices)
            # Filter out None values
            data_points = [
                (r.n_vertices, r.exact_operations)
                for r in results_at_density
                if r.exact_operations is not None
            ]
            if data_points:
                vertices, operations = zip(*data_points)
                ax.plot(
                    vertices, operations, marker="o", label=f"{density:.1f}% density"
                )

        ax.set_xlabel("Number of Vertices", fontsize=12)
        ax.set_ylabel("Basic Operations (adjacency checks)", fontsize=12)
        ax.set_title(
            "Exhaustive Search - Operations Count", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        # ax.set_yscale('log')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Operations count plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_configurations_tested(
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot number of configurations tested vs number of vertices.

        Args:
            results: list of benchmark results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        results_sorted = sorted(results, key=lambda r: r.n_vertices)
        vertices = [r.n_vertices for r in results_sorted]
        configs = [r.exact_configurations or 0 for r in results_sorted]

        # Plot configurations tested
        ax.plot(
            vertices,
            configs,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Configurations tested",
            color="darkblue",
        )

        # Plot theoretical 2^n line for reference
        import numpy as np

        v_range = np.array(vertices)
        theoretical = 2**v_range
        ax.plot(
            v_range,
            theoretical,
            "--",
            linewidth=1.5,
            label="Theoretical $2^n$",
            color="red",
            alpha=0.7,
        )

        ax.set_xlabel("Number of Vertices", fontsize=12)
        ax.set_ylabel("Number of Configurations", fontsize=12)
        ax.set_title(
            "Exhaustive Search - Configurations Tested", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        # ax.set_yscale('log')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Configurations plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_heuristic_precision(
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot greedy heuristic precision vs number of vertices and density.

        Args:
            results: list of benchmark results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        # Group by density
        densities: dict[float, list[BenchmarkResult]] = {}
        for result in results:
            density = result.edge_density_percent
            if density not in densities:
                densities[density] = []
            densities[density].append(result)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot for each density
        for density in sorted(densities.keys()):
            results_at_density = sorted(densities[density], key=lambda r: r.n_vertices)
            # Filter out None values
            data_points = [
                (r.n_vertices, r.precision_percent)
                for r in results_at_density
                if r.precision_percent is not None
            ]
            if data_points:
                vertices, precision = zip(*data_points)
                ax.plot(
                    vertices, precision, marker="o", label=f"{density:.1f}% density"
                )

        ax.set_xlabel("Number of Vertices", fontsize=12)
        ax.set_ylabel("Precision (%)", fontsize=12)
        ax.set_title("Greedy Heuristic Precision", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim((0.0, 105.0))
        ax.axhline(y=100, color="green", linestyle="--", linewidth=1, alpha=0.5)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Precision plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_randomized_performance(
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot randomized algorithm performance compared to exact and greedy.

        Args:
            results: list of benchmark results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        # Filter results that have random algorithm data
        random_results = [r for r in results if r.random_time_seconds is not None]

        if not random_results:
            print("No randomized algorithm results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract data
        vertices = [r.n_vertices for r in random_results]
        random_times = [r.random_time_seconds for r in random_results]
        exact_times = [
            r.exact_time_seconds
            for r in random_results
            if r.exact_time_seconds is not None
        ]
        greedy_times = [
            r.greedy_time_seconds
            for r in random_results
            if r.greedy_time_seconds is not None
        ]
        random_precision = [
            r.random_precision_percent
            for r in random_results
            if r.random_precision_percent is not None
        ]

        # Plot 1: Execution time comparison
        ax1 = axes[0, 0]
        ax1.plot(vertices, random_times, "o-", label="Randomized", color="green")
        if exact_times:
            exact_vertices = [
                r.n_vertices for r in random_results if r.exact_time_seconds is not None
            ]
            ax1.plot(exact_vertices, exact_times, "s-", label="Exact", color="red")
        if greedy_times:
            greedy_vertices = [
                r.n_vertices
                for r in random_results
                if r.greedy_time_seconds is not None
            ]
            ax1.plot(greedy_vertices, greedy_times, "^-", label="Greedy", color="blue")
        ax1.set_xlabel("Number of Vertices", fontsize=12)
        ax1.set_ylabel("Execution Time (seconds)", fontsize=12)
        ax1.set_title("Execution Time Comparison", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Plot 2: Precision
        ax2 = axes[0, 1]
        if random_precision:
            prec_vertices = [
                r.n_vertices
                for r in random_results
                if r.random_precision_percent is not None
            ]
            ax2.plot(
                prec_vertices,
                random_precision,
                "o-",
                label="Randomized Precision",
                color="green",
            )
            ax2.axhline(
                y=100,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Optimal",
            )
        ax2.set_xlabel("Number of Vertices", fontsize=12)
        ax2.set_ylabel("Precision (%)", fontsize=12)
        ax2.set_title("Randomized Algorithm Precision", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim((0.0, 105.0))

        # Plot 3: Configuration efficiency
        ax3 = axes[1, 0]
        total_configs = [
            r.random_configurations
            for r in random_results
            if r.random_configurations is not None
        ]
        unique_configs = [
            r.random_unique_configurations
            for r in random_results
            if r.random_unique_configurations is not None
        ]
        config_vertices = [
            r.n_vertices for r in random_results if r.random_configurations is not None
        ]
        if total_configs and unique_configs:
            ax3.plot(
                config_vertices,
                total_configs,
                "o-",
                label="Total Configurations",
                color="orange",
            )
            ax3.plot(
                config_vertices,
                unique_configs,
                "s-",
                label="Unique Configurations",
                color="purple",
            )
        ax3.set_xlabel("Number of Vertices", fontsize=12)
        ax3.set_ylabel("Number of Configurations", fontsize=12)
        ax3.set_title("Configuration Efficiency", fontsize=14, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

        # Plot 4: Duplicate rate
        ax4 = axes[1, 1]
        duplicates = [
            r.random_duplicates
            for r in random_results
            if r.random_duplicates is not None
        ]
        dup_vertices = [
            r.n_vertices for r in random_results if r.random_duplicates is not None
        ]
        if duplicates and total_configs:
            duplicate_rates = [
                d / t * 100 if t > 0 else 0 for d, t in zip(duplicates, total_configs)
            ]
            ax4.plot(
                dup_vertices, duplicate_rates, "o-", label="Duplicate Rate", color="red"
            )
        ax4.set_xlabel("Number of Vertices", fontsize=12)
        ax4.set_ylabel("Duplicate Rate (%)", fontsize=12)
        ax4.set_title("Duplicate Configuration Rate", fontsize=14, fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Randomized performance plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_configuration_efficiency(
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot unique vs total configurations tested for randomized algorithms.

        Args:
            results: list of benchmark results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        random_results = [
            r for r in results if r.random_unique_configurations is not None
        ]

        if not random_results:
            print("No randomized algorithm results to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        vertices = [r.n_vertices for r in random_results]
        total = [r.random_configurations or 0 for r in random_results]
        unique = [r.random_unique_configurations or 0 for r in random_results]
        duplicates = [r.random_duplicates or 0 for r in random_results]

        x = range(len(vertices))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            unique,
            width,
            label="Unique",
            color="green",
            alpha=0.7,
        )
        ax.bar(
            [i + width / 2 for i in x],
            duplicates,
            width,
            label="Duplicates",
            color="red",
            alpha=0.7,
        )

        ax.set_xlabel("Graph Index", fontsize=12)
        ax.set_ylabel("Number of Configurations", fontsize=12)
        ax.set_title(
            "Configuration Efficiency: Unique vs Duplicates",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"n={v}" for v in vertices], rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Configuration efficiency plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_stopping_criteria_analysis(
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Analyze stopping criteria for randomized algorithms.

        Args:
            results: list of benchmark results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        random_results = [r for r in results if r.random_stopping_reason is not None]

        if not random_results:
            print("No randomized algorithm results to plot")
            return

        # Count stopping reasons
        stopping_reasons = Counter(r.random_stopping_reason for r in random_results)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart of stopping reasons
        if stopping_reasons:
            labels = list(stopping_reasons.keys())
            sizes = list(stopping_reasons.values())
            colors = plt.cm.Set3(range(len(labels)))

            ax1.pie(
                sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
            )
            ax1.set_title(
                "Stopping Reasons Distribution", fontsize=14, fontweight="bold"
            )

        # Bar chart by graph size
        ax2_data: dict[str, list[int]] = {}
        for r in random_results:
            reason = r.random_stopping_reason or "unknown"
            if reason not in ax2_data:
                ax2_data[reason] = []
            ax2_data[reason].append(r.n_vertices)

        if ax2_data:
            x_pos = range(len(ax2_data))
            means = [sum(ax2_data[r]) / len(ax2_data[r]) for r in ax2_data.keys()]
            ax2.bar(x_pos, means, color=plt.cm.Set3(range(len(ax2_data))))
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(list(ax2_data.keys()), rotation=45, ha="right")
            ax2.set_ylabel("Average Graph Size", fontsize=12)
            ax2.set_title(
                "Average Graph Size by Stopping Reason", fontsize=14, fontweight="bold"
            )
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Stopping criteria analysis plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_all_metrics(
        results: list[BenchmarkResult],
        output_dir: Path = Path("experiments/plots"),
    ) -> None:
        """
        Generate all performance plots.

        Args:
            results: list of benchmark results
            output_dir: Directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("GENERATING PLOTS")
        print(f"{'=' * 60}\n")

        ResultsVisualizer.plot_execution_time(
            results, output_dir / "execution_time.png", show=False
        )
        ResultsVisualizer.plot_operations_count(
            results, output_dir / "operations_count.png", show=False
        )
        ResultsVisualizer.plot_configurations_tested(
            results, output_dir / "configurations_tested.png", show=False
        )
        ResultsVisualizer.plot_heuristic_precision(
            results, output_dir / "heuristic_precision.png", show=False
        )

        # Add randomized algorithm plots if data is available
        has_random_data = any(r.random_time_seconds is not None for r in results)
        if has_random_data:
            ResultsVisualizer.plot_randomized_performance(
                results, output_dir / "randomized_performance.png", show=False
            )
            ResultsVisualizer.plot_configuration_efficiency(
                results, output_dir / "configuration_efficiency.png", show=False
            )
            ResultsVisualizer.plot_stopping_criteria_analysis(
                results, output_dir / "stopping_criteria_analysis.png", show=False
            )

        print(f"\n✓ All plots saved to {output_dir}")

    @staticmethod
    def plot_algorithm_comparison(
        results_by_algorithm: dict[str, list[dict]],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot comparison of multiple algorithms on the same graphs.

        Args:
            results_by_algorithm: Dict mapping algorithm name to list of results
                Each result should have: n_vertices, weight, time_seconds, operations
            output_path: Path to save figure
            show: Whether to display the plot
        """
        if not results_by_algorithm:
            print("No algorithm comparison data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = plt.cm.tab10(range(len(results_by_algorithm)))

        # Plot 1: Solution quality (weight) vs graph size
        ax1 = axes[0, 0]
        for (alg_name, results), color in zip(results_by_algorithm.items(), colors):
            if results:
                vertices = [r.get("n_vertices", 0) for r in results]
                weights = [r.get("weight", 0) for r in results]
                ax1.plot(vertices, weights, "o-", label=alg_name, color=color)
        ax1.set_xlabel("Number of Vertices", fontsize=12)
        ax1.set_ylabel("Solution Weight", fontsize=12)
        ax1.set_title("Solution Quality Comparison", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Execution time vs graph size
        ax2 = axes[0, 1]
        for (alg_name, results), color in zip(results_by_algorithm.items(), colors):
            if results:
                vertices = [r.get("n_vertices", 0) for r in results]
                times = [r.get("time_seconds", 0) for r in results]
                ax2.plot(vertices, times, "o-", label=alg_name, color=color)
        ax2.set_xlabel("Number of Vertices", fontsize=12)
        ax2.set_ylabel("Execution Time (seconds)", fontsize=12)
        ax2.set_title("Execution Time Comparison", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        # Plot 3: Operations count vs graph size
        ax3 = axes[1, 0]
        for (alg_name, results), color in zip(results_by_algorithm.items(), colors):
            if results:
                vertices = [r.get("n_vertices", 0) for r in results]
                ops = [r.get("operations", 0) for r in results]
                ax3.plot(vertices, ops, "o-", label=alg_name, color=color)
        ax3.set_xlabel("Number of Vertices", fontsize=12)
        ax3.set_ylabel("Basic Operations", fontsize=12)
        ax3.set_title("Operations Count Comparison", fontsize=14, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

        # Plot 4: Solution quality relative to optimal (if available)
        ax4 = axes[1, 1]
        optimal_alg = None
        for alg_name in results_by_algorithm.keys():
            if "exhaustive" in alg_name.lower() or "exact" in alg_name.lower():
                optimal_alg = alg_name
                break

        if optimal_alg and optimal_alg in results_by_algorithm:
            optimal_results = results_by_algorithm[optimal_alg]
            optimal_weights = {
                r.get("n_vertices", 0): r.get("weight", 0) for r in optimal_results
            }

            for (alg_name, results), color in zip(results_by_algorithm.items(), colors):
                if alg_name == optimal_alg:
                    continue
                if results:
                    vertices = []
                    precisions = []
                    for r in results:
                        n = r.get("n_vertices", 0)
                        w = r.get("weight", 0)
                        if n in optimal_weights and optimal_weights[n] > 0:
                            vertices.append(n)
                            precisions.append(w / optimal_weights[n] * 100)
                    if vertices:
                        ax4.plot(
                            vertices, precisions, "o-", label=alg_name, color=color
                        )

            ax4.axhline(
                y=100,
                color="green",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label="Optimal",
            )
            ax4.set_xlabel("Number of Vertices", fontsize=12)
            ax4.set_ylabel("Precision (%)", fontsize=12)
            ax4.set_title(
                "Algorithm Precision vs Optimal", fontsize=14, fontweight="bold"
            )
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim((0, 105))
        else:
            ax4.text(
                0.5,
                0.5,
                "No optimal reference\navailable",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=14,
            )
            ax4.set_title(
                "Algorithm Precision vs Optimal", fontsize=14, fontweight="bold"
            )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Algorithm comparison plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_convergence(
        iterations: list[int],
        weights: list[float],
        times: list[float] | None = None,
        algorithm_name: str = "Algorithm",
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot convergence of a randomized algorithm over iterations.

        Args:
            iterations: List of iteration numbers
            weights: List of best weights at each iteration
            times: Optional list of elapsed times at each iteration
            algorithm_name: Name of the algorithm
            output_path: Path to save figure
            show: Whether to display the plot
        """
        if not iterations or not weights:
            print("No convergence data to plot")
            return

        fig, axes = plt.subplots(1, 2 if times else 1, figsize=(14 if times else 10, 5))

        if times:
            ax1, ax2 = axes
        else:
            ax1 = axes

        # Plot weight vs iterations
        ax1.plot(iterations, weights, "b-", linewidth=2)
        ax1.fill_between(iterations, 0, weights, alpha=0.3)
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Best Weight Found", fontsize=12)
        ax1.set_title(f"{algorithm_name} Convergence", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Annotate final value
        if weights:
            ax1.annotate(
                f"Final: {weights[-1]:.2f}",
                xy=(iterations[-1], weights[-1]),
                xytext=(iterations[-1] * 0.8, weights[-1] * 1.02),
                fontsize=10,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"),
            )

        # Plot weight vs time if available
        if times:
            ax2.plot(times, weights, "g-", linewidth=2)
            ax2.fill_between(times, 0, weights, alpha=0.3, color="green")
            ax2.set_xlabel("Elapsed Time (seconds)", fontsize=12)
            ax2.set_ylabel("Best Weight Found", fontsize=12)
            ax2.set_title(
                f"{algorithm_name} Convergence over Time",
                fontsize=14,
                fontweight="bold",
            )
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Convergence plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_quality_vs_time_tradeoff(
        results: list[dict],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot solution quality vs execution time tradeoff.

        Args:
            results: List of dicts with keys: algorithm, weight, time_seconds, precision
            output_path: Path to save figure
            show: Whether to display the plot
        """
        if not results:
            print("No tradeoff data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        algorithms = set(r.get("algorithm", "Unknown") for r in results)
        colors = plt.cm.tab10(range(len(algorithms)))
        color_map = dict(zip(algorithms, colors))

        for r in results:
            alg = r.get("algorithm", "Unknown")
            time_s = r.get("time_seconds", 0)
            precision = r.get("precision", 100)
            n_vertices = r.get("n_vertices", 0)

            ax.scatter(
                time_s,
                precision,
                c=[color_map[alg]],
                s=n_vertices * 5,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

        # Add legend
        handles = [
            plt.scatter([], [], c=[color_map[alg]], s=100, label=alg)
            for alg in algorithms
        ]
        ax.legend(handles=handles, title="Algorithm")

        ax.axhline(y=100, color="green", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_xlabel("Execution Time (seconds)", fontsize=12)
        ax.set_ylabel("Precision (%)", fontsize=12)
        ax.set_title(
            "Quality vs Time Tradeoff\n(point size = graph size)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Quality vs time tradeoff plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_scalability_analysis(
        results_by_algorithm: dict[str, list[dict]],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot scalability analysis showing how algorithms scale with graph size.

        Args:
            results_by_algorithm: Dict mapping algorithm name to list of results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        if not results_by_algorithm:
            print("No scalability data to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1, ax2 = axes

        colors = plt.cm.tab10(range(len(results_by_algorithm)))

        # Plot 1: Time scalability
        for (alg_name, results), color in zip(results_by_algorithm.items(), colors):
            if results:
                vertices = sorted(set(r.get("n_vertices", 0) for r in results))
                avg_times = []
                for v in vertices:
                    times = [
                        r.get("time_seconds", 0)
                        for r in results
                        if r.get("n_vertices") == v
                    ]
                    avg_times.append(sum(times) / len(times) if times else 0)
                ax1.plot(vertices, avg_times, "o-", label=alg_name, color=color)

        ax1.set_xlabel("Number of Vertices", fontsize=12)
        ax1.set_ylabel("Average Execution Time (s)", fontsize=12)
        ax1.set_title("Time Scalability", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Plot 2: Operations scalability
        for (alg_name, results), color in zip(results_by_algorithm.items(), colors):
            if results:
                vertices = sorted(set(r.get("n_vertices", 0) for r in results))
                avg_ops = []
                for v in vertices:
                    ops = [
                        r.get("operations", 0)
                        for r in results
                        if r.get("n_vertices") == v
                    ]
                    avg_ops.append(sum(ops) / len(ops) if ops else 0)
                ax2.plot(vertices, avg_ops, "o-", label=alg_name, color=color)

        ax2.set_xlabel("Number of Vertices", fontsize=12)
        ax2.set_ylabel("Average Operations", fontsize=12)
        ax2.set_title("Operations Scalability", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Scalability analysis plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_density_impact(
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot the impact of graph density on algorithm performance.

        Args:
            results: List of benchmark results
            output_path: Path to save figure
            show: Whether to display the plot
        """
        if not results:
            print("No density impact data to plot")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Group by density
        densities_data: dict[float, list[BenchmarkResult]] = {}
        for r in results:
            d = r.edge_density_percent
            if d not in densities_data:
                densities_data[d] = []
            densities_data[d].append(r)

        densities = sorted(densities_data.keys())

        # Plot 1: Average precision by density
        ax1 = axes[0]
        avg_precisions = []
        for d in densities:
            precisions = [
                r.precision_percent
                for r in densities_data[d]
                if r.precision_percent is not None
            ]
            avg_precisions.append(
                sum(precisions) / len(precisions) if precisions else 0
            )
        ax1.bar(range(len(densities)), avg_precisions, color="steelblue")
        ax1.set_xticks(range(len(densities)))
        ax1.set_xticklabels([f"{d:.0f}%" for d in densities])
        ax1.set_xlabel("Edge Density", fontsize=12)
        ax1.set_ylabel("Average Precision (%)", fontsize=12)
        ax1.set_title("Heuristic Precision by Density", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.set_ylim((0, 105))

        # Plot 2: Average execution time by density
        ax2 = axes[1]
        avg_exact_times = []
        avg_greedy_times = []
        for d in densities:
            exact_times = [
                r.exact_time_seconds
                for r in densities_data[d]
                if r.exact_time_seconds is not None
            ]
            greedy_times = [
                r.greedy_time_seconds
                for r in densities_data[d]
                if r.greedy_time_seconds is not None
            ]
            avg_exact_times.append(
                sum(exact_times) / len(exact_times) if exact_times else 0
            )
            avg_greedy_times.append(
                sum(greedy_times) / len(greedy_times) if greedy_times else 0
            )

        x = range(len(densities))
        width = 0.35
        ax2.bar(
            [i - width / 2 for i in x],
            avg_exact_times,
            width,
            label="Exact",
            color="red",
            alpha=0.7,
        )
        ax2.bar(
            [i + width / 2 for i in x],
            avg_greedy_times,
            width,
            label="Greedy",
            color="blue",
            alpha=0.7,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{d:.0f}%" for d in densities])
        ax2.set_xlabel("Edge Density", fontsize=12)
        ax2.set_ylabel("Average Time (s)", fontsize=12)
        ax2.set_title("Execution Time by Density", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # Plot 3: Clique size distribution by density
        ax3 = axes[2]
        clique_sizes: dict[float, list[int]] = {}
        for d in densities:
            sizes = [
                r.exact_clique_size
                for r in densities_data[d]
                if r.exact_clique_size is not None
            ]
            clique_sizes[d] = sizes

        positions = []
        data = []
        labels = []
        for i, d in enumerate(densities):
            if clique_sizes[d]:
                positions.append(i)
                data.append(clique_sizes[d])
                labels.append(f"{d:.0f}%")

        if data:
            bp = ax3.boxplot(data, positions=positions, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightgreen")
            ax3.set_xticks(positions)
            ax3.set_xticklabels(labels)
        ax3.set_xlabel("Edge Density", fontsize=12)
        ax3.set_ylabel("Clique Size", fontsize=12)
        ax3.set_title(
            "Clique Size Distribution by Density", fontsize=14, fontweight="bold"
        )
        ax3.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Density impact plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Generate sample visualizations."""
    # Create a sample graph
    G: nx.Graph = nx.Graph()
    G.add_node(0, x=100, y=100, weight=10.0)
    G.add_node(1, x=200, y=150, weight=20.0)
    G.add_node(2, x=150, y=250, weight=15.0)
    G.add_node(3, x=300, y=200, weight=5.0)
    G.add_node(4, x=250, y=350, weight=12.0)

    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4)])

    # Visualize with a clique
    clique = {0, 1, 2}

    output_dir = Path("experiments/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    GraphVisualizer.visualize_graph_with_clique(
        G,
        clique,
        title="Sample Graph with Clique",
        output_path=output_dir / "sample_graph.png",
        show=False,
    )

    print("Sample visualization created!")


if __name__ == "__main__":
    main()
