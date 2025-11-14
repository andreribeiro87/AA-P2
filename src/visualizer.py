"""
Visualization module for graphs and benchmark results.

Creates visual representations of graphs, cliques, and performance metrics.
"""

from pathlib import Path
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.benchmark import BenchmarkResult


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
        pos = {node: (graph.nodes[node]['x'], graph.nodes[node]['y'])
               for node in graph.nodes()}

        # Separate nodes into clique and non-clique
        clique_nodes = list(clique)
        other_nodes = [n for n in graph.nodes() if n not in clique]

        # Draw edges
        # Clique internal edges (bold red)
        clique_edges = [(u, v) for u, v in graph.edges()
                        if u in clique and v in clique]
        nx.draw_networkx_edges(
            graph, pos, edgelist=clique_edges,
            width=2.5, alpha=0.8, edge_color='red', ax=ax
        )

        # Other edges (thin gray)
        other_edges = [(u, v) for u, v in graph.edges()
                       if (u, v) not in clique_edges]
        nx.draw_networkx_edges(
            graph, pos, edgelist=other_edges,
            width=1.0, alpha=0.3, edge_color='gray', ax=ax
        )

        # Draw nodes
        # Clique nodes (red)
        if clique_nodes:
            clique_weights = [graph.nodes[n]['weight'] for n in clique_nodes]
            nx.draw_networkx_nodes(
                graph, pos, nodelist=clique_nodes,
                node_color='red', node_size=[w*10 for w in clique_weights],
                alpha=0.9, ax=ax
            )

        # Other nodes (light blue)
        if other_nodes:
            other_weights = [graph.nodes[n]['weight'] for n in other_nodes]
            nx.draw_networkx_nodes(
                graph, pos, nodelist=other_nodes,
                node_color='lightblue', node_size=[w*10 for w in other_weights],
                alpha=0.7, ax=ax
            )

        # Draw labels
        labels = {node: f"{node}\n({graph.nodes[node]['weight']:.1f})"
                  for node in graph.nodes()}
        nx.draw_networkx_labels(
            graph, pos, labels, font_size=8, font_weight='bold', ax=ax
        )

        # Add legend
        clique_weight = sum(graph.nodes[n]['weight'] for n in clique)
        legend_elements = [
            mpatches.Patch(color='red', label=f'Clique (weight={clique_weight:.2f})'),
            mpatches.Patch(color='lightblue', label='Other vertices')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Graph visualization saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()


class ResultsVisualizer:
    """Visualize benchmark results and performance metrics."""

    @staticmethod
    def plot_execution_time(
        results: list[BenchmarkResult],
        output_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot execution time vs number of vertices.

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
            data_points = [(r.n_vertices, r.exact_time_seconds) 
                          for r in results_at_density 
                          if r.exact_time_seconds is not None]
            if data_points:
                vertices, times = zip(*data_points)
                ax1.plot(vertices, times, marker='o', label=f'{density:.1f}% density')

        ax1.set_xlabel('Number of Vertices', fontsize=12)
        ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax1.set_title('Exhaustive Search - Execution Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # ax1.set_yscale('log')

        # Plot greedy algorithm
        for density in sorted(densities.keys()):
            results_at_density = sorted(densities[density], key=lambda r: r.n_vertices)
            # Filter out None values
            data_points = [(r.n_vertices, r.greedy_time_seconds) 
                          for r in results_at_density 
                          if r.greedy_time_seconds is not None]
            if data_points:
                vertices, times = zip(*data_points)
                ax2.plot(vertices, times, marker='s', label=f'{density:.1f}% density')

        ax2.set_xlabel('Number of Vertices', fontsize=12)
        ax2.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax2.set_title('Greedy Heuristic - Execution Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # ax2.set_yscale('log')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
            data_points = [(r.n_vertices, r.exact_operations) 
                          for r in results_at_density 
                          if r.exact_operations is not None]
            if data_points:
                vertices, operations = zip(*data_points)
                ax.plot(vertices, operations, marker='o', label=f'{density:.1f}% density')

        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_ylabel('Basic Operations (adjacency checks)', fontsize=12)
        ax.set_title('Exhaustive Search - Operations Count', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # ax.set_yscale('log')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
        ax.plot(vertices, configs, marker='o', linewidth=2, markersize=8,
                label='Configurations tested', color='darkblue')

        # Plot theoretical 2^n line for reference
        import numpy as np
        v_range = np.array(vertices)
        theoretical = 2 ** v_range
        ax.plot(v_range, theoretical, '--', linewidth=1.5,
                label='Theoretical $2^n$', color='red', alpha=0.7)

        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_ylabel('Number of Configurations', fontsize=12)
        ax.set_title('Exhaustive Search - Configurations Tested',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # ax.set_yscale('log')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
            data_points = [(r.n_vertices, r.precision_percent) 
                          for r in results_at_density 
                          if r.precision_percent is not None]
            if data_points:
                vertices, precision = zip(*data_points)
                ax.plot(vertices, precision, marker='o', label=f'{density:.1f}% density')

        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_ylabel('Precision (%)', fontsize=12)
        ax.set_title('Greedy Heuristic Precision', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim((0.0, 105.0))
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Precision plot saved to {output_path}")

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

        print(f"\n{'='*60}")
        print("GENERATING PLOTS")
        print(f"{'='*60}\n")

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

        print(f"\n✓ All plots saved to {output_dir}")


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
        G, clique,
        title="Sample Graph with Clique",
        output_path=output_dir / "sample_graph.png",
        show=False
    )

    print("Sample visualization created!")


if __name__ == "__main__":
    main()

