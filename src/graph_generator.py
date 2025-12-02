"""
Graph Generator for Maximum Weight Clique Problem.

Generates random graphs with 2D point vertices and configurable edge density.
"""

import random
from pathlib import Path

import networkx as nx


class GraphGenerator:
    """Generate random graphs with 2D point vertices for computational experiments."""

    def __init__(
        self,
        seed: int = 112974,
        min_distance: float = 10.0,
        coord_min: int = 1,
        coord_max: int = 500,
        min_weight: float = 1.0,
        max_weight: float = 100.0,
    ):
        """
        Initialize the graph generator.

        Args:
            seed: Random seed for reproducibility
            min_distance: Minimum distance between vertices
            coord_min: Minimum coordinate value
            coord_max: Maximum coordinate value
            min_weight: Minimum vertex weight
            max_weight: Maximum vertex weight
        """
        self.seed = seed
        self.min_distance = min_distance
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.min_weight = min_weight
        self.max_weight = max_weight
        random.seed(seed)

    def _generate_points(self, n: int) -> list[tuple[int, int]]:
        """
        Generate n 2D points with integer coordinates that are not too close.

        Args:
            n: Number of points to generate

        Returns:
            list of (x, y) coordinate tuples
        """
        points: list[tuple[int, int]] = []
        max_attempts = 10000

        for _ in range(n):
            attempts = 0
            while attempts < max_attempts:
                x = random.randint(self.coord_min, self.coord_max)
                y = random.randint(self.coord_min, self.coord_max)
                point = (x, y)

                if all(
                    self._distance(point, existing) >= self.min_distance
                    for existing in points
                ):
                    points.append(point)
                    break

                attempts += 1

            if attempts >= max_attempts:
                raise ValueError(
                    f"Could not generate {n} well-separated points. "
                    f"Try reducing min_distance or increasing coordinate range."
                )

        return points

    @staticmethod
    def _distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def generate_graph(self, n_vertices: int, edge_density_percent: float) -> nx.Graph:
        """
        Generate a random graph with specified number of vertices and edge density.

        Args:
            n_vertices: Number of vertices in the graph
            edge_density_percent: Percentage of maximum possible edges (0-100)

        Returns:
            NetworkX Graph with weighted vertices and 2D coordinates
        """

        points = self._generate_points(n_vertices)

        G: nx.Graph = nx.Graph()

        for i, (x, y) in enumerate(points):
            weight = random.uniform(self.min_weight, self.max_weight)
            G.add_node(i, x=x, y=y, weight=weight)

        max_edges = n_vertices * (n_vertices - 1) // 2
        n_edges = int(max_edges * edge_density_percent / 100)

        possible_edges = [
            (i, j) for i in range(n_vertices) for j in range(i + 1, n_vertices)
        ]

        selected_edges = random.sample(
            possible_edges, min(n_edges, len(possible_edges))
        )

        G.add_edges_from(selected_edges)

        return G

    def generate_graph_series(
        self,
        min_vertices: int = 4,
        max_vertices: int = 12,
        densities: list[float] = [12.5, 25.0, 50.0, 75.0],
        output_dir: Path = Path("experiments/graphs"),
    ) -> list[tuple[int, float, Path]]:
        """
        Generate a series of graphs with increasing size and varying densities.

        Args:
            min_vertices: Minimum number of vertices
            max_vertices: Maximum number of vertices
            densities: list of edge density percentages
            output_dir: Directory to save generated graphs

        Returns:
            list of tuples (n_vertices, density, file_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files: list[tuple[int, float, Path]] = []

        for n in range(min_vertices, max_vertices + 1):
            for density in densities:
                random.seed(self.seed + n * 1000 + int(density * 10))

                G = self.generate_graph(n, density)

                filename = f"graph_n{n}_d{int(density)}.graphml"
                filepath = output_dir / filename
                nx.write_graphml(G, filepath)

                generated_files.append((n, density, filepath))
                print(
                    f"Generated: {filename} ({n} vertices, {G.number_of_edges()} edges, {density}% density)"
                )

        return generated_files


def main() -> None:
    """Generate sample graphs for testing."""
    generator = GraphGenerator(seed=42)

    G = generator.generate_graph(n_vertices=6, edge_density_percent=50.0)

    print(f"Test graph: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
    print("\nVertex data:")
    for node, data in G.nodes(data=True):
        print(
            f"  Node {node}: pos=({data['x']}, {data['y']}), weight={data['weight']:.2f}"
        )

    print("\n" + "=" * 60)
    print("Generating full graph series...")
    print("=" * 60 + "\n")

    files = generator.generate_graph_series(min_vertices=4, max_vertices=10)

    print(f"\n✓ Generated {len(files)} graphs in experiments/graphs/")


if __name__ == "__main__":
    main()
