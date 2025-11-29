"""
Graph loader utilities for benchmark instances.

Supports loading graphs from various formats (GraphML, DIMACS, Sedgewick & Wayne TXT, etc.)
and handling different weight schemes.
"""

import random
import re
from pathlib import Path
from typing import Iterator, Optional

import networkx as nx


class BenchmarkGraphLoader:
    """Load benchmark graphs from various formats."""

    def __init__(self, default_weight_range: tuple[float, float] = (1.0, 100.0)):
        """
        Initialize the graph loader.

        Args:
            default_weight_range: Default weight range (min, max) for vertices without weights
        """
        self.default_weight_range = default_weight_range

    def load_graphml(self, filepath: Path) -> nx.Graph:
        """
        Load a graph from GraphML format.

        Args:
            filepath: Path to GraphML file

        Returns:
            NetworkX graph with weights
        """
        graph = nx.read_graphml(filepath)
        graph = nx.convert_node_labels_to_integers(graph)

        # Ensure weights exist
        for node in graph.nodes():
            if "weight" not in graph.nodes[node]:
                # Assign random weight if not present
                weight = random.uniform(
                    self.default_weight_range[0], self.default_weight_range[1]
                )
                graph.nodes[node]["weight"] = weight
            else:
                # Convert to float if needed
                graph.nodes[node]["weight"] = float(graph.nodes[node]["weight"])

        return graph

    def load_dimacs(self, filepath: Path) -> nx.Graph:
        """
        Load a graph from DIMACS format.

        DIMACS format for maximum clique problems:
        - Lines starting with 'c' are comments
        - Line starting with 'p edge' contains problem info: p edge n_vertices n_edges
        - Lines starting with 'e' contain edges: e v1 v2
        - Weights may be in separate lines or comments

        Args:
            filepath: Path to DIMACS file

        Returns:
            NetworkX graph with weights
        """
        graph = nx.Graph()
        weights: dict[int, float] = {}
        n_vertices = 0
        n_edges = 0

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("c"):
                    # Comment line - check for weight information
                    if "weight" in line.lower():
                        # Try to parse weight from comment (format may vary)
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit() and i + 1 < len(parts):
                                try:
                                    vertex = int(part)
                                    weight = float(parts[i + 1])
                                    weights[vertex] = weight
                                except (ValueError, IndexError):
                                    pass
                    continue

                if line.startswith("p"):
                    # Problem line: p edge n_vertices n_edges
                    parts = line.split()
                    if len(parts) >= 4:
                        n_vertices = int(parts[2])
                        n_edges = int(parts[3])
                        # Initialize graph with vertices
                        for v in range(1, n_vertices + 1):
                            graph.add_node(v - 1)  # Convert to 0-indexed

                elif line.startswith("e"):
                    # Edge line: e v1 v2
                    parts = line.split()
                    if len(parts) >= 3:
                        v1 = int(parts[1]) - 1  # Convert to 0-indexed
                        v2 = int(parts[2]) - 1
                        graph.add_edge(v1, v2)

        # Assign weights
        for node in graph.nodes():
            if node + 1 in weights:  # Convert back to 1-indexed for lookup
                graph.nodes[node]["weight"] = weights[node + 1]
            else:
                # Assign random weight if not provided
                graph.nodes[node]["weight"] = random.uniform(
                    self.default_weight_range[0], self.default_weight_range[1]
                )

        return graph

    def load_sw_txt(self, filepath: Path) -> nx.Graph:
        """
        Load a graph from Sedgewick & Wayne TXT format.

        Format specification:
        - Line 1: 0 or 1 (is directed? - we treat all as undirected for MWC)
        - Line 2: 0 or 1 (has edge weights?)
        - Line 3: Number of vertices
        - Line 4: Number of edges
        - Following lines: vertex_from vertex_to [weight]

        Note: Self-loops (vi == vj) are skipped.
        Note: Edge weights are used to assign vertex weights (sum of incident edge weights).
        Note: Digraphs are treated as undirected for MWC problem.

        Args:
            filepath: Path to SW TXT file

        Returns:
            NetworkX graph with vertex weights
        """
        graph = nx.Graph()

        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 4:
            raise ValueError(f"Invalid SW format file: {filepath} - insufficient lines")

        # Parse header
        try:
            is_directed = int(lines[0]) == 1
            has_edge_weights = int(lines[1]) == 1
            n_vertices = int(lines[2])
            n_edges = int(lines[3])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid SW format header in {filepath}") from e

        # Initialize vertices with zero weight (will accumulate from edges or assign random)
        for v in range(n_vertices):
            graph.add_node(v, weight=0.0)

        # Track edge weights per vertex to compute vertex weights
        vertex_edge_weights: dict[int, float] = {v: 0.0 for v in range(n_vertices)}

        # Parse edges
        edge_lines = lines[4:]
        edges_added = 0

        for line in edge_lines:
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                v1 = int(parts[0])
                v2 = int(parts[1])

                # Skip self-loops
                if v1 == v2:
                    continue

                # Parse edge weight if present
                edge_weight = 1.0
                if has_edge_weights and len(parts) >= 3:
                    edge_weight = float(parts[2])

                # Add edge (undirected)
                if not graph.has_edge(v1, v2):
                    graph.add_edge(v1, v2)
                    edges_added += 1

                # Accumulate edge weights for vertex weight calculation
                vertex_edge_weights[v1] += abs(edge_weight)
                vertex_edge_weights[v2] += abs(edge_weight)

            except (ValueError, IndexError):
                continue

        # Assign vertex weights
        # Strategy: Use accumulated edge weights, or assign random if no edges
        for node in graph.nodes():
            accumulated_weight = vertex_edge_weights.get(node, 0.0)
            if accumulated_weight > 0:
                # Normalize to reasonable range and add base weight
                graph.nodes[node]["weight"] = (
                    accumulated_weight * 10.0 + random.uniform(1.0, 10.0)
                )
            else:
                # Isolated vertex - assign random weight
                graph.nodes[node]["weight"] = random.uniform(
                    self.default_weight_range[0], self.default_weight_range[1]
                )

        return graph

    def load_graph(self, filepath: Path) -> nx.Graph:
        """
        Load a graph from file, auto-detecting format.

        Args:
            filepath: Path to graph file

        Returns:
            NetworkX graph with weights

        Raises:
            ValueError: If file format is not supported
        """
        suffix = filepath.suffix.lower()

        if suffix == ".graphml":
            return self.load_graphml(filepath)
        elif suffix in [".clq", ".dimacs", ".col"]:
            return self.load_dimacs(filepath)
        elif suffix == ".txt":
            # Try SW format for .txt files
            return self.load_sw_txt(filepath)
        else:
            # Try different formats in order
            try:
                return self.load_graphml(filepath)
            except Exception:
                try:
                    return self.load_dimacs(filepath)
                except Exception:
                    try:
                        return self.load_sw_txt(filepath)
                    except Exception as e:
                        raise ValueError(
                            f"Could not load graph from {filepath}. "
                            f"Supported formats: .graphml, .clq, .dimacs, .col, .txt (SW format)"
                        ) from e

    def list_graph_files(
        self,
        directory: Path,
        pattern: str = "*",
        recursive: bool = False,
        min_vertices: Optional[int] = None,
        max_vertices: Optional[int] = None,
    ) -> list[Path]:
        """
        List graph files in a directory without loading them.
        Can filter by vertex count based on filename pattern.

        Args:
            directory: Directory containing graph files
            pattern: Glob pattern for file matching (default: "*")
            recursive: Whether to search recursively
            min_vertices: Minimum vertex count (extracted from filename like graph_n{count}_d*.graphml)
            max_vertices: Maximum vertex count (extracted from filename)

        Returns:
            List of file paths matching the criteria
        """
        if not directory.exists():
            return []

        # Get all graph files
        if recursive:
            all_files = list(directory.rglob(pattern))
        else:
            all_files = list(directory.glob(pattern))

        # Filter for common graph file extensions
        extensions = {".graphml", ".clq", ".dimacs", ".col", ".txt"}
        graph_files = [
            f for f in all_files if f.is_file() and f.suffix.lower() in extensions
        ]

        # Filter by vertex count if specified (parse from filename)
        if min_vertices is not None or max_vertices is not None:
            filtered_files = []
            for filepath in graph_files:
                # Try to extract vertex count from filename (format: graph_n{count}_d*.ext)
                name = filepath.stem
                match = re.search(r"_n(\d+)_", name)
                if match:
                    vertex_count = int(match.group(1))
                    if min_vertices is not None and vertex_count < min_vertices:
                        continue
                    if max_vertices is not None and vertex_count > max_vertices:
                        continue
                    filtered_files.append(filepath)
                else:
                    # If filename doesn't match pattern, include it (don't filter)
                    filtered_files.append(filepath)
            graph_files = filtered_files

        return sorted(graph_files)

    def load_graphs_lazy(
        self, graph_files: list[Path]
    ) -> Iterator[tuple[Path, nx.Graph]]:
        """
        Lazy loader for graphs. Yields graphs one at a time.

        Args:
            graph_files: List of graph file paths

        Yields:
            Tuples of (filepath, graph)
        """
        for filepath in graph_files:
            try:
                graph = self.load_graph(filepath)
                yield (filepath, graph)
            except Exception as e:
                # Skip files that can't be loaded
                continue

    def load_directory(
        self, directory: Path, pattern: str = "*", recursive: bool = False
    ) -> list[tuple[Path, nx.Graph]]:
        """
        Load all graphs from a directory.

        WARNING: This loads all graphs into memory at once.
        For large directories, use list_graph_files() and load_graphs_lazy() instead.

        Args:
            directory: Directory containing graph files
            pattern: Glob pattern for file matching (default: "*")
            recursive: Whether to search recursively

        Returns:
            List of tuples (filepath, graph)
        """
        graph_files = self.list_graph_files(directory, pattern, recursive)
        graphs: list[tuple[Path, nx.Graph]] = []

        for filepath in graph_files:
            try:
                graph = self.load_graph(filepath)
                graphs.append((filepath, graph))
            except Exception as e:
                # Skip files that can't be loaded
                continue

        return graphs

    def convert_to_graphml(
        self, input_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """
        Convert a graph to GraphML format.

        Args:
            input_path: Path to input graph file
            output_path: Path for output GraphML file (None to auto-generate)

        Returns:
            Path to output GraphML file
        """
        graph = self.load_graph(input_path)

        if output_path is None:
            output_path = input_path.with_suffix(".graphml")

        nx.write_graphml(graph, output_path)
        return output_path


def main() -> None:
    """Test the graph loader."""
    loader = BenchmarkGraphLoader()

    # Test loading from a directory
    test_dir = Path("experiments/graphs")
    if test_dir.exists():
        graphs = loader.load_directory(test_dir, pattern="*.graphml")
        print(f"Loaded {len(graphs)} graphs from {test_dir}")
        for filepath, graph in graphs[:3]:  # Show first 3
            print(
                f"  {filepath.name}: {graph.number_of_nodes()} vertices, "
                f"{graph.number_of_edges()} edges"
            )
    else:
        print(f"Test directory {test_dir} does not exist")


if __name__ == "__main__":
    main()
