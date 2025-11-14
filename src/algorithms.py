"""
Algorithms for solving the Maximum Weight Clique Problem.

Implements exhaustive search and greedy heuristic approaches.
"""

from dataclasses import dataclass
from itertools import combinations

import networkx as nx


@dataclass
class AlgorithmResult:
    """Result of a maximum weight clique algorithm."""

    clique: set[int]
    total_weight: float
    basic_operations: int
    configurations_tested: int


class MaxWeightCliqueSolver:
    """Solver for the Maximum Weight Clique Problem."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize solver with a graph.

        Args:
            graph: NetworkX graph with 'weight' attribute on vertices
        """
        self.graph = graph
        self.n_vertices = graph.number_of_nodes()

    def _is_clique(self, vertices: set[int]) -> tuple[bool, int]:
        """
        Check if a set of vertices forms a clique.

        Args:
            vertices: set of vertex indices

        Returns:
            tuple of (is_clique, number_of_checks)
        """
        checks = 0
        vertices_list = list(vertices)
        
        # Check all pairs of vertices for adjacency
        for i in range(len(vertices_list)):
            for j in range(i + 1, len(vertices_list)):
                checks += 1
                if not self.graph.has_edge(vertices_list[i], vertices_list[j]):
                    return False, checks
        
        return True, checks

    def _calculate_weight(self, vertices: set[int]) -> float:
        """Calculate total weight of a set of vertices."""
        return sum(self.graph.nodes[v]["weight"] for v in vertices)

    def exhaustive_search(self) -> AlgorithmResult:
        """
        Find maximum weight clique using exhaustive search.

        Tests all possible subsets of vertices and returns the clique
        with maximum total weight.

        Returns:
            AlgorithmResult with the maximum weight clique found
        """
        nodes = list(self.graph.nodes())
        max_clique: set[int] = set()
        max_weight = 0.0
        total_operations = 0
        configurations_tested = 0

        # Test all possible subsets (including empty set)
        for size in range(self.n_vertices + 1):
            for subset in combinations(nodes, size):
                configurations_tested += 1
                subset_set = set(subset)
                
                # Check if subset is a clique
                is_clique, ops = self._is_clique(subset_set)
                total_operations += ops
                
                if is_clique:
                    weight = self._calculate_weight(subset_set)
                    if weight > max_weight:
                        max_weight = weight
                        max_clique = subset_set

        return AlgorithmResult(
            clique=max_clique,
            total_weight=max_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
        )

    def greedy_heuristic(self, try_all_starts: bool = True) -> AlgorithmResult:
        """
        Find maximum weight clique using greedy heuristic.

        Strategy: Start with the highest weight vertex (or try all starting vertices),
        then iteratively add the highest weight compatible vertex that maintains
        the clique property.

        Args:
            try_all_starts: If True, try starting from each vertex and keep best result

        Returns:
            AlgorithmResult with the greedy solution
        """
        if try_all_starts:
            return self._greedy_multi_start()
        else:
            return self._greedy_single_start(None)

    def _greedy_single_start(self, start_vertex: int | None) -> AlgorithmResult:
        """
        Greedy algorithm starting from a specific vertex.

        Args:
            start_vertex: Starting vertex (None for highest weight vertex)

        Returns:
            AlgorithmResult with the greedy solution
        """
        nodes = list(self.graph.nodes())
        total_operations = 0
        configurations_tested = 0

        # If no start vertex specified, choose the one with highest weight
        if start_vertex is None:
            start_vertex = max(nodes, key=lambda v: self.graph.nodes[v]["weight"])

        # Initialize clique with start vertex
        clique: set[int] = {start_vertex}
        configurations_tested += 1

        # Get candidates: all other vertices
        candidates = set(nodes) - clique

        # Iteratively add vertices
        while candidates:
            # Find compatible candidates (adjacent to all vertices in current clique)
            compatible: list[tuple[int, float]] = []
            
            for candidate in candidates:
                is_compatible = True
                for clique_vertex in clique:
                    total_operations += 1
                    if not self.graph.has_edge(candidate, clique_vertex):
                        is_compatible = False
                        break
                
                if is_compatible:
                    weight = self.graph.nodes[candidate]["weight"]
                    compatible.append((candidate, weight))

            # If no compatible candidates, stop
            if not compatible:
                break

            # Add the highest weight compatible candidate
            best_candidate = max(compatible, key=lambda x: x[1])[0]
            clique.add(best_candidate)
            candidates.remove(best_candidate)
            configurations_tested += 1

        total_weight = self._calculate_weight(clique)

        return AlgorithmResult(
            clique=clique,
            total_weight=total_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
        )

    def _greedy_multi_start(self) -> AlgorithmResult:
        """
        Try greedy algorithm starting from each vertex, keep best result.

        Returns:
            AlgorithmResult with the best greedy solution found
        """
        nodes = list(self.graph.nodes())
        best_result: AlgorithmResult | None = None
        total_operations = 0
        total_configurations = 0

        for start_vertex in nodes:
            result = self._greedy_single_start(start_vertex)
            total_operations += result.basic_operations
            total_configurations += result.configurations_tested

            if best_result is None or result.total_weight > best_result.total_weight:
                best_result = result

        # Update with accumulated statistics
        if best_result:
            best_result.basic_operations = total_operations
            best_result.configurations_tested = total_configurations

        return best_result if best_result else AlgorithmResult(
            clique=set(), total_weight=0.0, basic_operations=0, configurations_tested=0
        )


def compare_solutions(
    exact: AlgorithmResult, heuristic: AlgorithmResult
) -> dict[str, float]:
    """
    Compare exact and heuristic solutions.

    Args:
        exact: Result from exhaustive search
        heuristic: Result from greedy heuristic

    Returns:
        Dictionary with comparison metrics
    """
    precision = (
        (heuristic.total_weight / exact.total_weight * 100.0)
        if exact.total_weight > 0
        else 100.0
    )

    return {
        "exact_weight": exact.total_weight,
        "heuristic_weight": heuristic.total_weight,
        "precision_percent": precision,
        "exact_operations": exact.basic_operations,
        "heuristic_operations": heuristic.basic_operations,
        "exact_configs": exact.configurations_tested,
        "heuristic_configs": heuristic.configurations_tested,
    }


def main() -> None:
    """Test the algorithms on a small example."""
    # Create a simple test graph
    G: nx.Graph = nx.Graph()
    G.add_node(0, weight=10.0)
    G.add_node(1, weight=20.0)
    G.add_node(2, weight=15.0)
    G.add_node(3, weight=5.0)
    
    # Add edges to form a clique of {0, 1, 2}
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3)])

    print("Test Graph:")
    print(f"  Vertices: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Vertex weights: {[G.nodes[v]['weight'] for v in G.nodes()]}")

    solver = MaxWeightCliqueSolver(G)

    # Run exhaustive search
    print("\n" + "="*60)
    print("EXHAUSTIVE SEARCH")
    print("="*60)
    exact_result = solver.exhaustive_search()
    print(f"Clique found: {exact_result.clique}")
    print(f"Total weight: {exact_result.total_weight:.2f}")
    print(f"Basic operations: {exact_result.basic_operations}")
    print(f"Configurations tested: {exact_result.configurations_tested}")

    # Run greedy heuristic
    print("\n" + "="*60)
    print("GREEDY HEURISTIC")
    print("="*60)
    greedy_result = solver.greedy_heuristic()
    print(f"Clique found: {greedy_result.clique}")
    print(f"Total weight: {greedy_result.total_weight:.2f}")
    print(f"Basic operations: {greedy_result.basic_operations}")
    print(f"Configurations tested: {greedy_result.configurations_tested}")

    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    comparison = compare_solutions(exact_result, greedy_result)
    for key, value in comparison.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

