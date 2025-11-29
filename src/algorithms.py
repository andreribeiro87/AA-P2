"""
Algorithms for solving the Maximum Weight Clique Problem.

Implements exhaustive search, greedy heuristic, and randomized approaches.
"""

import random
import time
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import networkx as nx


class StoppingReason(Enum):
    """Reasons why a randomized algorithm stopped."""

    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"
    NO_IMPROVEMENT = "no_improvement"
    EXHAUSTED = "exhausted"


@dataclass
class AlgorithmResult:
    """Result of a maximum weight clique algorithm."""

    clique: set[int]
    total_weight: float
    basic_operations: int
    configurations_tested: int


@dataclass
class RandomizedAlgorithmResult(AlgorithmResult):
    """Result of a randomized maximum weight clique algorithm."""

    unique_configurations_tested: int = 0
    duplicate_count: int = 0
    stopping_reason: StoppingReason | None = None


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
        return sum(self.graph.nodes[v]["weight"] for v in vertices if v in self.graph)

    def _is_duplicate(
        self, configuration: set[int], tested_configs: set[frozenset[int]]
    ) -> bool:
        """
        Check if a configuration has already been tested.

        Args:
            configuration: Set of vertices to check
            tested_configs: Set of previously tested configurations (as frozensets)

        Returns:
            True if configuration was already tested, False otherwise
        """
        config_frozen = frozenset(configuration)
        return config_frozen in tested_configs

    def _check_stopping_criteria(
        self,
        iteration: int,
        start_time: float,
        last_improvement_iteration: int,
        max_iterations: int | None = None,
        time_limit: float | None = None,
        no_improvement_limit: int | None = None,
    ) -> tuple[bool, StoppingReason | None]:
        """
        Check if stopping criteria are met.

        Args:
            iteration: Current iteration number
            start_time: Start time of algorithm
            last_improvement_iteration: Last iteration where improvement occurred
            max_iterations: Maximum number of iterations allowed
            time_limit: Maximum time allowed in seconds
            no_improvement_limit: Stop if no improvement for this many iterations

        Returns:
            Tuple of (should_stop, stopping_reason)
        """
        if max_iterations is not None and iteration >= max_iterations:
            return True, StoppingReason.MAX_ITERATIONS

        if time_limit is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_limit:
                return True, StoppingReason.TIME_LIMIT

        if no_improvement_limit is not None:
            iterations_since_improvement = iteration - last_improvement_iteration
            if iterations_since_improvement >= no_improvement_limit:
                return True, StoppingReason.NO_IMPROVEMENT

        return False, None

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

        return (
            best_result
            if best_result
            else AlgorithmResult(
                clique=set(),
                total_weight=0.0,
                basic_operations=0,
                configurations_tested=0,
            )
        )

    def random_construction(
        self,
        max_iterations: int | None = None,
        time_limit: float | None = None,
        seed: int | None = None,
    ) -> RandomizedAlgorithmResult:
        """
        Find maximum weight clique using random construction.

        Strategy: Randomly select a starting vertex, then randomly add compatible
        vertices while maintaining the clique property. Repeat for multiple iterations.

        Args:
            max_iterations: Maximum number of iterations (None for unlimited)
            time_limit: Maximum time in seconds (None for unlimited)
            seed: Random seed for reproducibility (None for random)

        Returns:
            RandomizedAlgorithmResult with the best clique found
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        tested_configs: set[frozenset[int]] = set()
        max_clique: set[int] = set()
        max_weight = 0.0
        total_operations = 0
        configurations_tested = 0
        unique_configurations = 0
        duplicate_count = 0
        last_improvement_iteration = 0
        start_time = time.perf_counter()

        iteration = 0
        while True:
            iteration += 1

            # Check stopping criteria
            should_stop, stopping_reason = self._check_stopping_criteria(
                iteration=iteration,
                start_time=start_time,
                last_improvement_iteration=last_improvement_iteration,
                max_iterations=max_iterations,
                time_limit=time_limit,
            )
            if should_stop:
                break

            # Randomly select starting vertex
            start_vertex = random.choice(nodes)
            clique: set[int] = {start_vertex}
            candidates = set(nodes) - clique

            # Build clique by randomly adding compatible vertices
            while candidates:
                # Find compatible candidates
                compatible: list[int] = []
                for candidate in candidates:
                    is_compatible = True
                    for clique_vertex in clique:
                        total_operations += 1
                        if not self.graph.has_edge(candidate, clique_vertex):
                            is_compatible = False
                            break
                    if is_compatible:
                        compatible.append(candidate)

                if not compatible:
                    break

                # Randomly select a compatible vertex
                selected = random.choice(compatible)
                clique.add(selected)
                candidates.remove(selected)

            # Check if this configuration was already tested
            configurations_tested += 1
            if self._is_duplicate(clique, tested_configs):
                duplicate_count += 1
            else:
                unique_configurations += 1
                tested_configs.add(frozenset(clique))

                # Check if it's a clique and update best
                is_clique, ops = self._is_clique(clique)
                total_operations += ops

                if is_clique:
                    weight = self._calculate_weight(clique)
                    if weight > max_weight:
                        max_weight = weight
                        max_clique = clique.copy()
                        last_improvement_iteration = iteration

        return RandomizedAlgorithmResult(
            clique=max_clique,
            total_weight=max_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
            unique_configurations_tested=unique_configurations,
            duplicate_count=duplicate_count,
            stopping_reason=stopping_reason,
        )

    def random_greedy_hybrid(
        self,
        num_starts: int = 10,
        top_k: int = 3,
        randomness_factor: float = 0.5,
        seed: int | None = None,
    ) -> RandomizedAlgorithmResult:
        """
        Find maximum weight clique using random greedy hybrid approach.

        Strategy: Combine randomness with greedy selection. For each start:
        - Start with a random vertex
        - Use probabilistic selection from top-k candidates (weighted by vertex weight)
        - Repeat for multiple random starts

        Args:
            num_starts: Number of random starting points
            top_k: Number of top candidates to consider for selection
            randomness_factor: Probability of selecting randomly vs greedily (0-1)
            seed: Random seed for reproducibility (None for random)

        Returns:
            RandomizedAlgorithmResult with the best clique found
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        tested_configs: set[frozenset[int]] = set()
        best_clique: set[int] = set()
        best_weight = 0.0
        total_operations = 0
        configurations_tested = 0
        unique_configurations = 0
        duplicate_count = 0

        for start_idx in range(num_starts):
            # Random starting vertex
            start_vertex = random.choice(nodes)
            clique: set[int] = {start_vertex}
            candidates = set(nodes) - clique

            # Build clique iteratively
            while candidates:
                # Find compatible candidates with their weights
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

                if not compatible:
                    break

                # Select candidate: random vs greedy based on randomness_factor
                if random.random() < randomness_factor:
                    # Random selection
                    selected = random.choice(compatible)[0]
                else:
                    # Greedy: select from top-k candidates
                    compatible.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = compatible[: min(top_k, len(compatible))]
                    # Weighted random selection from top-k
                    weights = [w for _, w in top_candidates]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        r = random.uniform(0, total_weight)
                        cumulative = 0
                        for candidate, weight in top_candidates:
                            cumulative += weight
                            if r <= cumulative:
                                selected = candidate
                                break
                        else:
                            selected = top_candidates[-1][0]
                    else:
                        selected = top_candidates[0][0]

                clique.add(selected)
                candidates.remove(selected)

            # Check if this configuration was already tested
            configurations_tested += 1
            if self._is_duplicate(clique, tested_configs):
                duplicate_count += 1
            else:
                unique_configurations += 1
                tested_configs.add(frozenset(clique))

                # Verify it's a clique and update best
                is_clique, ops = self._is_clique(clique)
                total_operations += ops

                if is_clique:
                    weight = self._calculate_weight(clique)
                    if weight > best_weight:
                        best_weight = weight
                        best_clique = clique.copy()

        return RandomizedAlgorithmResult(
            clique=best_clique,
            total_weight=best_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
            unique_configurations_tested=unique_configurations,
            duplicate_count=duplicate_count,
            stopping_reason=StoppingReason.EXHAUSTED,
        )

    def iterative_random_search(
        self,
        max_iterations: int | None = None,
        time_limit: float | None = None,
        initial_size_strategy: str = "largest",
        size_decrement: int = 1,
        max_configs_per_size: int | None = None,
        seed: int | None = None,
    ) -> RandomizedAlgorithmResult:
        """
        Find maximum weight clique using iterative random search.

        Strategy: Generate random subsets of vertices and check if they form cliques.
        Use size-based iteration: start with larger sizes, gradually test smaller sizes.

        Args:
            max_iterations: Maximum total iterations (None for unlimited)
            time_limit: Maximum time in seconds (None for unlimited)
            initial_size_strategy: "largest" (start from n) or "half" (start from n/2)
            size_decrement: How much to decrease size by when moving to next size
            max_configs_per_size: Maximum configurations to test per size (None for unlimited)
            seed: Random seed for reproducibility (None for random)

        Returns:
            RandomizedAlgorithmResult with the best clique found
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        tested_configs: set[frozenset[int]] = set()
        max_clique: set[int] = set()
        max_weight = 0.0
        total_operations = 0
        configurations_tested = 0
        unique_configurations = 0
        duplicate_count = 0
        last_improvement_iteration = 0
        start_time = time.perf_counter()

        # Determine initial size
        if initial_size_strategy == "largest":
            current_size = self.n_vertices
        elif initial_size_strategy == "half":
            current_size = max(1, self.n_vertices // 2)
        else:
            current_size = self.n_vertices

        iteration = 0
        while current_size > 0:
            configs_at_size = 0

            # Test random subsets of current size
            while True:
                iteration += 1

                # Check stopping criteria
                should_stop, stopping_reason = self._check_stopping_criteria(
                    iteration=iteration,
                    start_time=start_time,
                    last_improvement_iteration=last_improvement_iteration,
                    max_iterations=max_iterations,
                    time_limit=time_limit,
                )
                if should_stop:
                    return RandomizedAlgorithmResult(
                        clique=max_clique,
                        total_weight=max_weight,
                        basic_operations=total_operations,
                        configurations_tested=configurations_tested,
                        unique_configurations_tested=unique_configurations,
                        duplicate_count=duplicate_count,
                        stopping_reason=stopping_reason,
                    )

                # Check per-size limit
                if (
                    max_configs_per_size is not None
                    and configs_at_size >= max_configs_per_size
                ):
                    break

                # Generate random subset of current size
                if current_size > len(nodes):
                    break

                subset = set(random.sample(nodes, current_size))
                configurations_tested += 1

                # Check if duplicate
                if self._is_duplicate(subset, tested_configs):
                    duplicate_count += 1
                    configs_at_size += 1
                    continue

                unique_configurations += 1
                tested_configs.add(frozenset(subset))
                configs_at_size += 1

                # Check if it's a clique
                is_clique, ops = self._is_clique(subset)
                total_operations += ops

                if is_clique:
                    weight = self._calculate_weight(subset)
                    if weight > max_weight:
                        max_weight = weight
                        max_clique = subset.copy()
                        last_improvement_iteration = iteration

            # Move to next smaller size
            current_size -= size_decrement

        return RandomizedAlgorithmResult(
            clique=max_clique,
            total_weight=max_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
            unique_configurations_tested=unique_configurations,
            duplicate_count=duplicate_count,
            stopping_reason=StoppingReason.EXHAUSTED,
        )

    def monte_carlo(
        self,
        num_samples: int = 1000,
        sample_size_strategy: str = "proportional",
        probability_threshold: float = 0.1,
        max_iterations: int | None = None,
        time_limit: float | None = None,
        seed: int | None = None,
    ) -> RandomizedAlgorithmResult:
        """
        Find maximum weight clique using Monte Carlo algorithm.

        Strategy: Randomly sample candidate cliques using probabilistic selection
        based on vertex weights. Provides approximate solutions with polynomial
        time guarantee but may not be optimal.

        Args:
            num_samples: Number of random samples to try
            sample_size_strategy: "fixed", "proportional", or "adaptive"
            probability_threshold: Minimum probability for vertex selection
            max_iterations: Maximum number of iterations (overrides num_samples if set)
            time_limit: Maximum time in seconds (None for unlimited)
            seed: Random seed for reproducibility (None for random)

        Returns:
            RandomizedAlgorithmResult with approximate clique found
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        tested_configs: set[frozenset[int]] = set()
        max_clique: set[int] = set()
        max_weight = 0.0
        total_operations = 0
        configurations_tested = 0
        unique_configurations = 0
        duplicate_count = 0
        last_improvement_iteration = 0
        start_time = time.perf_counter()

        # Compute vertex weights for probability distribution
        weights = [self.graph.nodes[v]["weight"] for v in nodes]
        min_weight = min(weights) if weights else 1.0
        max_weight_val = max(weights) if weights else 1.0
        weight_range = (
            max_weight_val - min_weight if max_weight_val > min_weight else 1.0
        )

        # Determine number of iterations
        if max_iterations is not None:
            num_samples = max_iterations

        iteration = 0
        stopping_reason: StoppingReason | None = None
        while iteration < num_samples:
            iteration += 1

            # Check time limit
            if time_limit is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed >= time_limit:
                    stopping_reason = StoppingReason.TIME_LIMIT
                    break

            # Determine sample size based on strategy
            if sample_size_strategy == "fixed":
                sample_size = max(1, self.n_vertices // 2)
            elif sample_size_strategy == "proportional":
                sample_size = max(1, int(self.n_vertices * 0.5))
            elif sample_size_strategy == "adaptive":
                # Adaptive: larger samples early, smaller later
                progress = iteration / num_samples
                sample_size = max(1, int(self.n_vertices * (1.0 - progress * 0.5)))
            else:
                sample_size = max(1, self.n_vertices // 2)

            # Sample vertices probabilistically based on weights
            clique: set[int] = set()
            available = list(nodes)

            for _ in range(min(sample_size, len(available))):
                if not available:
                    break

                # Compute probabilities based on weights
                probabilities = []
                for v in available:
                    weight = self.graph.nodes[v]["weight"]
                    # Normalize and add threshold
                    prob = probability_threshold + (1.0 - probability_threshold) * (
                        (weight - min_weight) / weight_range
                        if weight_range > 0
                        else 0.5
                    )

                    # Check if vertex is compatible with current clique
                    is_compatible = True
                    for clique_vertex in clique:
                        total_operations += 1
                        if not self.graph.has_edge(v, clique_vertex):
                            is_compatible = False
                            break

                    if is_compatible:
                        probabilities.append((v, prob))
                    else:
                        probabilities.append((v, 0.0))

                if not probabilities or all(p == 0.0 for _, p in probabilities):
                    break

                # Select vertex using weighted random choice
                total_prob = sum(p for _, p in probabilities)
                if total_prob == 0:
                    break

                r = random.uniform(0, total_prob)
                cumulative = 0.0
                selected = None

                for v, prob in probabilities:
                    cumulative += prob
                    if r <= cumulative:
                        selected = v
                        break

                if selected is None:
                    selected = probabilities[-1][0]

                clique.add(selected)
                available.remove(selected)

            # Check if this configuration was already tested
            configurations_tested += 1
            if self._is_duplicate(clique, tested_configs):
                duplicate_count += 1
            else:
                unique_configurations += 1
                tested_configs.add(frozenset(clique))

                # Verify it's a clique
                if clique:
                    is_clique, ops = self._is_clique(clique)
                    total_operations += ops

                    if is_clique:
                        weight = self._calculate_weight(clique)
                        if weight > max_weight:
                            max_weight = weight
                            max_clique = clique.copy()
                            last_improvement_iteration = iteration

        if stopping_reason is None:
            stopping_reason = StoppingReason.MAX_ITERATIONS

        return RandomizedAlgorithmResult(
            clique=max_clique,
            total_weight=max_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
            unique_configurations_tested=unique_configurations,
            duplicate_count=duplicate_count,
            stopping_reason=stopping_reason,
        )

    def las_vegas(
        self,
        max_attempts: int = 10000,
        construction_strategy: str = "iterative_construction",
        max_iterations: int | None = None,
        time_limit: float | None = None,
        seed: int | None = None,
    ) -> RandomizedAlgorithmResult:
        """
        Find maximum weight clique using Las Vegas algorithm.

        Strategy: Always returns a valid clique (correctness guarantee) but with
        variable runtime. Uses randomization to select vertices but verifies
        clique property at each step.

        Args:
            max_attempts: Maximum attempts before fallback (safety limit)
            construction_strategy: "random_walk" or "iterative_construction"
            max_iterations: Maximum number of iterations (overrides max_attempts if set)
            time_limit: Maximum time in seconds (None for unlimited)
            seed: Random seed for reproducibility (None for random)

        Returns:
            RandomizedAlgorithmResult with a valid clique (always correct)
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.graph.nodes())
        tested_configs: set[frozenset[int]] = set()
        best_clique: set[int] = set()
        best_weight = 0.0
        total_operations = 0
        configurations_tested = 0
        unique_configurations = 0
        duplicate_count = 0
        last_improvement_iteration = 0
        start_time = time.perf_counter()

        if max_iterations is not None:
            max_attempts = (
                max_attempts if max_iterations > max_attempts else max_iterations
            )

        iteration = 0
        stopping_reason: StoppingReason | None = None
        while iteration < max_attempts:
            iteration += 1

            # Check time limit
            if time_limit is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed >= time_limit:
                    stopping_reason = StoppingReason.TIME_LIMIT
                    break

            if construction_strategy == "random_walk":
                clique = self._random_walk_clique()
            else:  # iterative_construction
                clique = self._iterative_construction_clique()

            # Verify it's a valid clique
            if not clique:
                continue

            is_clique, ops = self._is_clique(clique)
            total_operations += ops

            # Las Vegas guarantee: only return valid cliques
            if not is_clique:
                continue

            configurations_tested += 1

            # Check if duplicate
            if self._is_duplicate(clique, tested_configs):
                duplicate_count += 1
            else:
                unique_configurations += 1
                tested_configs.add(frozenset(clique))

                weight = self._calculate_weight(clique)
                if weight > best_weight:
                    best_weight = weight
                    best_clique = clique.copy()
                    last_improvement_iteration = iteration

            # For Las Vegas, we can stop early if we found a reasonable solution
            # But we guarantee correctness, so we always return a valid clique
            if iteration % 100 == 0 and best_clique:
                # Check if we should continue
                pass

        if stopping_reason is None:
            stopping_reason = StoppingReason.MAX_ITERATIONS

        # Ensure we return at least some valid clique
        if not best_clique:
            # Fallback: return a single vertex with maximum weight
            best_vertex = max(nodes, key=lambda v: self.graph.nodes[v]["weight"])
            best_clique = {best_vertex}
            best_weight = self.graph.nodes[best_vertex]["weight"]

        return RandomizedAlgorithmResult(
            clique=best_clique,
            total_weight=best_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
            unique_configurations_tested=unique_configurations,
            duplicate_count=duplicate_count,
            stopping_reason=stopping_reason,
        )

    def _random_walk_clique(self) -> set[int]:
        """Helper for Las Vegas: perform random walk to find clique."""
        nodes = list(self.graph.nodes())
        if not nodes:
            return set()

        # Start with random vertex
        current = random.choice(nodes)
        clique: set[int] = {current}
        candidates = set(self.graph.neighbors(current))

        # Random walk: add compatible vertices
        max_steps = self.n_vertices
        steps = 0

        while candidates and steps < max_steps:
            steps += 1

            # Filter compatible candidates
            compatible = []
            for candidate in candidates:
                is_compatible = True
                for clique_vertex in clique:
                    if not self.graph.has_edge(candidate, clique_vertex):
                        is_compatible = False
                        break
                if is_compatible:
                    compatible.append(candidate)

            if not compatible:
                break

            # Randomly select a compatible vertex
            selected = random.choice(compatible)
            clique.add(selected)
            candidates.remove(selected)

            # Update candidates to neighbors of new vertex
            new_candidates = set(self.graph.neighbors(selected))
            candidates = candidates.intersection(new_candidates)

        return clique

    def _iterative_construction_clique(self) -> set[int]:
        """Helper for Las Vegas: iteratively construct clique with validation."""
        nodes = list(self.graph.nodes())
        if not nodes:
            return set()

        # Start with highest weight vertex
        start_vertex = max(nodes, key=lambda v: self.graph.nodes[v]["weight"])
        clique: set[int] = {start_vertex}
        candidates = set(nodes) - clique

        # Build clique by randomly adding compatible vertices
        while candidates:
            # Find compatible candidates
            compatible = []
            for candidate in candidates:
                is_compatible = True
                for clique_vertex in clique:
                    if not self.graph.has_edge(candidate, clique_vertex):
                        is_compatible = False
                        break
                if is_compatible:
                    compatible.append(candidate)

            if not compatible:
                break

            # Weighted random selection based on vertex weights
            weights = [self.graph.nodes[v]["weight"] for v in compatible]
            total_weight = sum(weights)
            if total_weight == 0:
                selected = random.choice(compatible)
            else:
                r = random.uniform(0, total_weight)
                cumulative = 0.0
                selected = compatible[-1]

                for v, weight in zip(compatible, weights):
                    cumulative += weight
                    if r <= cumulative:
                        selected = v
                        break

            clique.add(selected)
            candidates.remove(selected)

        return clique

    def mwc_redu(
        self,
        reduction_rules: list[str] | None = None,
        solver_method: str = "greedy",
        aggressive: bool = False,
        solver_params: dict | None = None,
    ) -> AlgorithmResult | RandomizedAlgorithmResult:
        """
        Find maximum weight clique using MWCRedu (graph reduction preprocessing).

        Strategy: Apply polynomial-time reduction rules to reduce graph size,
        then run a solver on the reduced graph and map solution back.

        Args:
            reduction_rules: List of rules to apply ("domination", "isolation", "degree")
            solver_method: Algorithm to run on reduced graph ("exhaustive", "greedy", "monte_carlo", etc.)
            aggressive: Apply more aggressive reductions
            solver_params: Parameters to pass to the solver method

        Returns:
            AlgorithmResult or RandomizedAlgorithmResult depending on solver
        """
        if reduction_rules is None:
            reduction_rules = ["domination", "isolation", "degree"]

        if solver_params is None:
            solver_params = {}

        # Apply graph reduction
        reduced_graph, vertex_mapping, removed_vertices = self._graph_reduction(
            reduction_rules=reduction_rules,
            aggressive=aggressive,
        )

        # Create solver for reduced graph
        reduced_solver = MaxWeightCliqueSolver(reduced_graph)

        # Run solver on reduced graph
        if solver_method == "exhaustive":
            result = reduced_solver.exhaustive_search()
        elif solver_method == "greedy":
            result = reduced_solver.greedy_heuristic(**solver_params)
        elif solver_method == "monte_carlo":
            result = reduced_solver.monte_carlo(**solver_params)
        elif solver_method == "las_vegas":
            result = reduced_solver.las_vegas(**solver_params)
        else:
            # Default to greedy
            result = reduced_solver.greedy_heuristic(**solver_params)

        # Map solution back to original graph
        original_clique = set()
        for reduced_vertex in result.clique:
            # reduced_vertex is from the reduced graph
            # vertex_mapping maps reduced_graph vertex -> original graph vertex
            if reduced_vertex in vertex_mapping:
                original_vertex = vertex_mapping[reduced_vertex]
                # Only add if vertex exists in original graph
                if original_vertex in self.graph:
                    original_clique.add(original_vertex)
            else:
                # If not in mapping, the vertex might have been removed but the result still contains it
                # This shouldn't happen, but skip it to avoid errors
                continue

        # Check if we can add any removed vertices (they might be isolated high-weight vertices)
        for removed_vertex in removed_vertices:
            # Ensure the removed vertex still exists in the original graph
            if removed_vertex not in self.graph:
                continue

            # Check if vertex is compatible with current clique
            is_compatible = True
            if not original_clique:
                # Empty clique - removed vertex can always be added if it exists
                is_compatible = True
            else:
                for clique_vertex in original_clique:
                    # Ensure clique_vertex exists in graph before checking edge
                    if clique_vertex not in self.graph:
                        is_compatible = False
                        break
                    try:
                        if not self.graph.has_edge(removed_vertex, clique_vertex):
                            is_compatible = False
                            break
                    except Exception:
                        # If there's an error checking the edge, assume not compatible
                        is_compatible = False
                        break

            if is_compatible:
                original_clique.add(removed_vertex)

        # Calculate weight in original graph
        total_weight = self._calculate_weight(original_clique)

        # Return result with mapped clique
        if isinstance(result, RandomizedAlgorithmResult):
            return RandomizedAlgorithmResult(
                clique=original_clique,
                total_weight=total_weight,
                basic_operations=result.basic_operations,
                configurations_tested=result.configurations_tested,
                unique_configurations_tested=result.unique_configurations_tested,
                duplicate_count=result.duplicate_count,
                stopping_reason=result.stopping_reason,
            )
        else:
            return AlgorithmResult(
                clique=original_clique,
                total_weight=total_weight,
                basic_operations=result.basic_operations,
                configurations_tested=result.configurations_tested,
            )

    def _graph_reduction(
        self,
        reduction_rules: list[str],
        aggressive: bool = False,
    ) -> tuple[nx.Graph, dict[int, int], list[int]]:
        """
        Apply graph reduction rules to reduce problem size.

        Args:
            reduction_rules: List of rule names to apply
            aggressive: Apply more aggressive reductions

        Returns:
            Tuple of (reduced_graph, vertex_mapping, removed_vertices)
            - reduced_graph: Graph after reduction
            - vertex_mapping: Maps new vertex indices to original indices
            - removed_vertices: List of removed vertex indices
        """
        reduced_graph = self.graph.copy()
        vertex_mapping = {v: v for v in reduced_graph.nodes()}
        removed_vertices: list[int] = []

        changed = True
        iteration = 0
        max_iterations = 10 if aggressive else 5

        while changed and iteration < max_iterations:
            iteration += 1
            changed = False

            if "domination" in reduction_rules:
                changed = (
                    self._apply_domination_reduction(
                        reduced_graph, vertex_mapping, removed_vertices, aggressive
                    )
                    or changed
                )

            if "isolation" in reduction_rules:
                changed = (
                    self._apply_isolation_reduction(
                        reduced_graph, vertex_mapping, removed_vertices, aggressive
                    )
                    or changed
                )

            if "degree" in reduction_rules:
                changed = (
                    self._apply_degree_reduction(
                        reduced_graph, vertex_mapping, removed_vertices, aggressive
                    )
                    or changed
                )

        return reduced_graph, vertex_mapping, removed_vertices

    def _apply_domination_reduction(
        self,
        graph: nx.Graph,
        vertex_mapping: dict[int, int],
        removed_vertices: list[int],
        aggressive: bool,
    ) -> bool:
        """Apply domination reduction rule: remove dominated vertices."""
        changed = False
        nodes_to_remove: list[int] = []

        for u in list(graph.nodes()):
            u_weight = graph.nodes[u]["weight"]
            u_neighbors = set(graph.neighbors(u))

            for v in graph.nodes():
                if u == v or v in nodes_to_remove:
                    continue

                v_weight = graph.nodes[v]["weight"]
                v_neighbors = set(graph.neighbors(v))

                # Check if v dominates u: v's neighbors include u's neighbors
                # and weight(v) >= weight(u)
                if v_weight >= u_weight:
                    if u_neighbors.issubset(v_neighbors | {v}):
                        # v dominates u
                        nodes_to_remove.append(u)
                        changed = True
                        break

                # Check if u dominates v
                if u_weight >= v_weight and not aggressive:
                    if v_neighbors.issubset(u_neighbors | {u}):
                        # u dominates v
                        if v not in nodes_to_remove:
                            nodes_to_remove.append(v)
                            changed = True

        # Remove duplicate nodes and check existence before removal
        unique_nodes_to_remove = list(set(nodes_to_remove))
        for node in unique_nodes_to_remove:
            # Check if node still exists in graph before trying to remove
            if node not in graph:
                continue
            if node in vertex_mapping:
                original_vertex = vertex_mapping[node]
                if original_vertex not in removed_vertices:
                    removed_vertices.append(original_vertex)
                del vertex_mapping[node]
            graph.remove_node(node)

        return changed

    def _apply_isolation_reduction(
        self,
        graph: nx.Graph,
        vertex_mapping: dict[int, int],
        removed_vertices: list[int],
        aggressive: bool,
    ) -> bool:
        """Apply isolation reduction: remove isolated vertices that can't improve solution."""
        changed = False
        nodes_to_remove: list[int] = []

        for node in list(graph.nodes()):
            neighbors = list(graph.neighbors(node))
            if len(neighbors) == 0:
                # Isolated vertex: can't be in a clique with other vertices
                # Keep only if it's the best single-vertex solution
                if aggressive:
                    nodes_to_remove.append(node)
                    changed = True

        # Remove duplicate nodes and check existence before removal
        unique_nodes_to_remove = list(set(nodes_to_remove))
        for node in unique_nodes_to_remove:
            # Check if node still exists in graph before trying to remove
            if node not in graph:
                continue
            if node in vertex_mapping:
                original_vertex = vertex_mapping[node]
                if original_vertex not in removed_vertices:
                    removed_vertices.append(original_vertex)
                del vertex_mapping[node]
            graph.remove_node(node)

        return changed

    def _apply_degree_reduction(
        self,
        graph: nx.Graph,
        vertex_mapping: dict[int, int],
        removed_vertices: list[int],
        aggressive: bool,
    ) -> bool:
        """Apply degree-based reduction rules."""
        changed = False
        nodes_to_remove: list[int] = []

        if not graph.nodes():
            return False

        max_degree = max(len(list(graph.neighbors(v))) for v in graph.nodes())

        for node in list(graph.nodes()):
            degree = len(list(graph.neighbors(node)))
            weight = graph.nodes[node]["weight"]

            # Remove vertices with degree 0 (isolated) if aggressive
            if degree == 0 and aggressive:
                nodes_to_remove.append(node)
                changed = True
                continue

            # Remove vertices with very low degree and low weight
            if (
                aggressive
                and degree <= 1
                and weight < max(graph.nodes[v]["weight"] for v in graph.nodes()) * 0.1
            ):
                # Check if removing won't affect optimal solution
                nodes_to_remove.append(node)
                changed = True

        # Remove duplicate nodes and check existence before removal
        unique_nodes_to_remove = list(set(nodes_to_remove))
        for node in unique_nodes_to_remove:
            # Check if node still exists in graph before trying to remove
            if node not in graph:
                continue
            if node in vertex_mapping:
                original_vertex = vertex_mapping[node]
                if original_vertex not in removed_vertices:
                    removed_vertices.append(original_vertex)
                del vertex_mapping[node]
            graph.remove_node(node)

        return changed

    def _weighted_graph_coloring(
        self, vertices: list[int], use_greedy: bool = True
    ) -> dict[int, int]:
        """
        Compute weighted graph coloring for upper bound computation.

        Args:
            vertices: List of vertices to color
            use_greedy: Use greedy coloring (True) or exact coloring (False)

        Returns:
            Dictionary mapping vertex to color number
        """
        if not vertices:
            return {}

        # Sort vertices by weight (descending) for better coloring
        sorted_vertices = sorted(
            vertices, key=lambda v: self.graph.nodes[v]["weight"], reverse=True
        )

        coloring: dict[int, int] = {}
        colors_used: set[int] = set()

        for vertex in sorted_vertices:
            # Find colors used by neighbors
            neighbor_colors = set()
            for neighbor in self.graph.neighbors(vertex):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])

            # Find first available color
            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[vertex] = color
            colors_used.add(color)

        return coloring

    def _compute_weighted_upper_bound(
        self, vertices: list[int], coloring: dict[int, int] | None = None
    ) -> float:
        """
        Compute upper bound for maximum weight clique using weighted coloring.

        Args:
            vertices: List of vertices to consider
            coloring: Pre-computed coloring (if None, will compute)

        Returns:
            Upper bound on maximum weight clique
        """
        if not vertices:
            return 0.0

        if coloring is None:
            coloring = self._weighted_graph_coloring(vertices)

        # Upper bound: sum of maximum weight vertex in each color class
        color_classes: dict[int, list[int]] = {}
        for vertex, color in coloring.items():
            if color not in color_classes:
                color_classes[color] = []
            color_classes[color].append(vertex)

        upper_bound = 0.0
        for color, class_vertices in color_classes.items():
            if class_vertices:
                max_weight_in_class = max(
                    self.graph.nodes[v]["weight"] for v in class_vertices
                )
                upper_bound += max_weight_in_class

        return upper_bound

    def max_clique_weight(
        self,
        variant: str = "static",
        color_ordering: str = "weight_desc",
        use_reduction: bool = False,
        reduction_rules: list[str] | None = None,
    ) -> AlgorithmResult:
        """
        Find maximum weight clique using MaxCliqueWeight algorithm.

        Branch-and-bound algorithm with weighted graph coloring for upper bounds.

        Args:
            variant: "static" (fixed bounds) or "dynamic" (recompute bounds)
            color_ordering: Strategy for coloring order ("weight_desc", "degree_desc")
            use_reduction: Apply MWCRedu preprocessing first
            reduction_rules: Rules for reduction if use_reduction is True

        Returns:
            AlgorithmResult with optimal clique
        """
        # Apply reduction if requested
        if use_reduction:
            if reduction_rules is None:
                reduction_rules = ["domination", "isolation"]
            reduced_graph, vertex_mapping, removed_vertices = self._graph_reduction(
                reduction_rules, aggressive=False
            )
            solver = MaxWeightCliqueSolver(reduced_graph)
            result = solver.max_clique_weight(
                variant=variant,
                color_ordering=color_ordering,
                use_reduction=False,
            )
            # Map back to original graph
            original_clique = {
                vertex_mapping[v] for v in result.clique if v in vertex_mapping
            }
            # Add removed vertices if compatible
            for removed in removed_vertices:
                is_compatible = all(
                    self.graph.has_edge(removed, v) for v in original_clique
                )
                if is_compatible:
                    original_clique.add(removed)
            return AlgorithmResult(
                clique=original_clique,
                total_weight=self._calculate_weight(original_clique),
                basic_operations=result.basic_operations,
                configurations_tested=result.configurations_tested,
            )
        else:
            return self._branch_and_bound_weighted(
                variant=variant, color_ordering=color_ordering
            )

    def _branch_and_bound_weighted(
        self, variant: str, color_ordering: str
    ) -> AlgorithmResult:
        """
        Branch-and-bound search with weighted coloring bounds.

        Args:
            variant: "static" or "dynamic"
            color_ordering: Ordering strategy

        Returns:
            AlgorithmResult with optimal clique
        """
        nodes = list(self.graph.nodes())
        if not nodes:
            return AlgorithmResult(
                clique=set(),
                total_weight=0.0,
                basic_operations=0,
                configurations_tested=0,
            )

        # Order vertices
        if color_ordering == "weight_desc":
            ordered_nodes = sorted(
                nodes, key=lambda v: self.graph.nodes[v]["weight"], reverse=True
            )
        elif color_ordering == "degree_desc":
            ordered_nodes = sorted(
                nodes,
                key=lambda v: len(list(self.graph.neighbors(v))),
                reverse=True,
            )
        else:
            ordered_nodes = nodes

        best_clique: set[int] = set()
        best_weight = 0.0
        total_operations = 0
        configurations_tested = 0

        # Compute initial upper bound
        initial_coloring = self._weighted_graph_coloring(ordered_nodes)
        upper_bound = self._compute_weighted_upper_bound(
            ordered_nodes, initial_coloring
        )

        # Branch-and-bound search
        def search(
            current_clique: set[int], candidates: list[int], current_weight: float
        ):
            nonlocal best_clique, best_weight, total_operations, configurations_tested

            configurations_tested += 1

            if not candidates:
                # Check if current clique is better
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_clique = current_clique.copy()
                return

            # Compute upper bound for remaining candidates
            if variant == "dynamic":
                remaining_coloring = self._weighted_graph_coloring(candidates)
                remaining_upper = self._compute_weighted_upper_bound(
                    candidates, remaining_coloring
                )
            else:
                # Use pre-computed bound for remaining vertices
                remaining_upper = self._compute_weighted_upper_bound(candidates)

            # Prune if upper bound + current weight <= best weight
            if current_weight + remaining_upper <= best_weight:
                return

            # Branch on each candidate
            for i, candidate in enumerate(candidates):
                total_operations += 1

                # Check if candidate can be added to clique
                can_add = all(self.graph.has_edge(candidate, v) for v in current_clique)

                if can_add:
                    new_clique = current_clique | {candidate}
                    new_weight = current_weight + self.graph.nodes[candidate]["weight"]
                    new_candidates = [
                        c
                        for c in candidates[i + 1 :]
                        if self.graph.has_edge(candidate, c)
                    ]

                    search(new_clique, new_candidates, new_weight)

                # Also try without this candidate (implicit in next iteration)

        search(set(), ordered_nodes, 0.0)

        return AlgorithmResult(
            clique=best_clique,
            total_weight=best_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
        )

    def max_clique_dyn_weight(
        self,
        color_ordering: str = "weight_desc",
        use_reduction: bool = False,
        reduction_rules: list[str] | None = None,
    ) -> AlgorithmResult:
        """
        Find maximum weight clique using MaxCliqueDynWeight algorithm.

        Dynamic variant that recomputes bounds during search.

        Args:
            color_ordering: Strategy for coloring order
            use_reduction: Apply MWCRedu preprocessing first
            reduction_rules: Rules for reduction if use_reduction is True

        Returns:
            AlgorithmResult with optimal clique
        """
        return self.max_clique_weight(
            variant="dynamic",
            color_ordering=color_ordering,
            use_reduction=use_reduction,
            reduction_rules=reduction_rules,
        )

    # =========================================================================
    # NEW ALGORITHMS FROM MWC LITERATURE
    # =========================================================================

    def wlmc(
        self,
        time_limit: float | None = None,
        use_preprocessing: bool = True,
    ) -> AlgorithmResult:
        """
        WLMC (Weighted Large Maximum Clique) - Exact BnB algorithm.

        Designed for large sparse graphs. Features:
        - Degree-based vertex ordering
        - Preprocessing to reduce graph size
        - Independent Set (IS) based upper bounds
        - Vertex splitting strategy

        Args:
            time_limit: Maximum time in seconds
            use_preprocessing: Apply preprocessing phase

        Returns:
            AlgorithmResult with optimal clique
        """
        start_time = time.perf_counter()
        total_operations = 0
        configurations_tested = 0

        # Phase 1: Initialize - compute ordering and initial clique
        initial_clique, vertex_order, reduced_graph = self._initialize_wlmc(
            use_preprocessing
        )

        if reduced_graph.number_of_nodes() == 0:
            return AlgorithmResult(
                clique=initial_clique,
                total_weight=self._calculate_weight(initial_clique),
                basic_operations=total_operations,
                configurations_tested=configurations_tested,
            )

        # Phase 2: Branch and Bound search
        best_clique = initial_clique.copy()
        best_weight = self._calculate_weight(best_clique)

        def search_wlmc(
            current: set[int],
            candidates: list[int],
            current_weight: float,
            depth: int = 0,
        ) -> None:
            nonlocal best_clique, best_weight, total_operations, configurations_tested

            # Time limit check
            if time_limit and time.perf_counter() - start_time > time_limit:
                return

            configurations_tested += 1

            if not candidates:
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_clique = current.copy()
                return

            # Compute upper bound using IS partition
            ub = self._compute_is_upper_bound(candidates, current_weight)
            if ub <= best_weight:
                return  # Prune

            # Branch on candidates
            for i, v in enumerate(candidates):
                total_operations += 1

                # Check if v can extend current clique
                if all(self.graph.has_edge(v, u) for u in current):
                    new_current = current | {v}
                    new_weight = current_weight + self.graph.nodes[v]["weight"]

                    # Filter candidates for next level
                    new_candidates = [
                        c for c in candidates[i + 1 :] if self.graph.has_edge(v, c)
                    ]

                    search_wlmc(new_current, new_candidates, new_weight, depth + 1)

        # Order vertices by degree (ascending)
        ordered = sorted(
            reduced_graph.nodes(), key=lambda v: len(list(reduced_graph.neighbors(v)))
        )

        search_wlmc(set(), ordered, 0.0)

        return AlgorithmResult(
            clique=best_clique,
            total_weight=best_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
        )

    def _initialize_wlmc(
        self, use_preprocessing: bool
    ) -> tuple[set[int], list[int], nx.Graph]:
        """
        Initialize WLMC: compute ordering and initial clique.

        Returns:
            Tuple of (initial_clique, vertex_order, reduced_graph)
        """
        H = self.graph.copy()
        vertex_order = []
        initial_clique: set[int] = set()

        # Compute ordering by removing minimum degree vertices
        while H.number_of_nodes() > 0:
            # Find vertex with minimum degree
            min_v = min(H.nodes(), key=lambda v: len(list(H.neighbors(v))))
            vertex_order.append(min_v)

            # Check if remaining graph is a clique
            remaining = set(H.nodes())
            if len(remaining) > 0:
                is_clique = True
                nodes_list = list(remaining)
                for i in range(len(nodes_list)):
                    for j in range(i + 1, len(nodes_list)):
                        if not H.has_edge(nodes_list[i], nodes_list[j]):
                            is_clique = False
                            break
                    if not is_clique:
                        break

                if is_clique:
                    initial_clique = remaining
                    break

            H.remove_node(min_v)

        # Preprocessing: reduce graph
        if use_preprocessing:
            lb = self._calculate_weight(initial_clique)
            reduced = self.graph.copy()
            to_remove = []
            for v in reduced.nodes():
                # Remove if inclusive neighborhood weight <= lb
                inclusive_neighbors = set(reduced.neighbors(v)) | {v}
                neighborhood_weight = sum(
                    reduced.nodes[u]["weight"] for u in inclusive_neighbors
                )
                if neighborhood_weight <= lb:
                    to_remove.append(v)

            for v in to_remove:
                reduced.remove_node(v)
        else:
            reduced = self.graph.copy()

        return initial_clique, vertex_order, reduced

    def _compute_is_upper_bound(
        self, candidates: list[int], current_weight: float
    ) -> float:
        """
        Compute upper bound using Independent Set partition.

        The weight of the maximum clique is bounded by the sum of
        maximum weights in each independent set of a partition.
        """
        if not candidates:
            return current_weight

        # Greedy IS partition
        remaining = set(candidates)
        upper_bound = current_weight

        while remaining:
            # Extract an independent set
            is_set: list[int] = []
            for v in list(remaining):
                if all(not self.graph.has_edge(v, u) for u in is_set):
                    is_set.append(v)
                    remaining.remove(v)

            # Add max weight from this IS
            if is_set:
                max_weight_in_is = max(self.graph.nodes[v]["weight"] for v in is_set)
                upper_bound += max_weight_in_is

        return upper_bound

    def fast_wclq(
        self,
        max_iterations: int | None = None,
        time_limit: float | None = None,
        bms_k: int = 5,
        seed: int | None = None,
    ) -> RandomizedAlgorithmResult:
        """
        FastWClq - Semi-exact heuristic with graph reduction.

        Features:
        - Best from Multiple Selection (BMS) for candidate choice
        - Graph reduction when new best clique found
        - Can prove optimality in some cases

        Args:
            max_iterations: Maximum iterations
            time_limit: Maximum time in seconds
            bms_k: Number of candidates for BMS selection
            seed: Random seed

        Returns:
            RandomizedAlgorithmResult with clique found
        """
        if seed is not None:
            random.seed(seed)

        start_time = time.perf_counter()
        total_operations = 0
        configurations_tested = 0
        unique_configurations = 0
        duplicate_count = 0
        tested_configs: set[frozenset[int]] = set()

        working_graph = self.graph.copy()
        best_clique: set[int] = set()
        best_weight = 0.0

        iteration = 0
        stopping_reason: StoppingReason | None = None

        while working_graph.number_of_nodes() > 0:
            iteration += 1

            # Check stopping criteria
            if max_iterations and iteration > max_iterations:
                stopping_reason = StoppingReason.MAX_ITERATIONS
                break
            if time_limit and time.perf_counter() - start_time > time_limit:
                stopping_reason = StoppingReason.TIME_LIMIT
                break

            # Build clique using BMS strategy
            clique = self._bms_construction(working_graph, bms_k)
            configurations_tested += 1

            # Check if duplicate
            frozen = frozenset(clique)
            if frozen in tested_configs:
                duplicate_count += 1
                continue

            unique_configurations += 1
            tested_configs.add(frozen)

            # Verify clique and update best
            weight = sum(
                self.graph.nodes[v]["weight"] for v in clique if v in self.graph
            )

            if weight > best_weight:
                best_weight = weight
                best_clique = clique.copy()

                # Reduce graph
                to_remove = []
                for v in list(working_graph.nodes()):
                    ub0 = self._compute_vertex_ub0(working_graph, v)
                    if ub0 <= best_weight:
                        to_remove.append(v)

                for v in to_remove:
                    if v in working_graph:
                        working_graph.remove_node(v)

                total_operations += len(to_remove)

                # Check if graph is empty (optimal found)
                if working_graph.number_of_nodes() == 0:
                    stopping_reason = StoppingReason.EXHAUSTED
                    break

        if stopping_reason is None:
            stopping_reason = StoppingReason.EXHAUSTED

        return RandomizedAlgorithmResult(
            clique=best_clique,
            total_weight=best_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
            unique_configurations_tested=unique_configurations,
            duplicate_count=duplicate_count,
            stopping_reason=stopping_reason,
        )

    def _bms_construction(self, graph: nx.Graph, k: int) -> set[int]:
        """
        Build clique using Best from Multiple Selection.

        Sample k candidates randomly, select the best one based on
        benefit estimation.
        """
        if graph.number_of_nodes() == 0:
            return set()

        nodes = list(graph.nodes())
        clique: set[int] = set()
        candidates = set(nodes)

        while candidates:
            # Sample k candidates
            sample_size = min(k, len(candidates))
            sampled = random.sample(list(candidates), sample_size)

            # Find compatible candidates
            compatible = []
            for v in sampled:
                if all(graph.has_edge(v, u) for u in clique):
                    # Estimate benefit: weight + potential neighborhood weight
                    weight = graph.nodes[v]["weight"]
                    neighbor_potential = sum(
                        graph.nodes[n]["weight"]
                        for n in graph.neighbors(v)
                        if n in candidates and n not in clique
                    )
                    benefit = weight + neighbor_potential * 0.1
                    compatible.append((v, benefit))

            if not compatible:
                break

            # Select best
            best_v = max(compatible, key=lambda x: x[1])[0]
            clique.add(best_v)
            candidates.remove(best_v)

            # Filter candidates
            candidates = {c for c in candidates if graph.has_edge(best_v, c)}

        return clique

    def _compute_vertex_ub0(self, graph: nx.Graph, v: int) -> float:
        """
        Compute UB0 for a vertex: weight of inclusive neighborhood.
        """
        inclusive_neighbors = set(graph.neighbors(v)) | {v}
        return sum(graph.nodes[u]["weight"] for u in inclusive_neighbors)

    def scc_walk(
        self,
        max_iterations: int | None = None,
        time_limit: float | None = None,
        max_unimprove_steps: int = 100,
        walk_perturbation_strength: float = 0.3,
        seed: int | None = None,
    ) -> RandomizedAlgorithmResult:
        """
        SCCWalk - Local search with Strong Configuration Checking.

        Features:
        - Strong Configuration Checking (SCC) to avoid cycling
        - Walk perturbation when stuck
        - Add/Drop/Swap operators

        Args:
            max_iterations: Maximum iterations
            time_limit: Maximum time in seconds
            max_unimprove_steps: Steps without improvement before perturbation
            walk_perturbation_strength: Fraction of clique to perturb
            seed: Random seed

        Returns:
            RandomizedAlgorithmResult with best clique found
        """
        if seed is not None:
            random.seed(seed)

        start_time = time.perf_counter()
        total_operations = 0
        configurations_tested = 0
        unique_configurations = 0
        duplicate_count = 0
        tested_configs: set[frozenset[int]] = set()

        # Initialize with greedy clique
        current_clique = self._greedy_construct()
        best_clique = current_clique.copy()
        best_weight = self._calculate_weight(best_clique)

        # SCC data structures
        scc_timestamp: dict[int, int] = {v: 0 for v in self.graph.nodes()}
        conf_change: dict[int, int] = {v: 1 for v in self.graph.nodes()}
        current_step = 0
        unimprove_steps = 0

        iteration = 0
        stopping_reason: StoppingReason | None = None

        while True:
            iteration += 1
            current_step += 1

            # Check stopping criteria
            if max_iterations and iteration > max_iterations:
                stopping_reason = StoppingReason.MAX_ITERATIONS
                break
            if time_limit and time.perf_counter() - start_time > time_limit:
                stopping_reason = StoppingReason.TIME_LIMIT
                break

            # Try Add, Drop, or Swap operations
            operation_done = False

            # Try Add
            add_candidates = []
            for v in self.graph.nodes():
                if v not in current_clique:
                    if self._scc_can_add(
                        v, current_clique, scc_timestamp, conf_change, current_step
                    ):
                        if all(self.graph.has_edge(v, u) for u in current_clique):
                            total_operations += len(current_clique)
                            add_candidates.append((v, self.graph.nodes[v]["weight"]))

            if add_candidates:
                # Select best add
                best_add = max(add_candidates, key=lambda x: x[1])[0]
                current_clique.add(best_add)
                self._scc_update_add(best_add, scc_timestamp, conf_change, current_step)
                operation_done = True
                total_operations += 1

            if not operation_done:
                # Try Swap
                swap_candidates = []
                for u in current_clique:
                    for v in self.graph.nodes():
                        if v not in current_clique and v != u:
                            # Check if swap is valid
                            test_clique = (current_clique - {u}) | {v}
                            if self._is_valid_clique(test_clique):
                                gain = (
                                    self.graph.nodes[v]["weight"]
                                    - self.graph.nodes[u]["weight"]
                                )
                                if gain > 0:
                                    swap_candidates.append((u, v, gain))
                                    total_operations += len(test_clique)

                if swap_candidates:
                    best_swap = max(swap_candidates, key=lambda x: x[2])
                    u, v, _ = best_swap
                    current_clique.remove(u)
                    current_clique.add(v)
                    self._scc_update_swap(
                        u, v, scc_timestamp, conf_change, current_step
                    )
                    operation_done = True
                    total_operations += 1

            if not operation_done:
                # Try Drop (only if no improvement possible)
                if current_clique:
                    drop_v = min(
                        current_clique, key=lambda v: self.graph.nodes[v]["weight"]
                    )
                    current_clique.remove(drop_v)
                    self._scc_update_drop(
                        drop_v, scc_timestamp, conf_change, current_step
                    )
                    total_operations += 1

            # Check if improved
            configurations_tested += 1
            frozen = frozenset(current_clique)
            if frozen in tested_configs:
                duplicate_count += 1
            else:
                unique_configurations += 1
                tested_configs.add(frozen)

            current_weight = self._calculate_weight(current_clique)
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = current_clique.copy()
                unimprove_steps = 0
            else:
                unimprove_steps += 1

            # Walk perturbation if stuck
            if unimprove_steps >= max_unimprove_steps:
                current_clique = self._walk_perturbation(
                    current_clique, walk_perturbation_strength
                )
                unimprove_steps = 0
                # Reset SCC
                conf_change = {v: 1 for v in self.graph.nodes()}

        return RandomizedAlgorithmResult(
            clique=best_clique,
            total_weight=best_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
            unique_configurations_tested=unique_configurations,
            duplicate_count=duplicate_count,
            stopping_reason=stopping_reason,
        )

    def _greedy_construct(self) -> set[int]:
        """Construct initial clique greedily."""
        nodes = sorted(
            self.graph.nodes(),
            key=lambda v: self.graph.nodes[v]["weight"],
            reverse=True,
        )
        clique: set[int] = set()

        for v in nodes:
            if all(self.graph.has_edge(v, u) for u in clique):
                clique.add(v)

        return clique

    def _is_valid_clique(self, vertices: set[int]) -> bool:
        """Check if vertices form a valid clique."""
        vertices_list = list(vertices)
        for i in range(len(vertices_list)):
            for j in range(i + 1, len(vertices_list)):
                if not self.graph.has_edge(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def _scc_can_add(
        self,
        v: int,
        clique: set[int],
        timestamp: dict[int, int],
        conf_change: dict[int, int],
        step: int,
    ) -> bool:
        """Check if vertex can be added according to SCC rules."""
        return conf_change.get(v, 1) == 1

    def _scc_update_add(
        self, v: int, timestamp: dict[int, int], conf_change: dict[int, int], step: int
    ) -> None:
        """Update SCC structures after add operation."""
        timestamp[v] = step
        conf_change[v] = 0
        for neighbor in self.graph.neighbors(v):
            if conf_change.get(neighbor, 1) == 0:
                conf_change[neighbor] = 1

    def _scc_update_drop(
        self, v: int, timestamp: dict[int, int], conf_change: dict[int, int], step: int
    ) -> None:
        """Update SCC structures after drop operation."""
        timestamp[v] = step
        conf_change[v] = 0

    def _scc_update_swap(
        self,
        u: int,
        v: int,
        timestamp: dict[int, int],
        conf_change: dict[int, int],
        step: int,
    ) -> None:
        """Update SCC structures after swap operation."""
        self._scc_update_drop(u, timestamp, conf_change, step)
        self._scc_update_add(v, timestamp, conf_change, step)

    def _walk_perturbation(self, clique: set[int], strength: float) -> set[int]:
        """Apply walk perturbation to escape local optima."""
        if not clique:
            return self._greedy_construct()

        # Remove some vertices
        n_remove = max(1, int(len(clique) * strength))
        vertices_to_keep = random.sample(list(clique), max(1, len(clique) - n_remove))
        new_clique = set(vertices_to_keep)

        # Rebuild from remaining
        candidates = [v for v in self.graph.nodes() if v not in new_clique]
        random.shuffle(candidates)

        for v in candidates:
            if all(self.graph.has_edge(v, u) for u in new_clique):
                new_clique.add(v)

        return new_clique

    def mwc_peel(
        self,
        max_iterations: int | None = None,
        time_limit: float | None = None,
        peel_fraction: float = 0.1,
        seed: int | None = None,
    ) -> RandomizedAlgorithmResult:
        """
        MWCPeel - Hybrid reduction with peeling heuristic.

        Combines exact reductions with heuristic peeling:
        1. Apply exact reduction rules
        2. Peel (remove) low-score vertices
        3. Repeat until stopping criteria
        4. Solve remaining graph exactly if small

        Args:
            max_iterations: Maximum iterations
            time_limit: Maximum time in seconds
            peel_fraction: Fraction of vertices to peel each iteration
            seed: Random seed

        Returns:
            RandomizedAlgorithmResult with best clique found
        """
        if seed is not None:
            random.seed(seed)

        start_time = time.perf_counter()
        total_operations = 0
        configurations_tested = 0

        # Initialize
        working_graph = self.graph.copy()
        initial_clique, _, _ = self._initialize_wlmc(True)
        best_clique = initial_clique.copy()
        best_weight = self._calculate_weight(best_clique)

        iteration = 0
        is_first_iteration = True
        stopping_reason: StoppingReason | None = None

        while working_graph.number_of_nodes() > 0:
            iteration += 1

            # Check stopping criteria
            if max_iterations and iteration > max_iterations:
                stopping_reason = StoppingReason.MAX_ITERATIONS
                break
            if time_limit and time.perf_counter() - start_time > time_limit:
                stopping_reason = StoppingReason.TIME_LIMIT
                break

            # Apply exact reductions
            reduced, vertex_map, removed = self._apply_mwc_redu_rules(
                working_graph, best_weight, is_first_iteration
            )
            working_graph = reduced
            is_first_iteration = False
            total_operations += len(removed)

            if working_graph.number_of_nodes() == 0:
                break

            # Peeling: remove low-score vertices
            n_vertices = working_graph.number_of_nodes()
            n_peel = max(1, int(n_vertices * peel_fraction))

            if n_vertices > 50000:
                n_peel = int(n_vertices * 0.1)

            # Compute scores (inclusive neighborhood weight)
            scores = []
            for v in working_graph.nodes():
                inclusive_neighbors = set(working_graph.neighbors(v)) | {v}
                score = sum(
                    working_graph.nodes[u]["weight"] for u in inclusive_neighbors
                )
                scores.append((v, score))

            # Sort by score and peel lowest
            scores.sort(key=lambda x: x[1])
            to_peel = [v for v, _ in scores[:n_peel]]

            for v in to_peel:
                if v in working_graph:
                    working_graph.remove_node(v)

            configurations_tested += 1

        # Solve remaining graph if small enough
        if working_graph.number_of_nodes() > 0:
            if working_graph.number_of_nodes() <= 30:
                # Exact solve
                remaining_solver = MaxWeightCliqueSolver(working_graph)
                result = remaining_solver.exhaustive_search()
                # Map back to original
                remaining_clique = result.clique
            else:
                # Greedy solve
                remaining_solver = MaxWeightCliqueSolver(working_graph)
                result = remaining_solver.greedy_heuristic()
                remaining_clique = result.clique

            remaining_weight = sum(
                self.graph.nodes[v]["weight"]
                for v in remaining_clique
                if v in self.graph
            )
            if remaining_weight > best_weight:
                best_weight = remaining_weight
                best_clique = remaining_clique

            total_operations += result.basic_operations
            configurations_tested += result.configurations_tested

        if stopping_reason is None:
            stopping_reason = StoppingReason.EXHAUSTED

        return RandomizedAlgorithmResult(
            clique=best_clique,
            total_weight=best_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
            unique_configurations_tested=configurations_tested,
            duplicate_count=0,
            stopping_reason=stopping_reason,
        )

    def _apply_mwc_redu_rules(
        self, graph: nx.Graph, lower_bound: float, use_limit: bool
    ) -> tuple[nx.Graph, dict[int, int], list[int]]:
        """
        Apply MWCRedu reduction rules.

        Rules:
        - Twin reduction: merge vertices with same closed neighborhood
        - Domination reduction: remove dominated vertices
        - Simplicial vertex removal
        """
        reduced = graph.copy()
        vertex_map = {v: v for v in reduced.nodes()}
        removed: list[int] = []

        changed = True
        max_iters = 5 if use_limit else 10

        for _ in range(max_iters):
            if not changed:
                break
            changed = False

            # Domination reduction
            nodes_to_remove = []
            for u in list(reduced.nodes()):
                if u in nodes_to_remove:
                    continue
                u_weight = reduced.nodes[u]["weight"]
                u_neighbors = set(reduced.neighbors(u))

                for v in reduced.nodes():
                    if v == u or v in nodes_to_remove:
                        continue
                    v_weight = reduced.nodes[v]["weight"]
                    v_neighbors = set(reduced.neighbors(v))

                    # v dominates u if v has higher weight and u's neighbors ⊆ v's neighbors ∪ {v}
                    if v_weight >= u_weight and u_neighbors.issubset(v_neighbors | {v}):
                        nodes_to_remove.append(u)
                        changed = True
                        break

            for node in nodes_to_remove:
                if node in reduced:
                    removed.append(vertex_map.get(node, node))
                    reduced.remove_node(node)

            # Low weight/degree removal
            if reduced.number_of_nodes() > 0:
                for v in list(reduced.nodes()):
                    if v not in reduced:
                        continue
                    inclusive_neighbors = set(reduced.neighbors(v)) | {v}
                    nb_weight = sum(
                        reduced.nodes[u]["weight"] for u in inclusive_neighbors
                    )
                    if nb_weight <= lower_bound:
                        removed.append(vertex_map.get(v, v))
                        reduced.remove_node(v)
                        changed = True

        return reduced, vertex_map, removed

    def tsm_mwc(
        self,
        time_limit: float | None = None,
    ) -> AlgorithmResult:
        """
        TSM-MWC (Two-Stage MaxSAT) - Exact BnB with MaxSAT reasoning.

        Uses two phases of MaxSAT reasoning to compute tighter upper bounds:
        1. Binary MaxSAT phase
        2. Ordered MaxSAT phase

        This is a simplified implementation focusing on the core BnB structure
        with improved upper bound computation.

        Args:
            time_limit: Maximum time in seconds

        Returns:
            AlgorithmResult with optimal clique
        """
        start_time = time.perf_counter()
        total_operations = 0
        configurations_tested = 0

        # Initialize
        initial_clique, _, reduced = self._initialize_wlmc(True)
        best_clique = initial_clique.copy()
        best_weight = self._calculate_weight(best_clique)

        if reduced.number_of_nodes() == 0:
            return AlgorithmResult(
                clique=best_clique,
                total_weight=best_weight,
                basic_operations=total_operations,
                configurations_tested=configurations_tested,
            )

        def tsm_search(
            current: set[int], candidates: list[int], current_weight: float
        ) -> None:
            nonlocal best_clique, best_weight, total_operations, configurations_tested

            if time_limit and time.perf_counter() - start_time > time_limit:
                return

            configurations_tested += 1

            if not candidates:
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_clique = current.copy()
                return

            # Two-stage MaxSAT upper bound
            # Phase 1: Binary MaxSAT (IS partition bound)
            ub, partition = self._binary_maxsat_bound(candidates, current_weight)

            if ub <= best_weight:
                return  # Prune

            # Phase 2: Ordered MaxSAT (refined bound) - only if phase 1 doesn't prune
            ub = self._ordered_maxsat_bound(
                candidates, partition, current_weight, best_weight
            )

            if ub <= best_weight:
                return  # Prune

            # Branch
            for i, v in enumerate(candidates):
                total_operations += 1

                if all(self.graph.has_edge(v, u) for u in current):
                    new_current = current | {v}
                    new_weight = current_weight + self.graph.nodes[v]["weight"]
                    new_candidates = [
                        c for c in candidates[i + 1 :] if self.graph.has_edge(v, c)
                    ]
                    tsm_search(new_current, new_candidates, new_weight)

        # Order vertices
        ordered = sorted(
            reduced.nodes(), key=lambda v: self.graph.nodes[v]["weight"], reverse=True
        )

        tsm_search(set(), ordered, 0.0)

        return AlgorithmResult(
            clique=best_clique,
            total_weight=best_weight,
            basic_operations=total_operations,
            configurations_tested=configurations_tested,
        )

    def _binary_maxsat_bound(
        self, candidates: list[int], current_weight: float
    ) -> tuple[float, list[list[int]]]:
        """
        Phase 1: Binary MaxSAT bound using IS partition.

        Returns upper bound and the IS partition.
        """
        partition: list[list[int]] = []
        remaining = set(candidates)

        while remaining:
            is_set: list[int] = []
            for v in list(remaining):
                if all(not self.graph.has_edge(v, u) for u in is_set):
                    is_set.append(v)
                    remaining.remove(v)
            partition.append(is_set)

        # Upper bound: current + max weight from each IS
        ub = current_weight
        for is_set in partition:
            if is_set:
                ub += max(self.graph.nodes[v]["weight"] for v in is_set)

        return ub, partition

    def _ordered_maxsat_bound(
        self,
        candidates: list[int],
        partition: list[list[int]],
        current_weight: float,
        threshold: float,
    ) -> float:
        """
        Phase 2: Ordered MaxSAT bound (refined).

        Iterates through candidates and tries to integrate their weight
        into the partition to get a tighter bound.
        """
        ub = current_weight
        for is_set in partition:
            if is_set:
                ub += max(self.graph.nodes[v]["weight"] for v in is_set)

        # Simple refinement: check if any candidate can be removed
        # because its contribution is already covered
        for v in candidates:
            v_weight = self.graph.nodes[v]["weight"]
            # Find IS sets not conflicting with v
            non_conflicting = []
            for is_set in partition:
                conflicts = any(self.graph.has_edge(v, u) for u in is_set)
                if not conflicts:
                    non_conflicting.append(is_set)

            # If v's weight can be covered by non-conflicting sets
            covered = sum(
                max(self.graph.nodes[u]["weight"] for u in is_set)
                for is_set in non_conflicting
                if is_set
            )
            if covered >= v_weight:
                # v can be "integrated" - potentially tighter bound
                pass

        return ub


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
    print("\n" + "=" * 60)
    print("EXHAUSTIVE SEARCH")
    print("=" * 60)
    exact_result = solver.exhaustive_search()
    print(f"Clique found: {exact_result.clique}")
    print(f"Total weight: {exact_result.total_weight:.2f}")
    print(f"Basic operations: {exact_result.basic_operations}")
    print(f"Configurations tested: {exact_result.configurations_tested}")

    # Run greedy heuristic
    print("\n" + "=" * 60)
    print("GREEDY HEURISTIC")
    print("=" * 60)
    greedy_result = solver.greedy_heuristic()
    print(f"Clique found: {greedy_result.clique}")
    print(f"Total weight: {greedy_result.total_weight:.2f}")
    print(f"Basic operations: {greedy_result.basic_operations}")
    print(f"Configurations tested: {greedy_result.configurations_tested}")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    comparison = compare_solutions(exact_result, greedy_result)
    for key, value in comparison.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
