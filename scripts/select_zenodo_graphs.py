#!/usr/bin/env python3
"""
Select and organize large graphs from the Zenodo dataset for scalability testing.

This script:
1. Scans the zenodo dataset for graph files
2. Computes edge density for each graph
3. Selects ~300 large graphs
4. Organizes them into density groups (25%, 50%, 75%)
5. Creates symlinks in the target directory structure
"""

import shutil
from pathlib import Path
from typing import NamedTuple


class GraphInfo(NamedTuple):
    """Information about a graph file."""

    path: Path
    n_vertices: int
    n_edges: int
    density: float
    file_size: int


def parse_graph_header(filepath: Path) -> tuple[int, int] | None:
    """
    Parse the graph header to get vertex and edge counts.

    Returns:
        Tuple of (n_vertices, n_edges) or None if parsing fails
    """
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("p "):
                    parts = line.split()
                    if len(parts) >= 4:
                        n_vertices = int(parts[2])
                        n_edges = int(parts[3])
                        return n_vertices, n_edges
                # Stop after reading first few lines to be efficient
                if not line.startswith("c") and not line.startswith("p"):
                    break
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    return None


def compute_density(n_vertices: int, n_edges: int) -> float:
    """
    Compute edge density of a graph.

    Density = 2*E / (V*(V-1)) for undirected graphs
    """
    if n_vertices <= 1:
        return 0.0
    max_edges = n_vertices * (n_vertices - 1) / 2
    return n_edges / max_edges if max_edges > 0 else 0.0


def scan_zenodo_graphs(zenodo_dir: Path) -> list[GraphInfo]:
    """
    Scan the zenodo directory for graph files and extract their info.
    """
    graphs = []

    # Find all graph files
    for ext in ["*.clq", "*.wclq"]:
        for filepath in zenodo_dir.rglob(ext):
            # Skip extremely large files (> 500MB) - would be too slow
            file_size = filepath.stat().st_size
            if file_size > 500_000_000:  # Skip files > 500MB
                print(
                    f"Skipping extremely large file: {filepath.name} ({file_size / 1e6:.1f} MB)"
                )
                continue

            header = parse_graph_header(filepath)
            if header:
                n_vertices, n_edges = header
                # Skip graphs with 0 vertices or very small graphs
                if n_vertices < 50:
                    continue
                density = compute_density(n_vertices, n_edges)
                graphs.append(
                    GraphInfo(
                        path=filepath,
                        n_vertices=n_vertices,
                        n_edges=n_edges,
                        density=density,
                        file_size=file_size,
                    )
                )

    return graphs


def classify_density(density: float) -> str:
    """
    Classify a graph into a density category.

    Categories:
    - sparse: density <= 0.33 (roughly 25% of max)
    - medium: 0.33 < density <= 0.66 (roughly 50%)
    - dense: density > 0.66 (roughly 75%)
    """
    if density <= 0.33:
        return "sparse_25"
    elif density <= 0.66:
        return "medium_50"
    else:
        return "dense_75"


def select_graphs(
    graphs: list[GraphInfo], target_count: int = 300
) -> dict[str, list[GraphInfo]]:
    """
    Select graphs aiming for roughly equal distribution across density categories.
    Prioritize larger graphs (more vertices) within each category.
    """
    # Categorize all graphs
    categories: dict[str, list[GraphInfo]] = {
        "sparse_25": [],
        "medium_50": [],
        "dense_75": [],
    }

    for g in graphs:
        cat = classify_density(g.density)
        categories[cat].append(g)

    # Sort each category by number of vertices (descending) to get larger graphs
    for cat in categories:
        categories[cat].sort(key=lambda x: x.n_vertices, reverse=True)

    # For sparse category, we take all available larger graphs
    # (there are naturally fewer sparse graphs in benchmark datasets)
    large_sparse = [g for g in categories["sparse_25"] if g.n_vertices >= 100]
    if large_sparse:
        categories["sparse_25"] = large_sparse

    # Select graphs - for sparse take all available, for others take per_category
    per_category = target_count // 3

    selected: dict[str, list[GraphInfo]] = {
        "sparse_25": [],
        "medium_50": [],
        "dense_75": [],
    }

    # Track seen base names to avoid duplicates across categories
    seen_names: set[str] = set()

    for cat in categories:
        # For sparse, take all available; for others limit to per_category
        max_count = len(categories[cat]) if cat == "sparse_25" else per_category
        count = 0
        for g in categories[cat]:
            if count >= max_count:
                break
            base_name = g.path.name
            if base_name not in seen_names:
                selected[cat].append(g)
                seen_names.add(base_name)
                count += 1

    # Print statistics
    print("\n=== Selection Statistics ===")
    total = 0
    for cat, graphs_list in selected.items():
        total += len(graphs_list)
        if graphs_list:
            avg_vertices = sum(g.n_vertices for g in graphs_list) / len(graphs_list)
            min_vertices = min(g.n_vertices for g in graphs_list)
            max_vertices = max(g.n_vertices for g in graphs_list)
            avg_density = sum(g.density for g in graphs_list) / len(graphs_list)
            min_density = min(g.density for g in graphs_list)
            max_density = max(g.density for g in graphs_list)
            print(f"{cat}: {len(graphs_list)} graphs")
            print(
                f"  Vertices: min={min_vertices}, max={max_vertices}, avg={avg_vertices:.0f}"
            )
            print(
                f"  Density: {min_density:.3f}-{max_density:.3f} (avg: {avg_density:.3f})"
            )
        else:
            print(f"{cat}: 0 graphs")
    print(f"Total selected: {total}")

    return selected


def copy_graphs_to_destination(selected: dict[str, list[GraphInfo]], dest_dir: Path):
    """
    Copy selected graphs to the destination directory structure.

    Creates structure:
    dest_dir/
        sparse_25/
        medium_50/
        dense_75/
    """
    for cat, graphs_list in selected.items():
        cat_dir = dest_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCopying {len(graphs_list)} graphs to {cat_dir}")

        # Track copied base names to avoid duplicates
        copied_names: set[str] = set()

        for g in graphs_list:
            # Use base filename only (avoid duplicates from different source folders)
            base_name = g.path.name

            if base_name in copied_names:
                print(f"  Skipped (duplicate): {base_name}")
                continue

            dest_path = cat_dir / base_name

            # Copy file (not symlink for portability)
            if not dest_path.exists():
                shutil.copy2(g.path, dest_path)
                print(
                    f"  Copied: {g.path.name} (V={g.n_vertices}, E={g.n_edges}, d={g.density:.3f})"
                )
                copied_names.add(base_name)
            else:
                print(f"  Skipped (exists): {base_name}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    zenodo_dir = (
        project_root
        / "experiments"
        / "graphs_zenodo"
        / "max-weight-clique-instances-master"
    )
    dest_dir = project_root / "results" / "large_graphs" / "zenodo_selection"

    print(f"Scanning graphs in: {zenodo_dir}")
    print(f"Destination: {dest_dir}")

    # Scan all graphs
    print("\n=== Scanning Zenodo Dataset ===")
    all_graphs = scan_zenodo_graphs(zenodo_dir)
    print(f"Found {len(all_graphs)} graph files")

    # Print overall statistics
    if all_graphs:
        print("\nOverall statistics:")
        print(
            f"  Vertices range: {min(g.n_vertices for g in all_graphs)} - {max(g.n_vertices for g in all_graphs)}"
        )
        print(
            f"  Edges range: {min(g.n_edges for g in all_graphs)} - {max(g.n_edges for g in all_graphs)}"
        )
        print(
            f"  Density range: {min(g.density for g in all_graphs):.4f} - {max(g.density for g in all_graphs):.4f}"
        )

    # Select graphs
    print("\n=== Selecting Graphs ===")
    selected = select_graphs(all_graphs, target_count=300)

    # Copy to destination
    print("\n=== Copying to Destination ===")
    copy_graphs_to_destination(selected, dest_dir)

    print("\n=== Done ===")
    print(f"Graphs organized in: {dest_dir}")


if __name__ == "__main__":
    main()
