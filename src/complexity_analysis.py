"""
Complexity Analysis Module for Maximum Weight Clique Algorithms.

Fits theoretical complexity models to experimental benchmark data
and computes R² correlation coefficients.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize


@dataclass
class FittedModel:
    """Result of fitting a complexity model to experimental data."""

    algorithm: str
    theoretical_complexity: str
    fitted_formula: str
    coefficients: dict[str, float]
    r_squared: float
    n_samples: int
    density: float | None = None


def load_benchmark_results(results_dir: Path, algorithm: str) -> list[dict]:
    """
    Load benchmark results for an algorithm from CSV files.

    Args:
        results_dir: Root directory containing benchmark results
        algorithm: Algorithm name (e.g., 'random_construction')

    Returns:
        List of result dictionaries with n_vertices, time, density, etc.
    """
    results = []

    # Try different possible paths
    possible_paths = [
        results_dir / algorithm / algorithm / "benchmark_results.csv",
        results_dir / algorithm / "benchmark_results.csv",
        results_dir / f"{algorithm}.csv",
    ]

    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        print(f"Warning: No results found for {algorithm}")
        return results

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract relevant fields
            n = int(row.get("n_vertices", 0))
            density = float(row.get("edge_density_percent", 0))

            # Get time based on algorithm type
            time_val = None
            if row.get("random_time_seconds"):
                time_val = float(row["random_time_seconds"])
            elif row.get("exact_time_seconds"):
                time_val = float(row["exact_time_seconds"])
            elif row.get("greedy_time_seconds"):
                time_val = float(row["greedy_time_seconds"])

            if n > 0 and time_val is not None and time_val > 0:
                results.append(
                    {
                        "n_vertices": n,
                        "time_seconds": time_val,
                        "density": density,
                    }
                )

    return results


def fit_quadratic_model(
    n_values: np.ndarray, time_values: np.ndarray
) -> tuple[float, float]:
    """
    Fit T(n) = a * n^2 model.

    Returns:
        Tuple of (coefficient_a, r_squared)
    """

    # Model: T = a * n^2
    def model(n, a):
        return a * n**2

    try:
        popt, _ = optimize.curve_fit(
            model, n_values, time_values, p0=[1e-6], maxfev=10000
        )
        a = popt[0]

        # Calculate R²
        predicted = model(n_values, a)
        ss_res = np.sum((time_values - predicted) ** 2)
        ss_tot = np.sum((time_values - np.mean(time_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return a, r_squared
    except Exception as e:
        print(f"  Warning: Could not fit quadratic model: {e}")
        return 0, 0


def fit_exponential_model(
    n_values: np.ndarray, time_values: np.ndarray
) -> tuple[float, float]:
    """
    Fit T(n) = a * 2^n * n model (for exhaustive search).

    Returns:
        Tuple of (coefficient_a, r_squared)
    """

    # Model: T = a * 2^n * n
    def model(n, a):
        return a * (2**n) * n

    try:
        # Use log transform for numerical stability
        popt, _ = optimize.curve_fit(
            model, n_values, time_values, p0=[1e-9], maxfev=10000
        )
        a = popt[0]

        # Calculate R²
        predicted = model(n_values, a)
        ss_res = np.sum((time_values - predicted) ** 2)
        ss_tot = np.sum((time_values - np.mean(time_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return a, r_squared
    except Exception as e:
        print(f"  Warning: Could not fit exponential model: {e}")
        return 0, 0


def fit_branch_bound_model(
    n_values: np.ndarray, time_values: np.ndarray
) -> tuple[float, float, float]:
    """
    Fit T(n) = a * 2^(b*n) * n^2 model (for branch-and-bound).

    This captures the exponential behavior with variable pruning effectiveness.

    Returns:
        Tuple of (coefficient_a, exponent_factor_b, r_squared)
    """

    # Model: T = a * 2^(b*n) * n^2  where b < 1 represents pruning effectiveness
    def model(n, a, b):
        return a * (2 ** (b * n)) * (n**2)

    try:
        popt, _ = optimize.curve_fit(
            model,
            n_values,
            time_values,
            p0=[1e-9, 0.1],
            bounds=([0, 0], [1, 1]),
            maxfev=10000,
        )
        a, b = popt

        # Calculate R²
        predicted = model(n_values, a, b)
        ss_res = np.sum((time_values - predicted) ** 2)
        ss_tot = np.sum((time_values - np.mean(time_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return a, b, r_squared
    except Exception as e:
        print(f"  Warning: Could not fit branch-bound model: {e}")
        return 0, 0, 0


def analyze_algorithm_complexity(
    results: list[dict],
    algorithm: str,
    model_type: str = "quadratic",
    filter_density: float | None = None,
) -> FittedModel | None:
    """
    Analyze complexity for a single algorithm.

    Args:
        results: List of benchmark results
        algorithm: Algorithm name
        model_type: "quadratic", "exponential", or "branch_bound"
        filter_density: If set, only use data from this density

    Returns:
        FittedModel with coefficients and R²
    """
    if not results:
        return None

    # Filter by density if specified
    if filter_density is not None:
        results = [r for r in results if abs(r["density"] - filter_density) < 1.0]

    if not results:
        return None

    # Convert to numpy arrays
    n_values = np.array([r["n_vertices"] for r in results])
    time_values = np.array([r["time_seconds"] for r in results])

    # Sort by n for better fitting
    sort_idx = np.argsort(n_values)
    n_values = n_values[sort_idx]
    time_values = time_values[sort_idx]

    # Remove outliers (times that are way off)
    if len(time_values) > 5:
        median_time = np.median(time_values)
        mask = time_values < median_time * 10  # Remove extreme outliers
        n_values = n_values[mask]
        time_values = time_values[mask]

    if len(n_values) < 3:
        print(f"  Warning: Not enough data points for {algorithm}")
        return None

    if model_type == "quadratic":
        a, r_squared = fit_quadratic_model(n_values, time_values)
        return FittedModel(
            algorithm=algorithm,
            theoretical_complexity="O(n²)" if algorithm == "greedy" else "O(Tn²)",
            fitted_formula=f"T(n) = {a:.2e} × n²",
            coefficients={"a": a},
            r_squared=r_squared,
            n_samples=len(n_values),
            density=filter_density,
        )

    elif model_type == "exponential":
        a, r_squared = fit_exponential_model(n_values, time_values)
        return FittedModel(
            algorithm=algorithm,
            theoretical_complexity="O(2ⁿ × n)",
            fitted_formula=f"T(n) = {a:.2e} × 2ⁿ × n",
            coefficients={"a": a},
            r_squared=r_squared,
            n_samples=len(n_values),
            density=filter_density,
        )

    elif model_type == "branch_bound":
        a, b, r_squared = fit_branch_bound_model(n_values, time_values)
        return FittedModel(
            algorithm=algorithm,
            theoretical_complexity="O(2^k × n²)",
            fitted_formula=f"T(n) = {a:.2e} × 2^({b:.2f}n) × n²",
            coefficients={"a": a, "b": b},
            r_squared=r_squared,
            n_samples=len(n_values),
            density=filter_density,
        )

    return None


def analyze_all_algorithms(
    results_dir: Path, filter_density: float | None = 50.0
) -> list[FittedModel]:
    """
    Analyze complexity for all algorithms.

    Args:
        results_dir: Directory containing benchmark results

    Returns:
        List of FittedModel objects
    """
    # Define algorithms and their expected complexity models
    algorithm_models = {
        # Randomized - O(T * n²)
        "random_construction": "quadratic",
        "random_greedy_hybrid": "quadratic",
        "iterative_random_search": "quadratic",
        "monte_carlo": "quadratic",
        "las_vegas": "quadratic",
        # Greedy - O(n²)
        "greedy": "quadratic",
        # Exact - O(2^n * n)
        "exhaustive": "exponential",
        # Branch-and-bound - O(2^k * n²) where k depends on pruning
        "wlmc": "branch_bound",
        "tsm_mwc": "branch_bound",
        # Reduction-based - varies
        "mwc_redu": "quadratic",
        "max_clique_weight": "branch_bound",
        "max_clique_dyn_weight": "branch_bound",
        # Heuristics
        "fast_wclq": "quadratic",
        "scc_walk": "quadratic",
        "mwc_peel": "quadratic",
    }

    fitted_models = []

    for algorithm, model_type in algorithm_models.items():
        print(f"Analyzing {algorithm}...")
        results = load_benchmark_results(results_dir, algorithm)

        if not results:
            continue

        model = analyze_algorithm_complexity(
            results, algorithm, model_type, filter_density
        )
        if model:
            fitted_models.append(model)
            print(
                f"  {model.fitted_formula} (R² = {model.r_squared:.3f}, n = {model.n_samples})"
            )

    return fitted_models


def generate_latex_table(models: list[FittedModel]) -> str:
    """
    Generate LaTeX table from fitted models.

    Args:
        models: List of FittedModel objects

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Experimental vs. Theoretical Complexity Validation}",
        r"    \label{tab:exp-vs-formal}",
        r"    \begin{tabular}{lccc}",
        r"        \toprule",
        r"        \textbf{Algorithm} & \textbf{Theoretical} & \textbf{Fitted Model} & $R^2$ \\",
        r"        \midrule",
    ]

    for model in models:
        alg_name = model.algorithm.replace("_", r"\_")
        theoretical = model.theoretical_complexity.replace("²", "^2")
        fitted = model.fitted_formula.replace("×", r"\times")
        r2 = f"{model.r_squared:.2f}"

        lines.append(f"        {alg_name} & ${theoretical}$ & ${fitted}$ & {r2} \\\\")

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def save_results(models: list[FittedModel], output_dir: Path) -> None:
    """
    Save fitted models to JSON and generate LaTeX table.

    Args:
        models: List of FittedModel objects
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = output_dir / "complexity_analysis.json"
    with open(json_path, "w") as f:
        json.dump(
            [
                {
                    "algorithm": m.algorithm,
                    "theoretical_complexity": m.theoretical_complexity,
                    "fitted_formula": m.fitted_formula,
                    "coefficients": m.coefficients,
                    "r_squared": m.r_squared,
                    "n_samples": m.n_samples,
                }
                for m in models
            ],
            f,
            indent=2,
        )
    print(f"✓ Results saved to {json_path}")

    # Generate LaTeX table
    latex_table = generate_latex_table(models)
    latex_path = output_dir / "complexity_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to {latex_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Theoretical':<15} {'R²':<8} {'Samples':<8}")
    print("-" * 60)
    for m in models:
        print(
            f"{m.algorithm:<25} {m.theoretical_complexity:<15} {m.r_squared:.3f}    {m.n_samples}"
        )


def main():
    """Main entry point for complexity analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze algorithm complexity from benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results_complete/"),
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/analysis"),
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    models = analyze_all_algorithms(args.results_dir)

    if models:
        save_results(models, args.output_dir)
    else:
        print("No models were fitted. Check that benchmark results exist.")


if __name__ == "__main__":
    main()
