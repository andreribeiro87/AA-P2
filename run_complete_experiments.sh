#!/bin/bash
# =============================================================================
# Complete MWC Experiments Script
# =============================================================================
# This script runs all Maximum Weight Clique algorithms on generated graphs
# and produces comprehensive visualizations for the assignment report.
#
# Tests:
# 1. DEFAULT: Run all algorithms with pairwise comparisons and visualizations
# 2. CORRECTNESS TEST: Run exhaustive + all algorithms on graphs 4..20 vertices
#    - Compare all algorithm results against exhaustive (ground truth)
# 3. SCALABILITY TEST: Run all algorithms (except exhaustive) on large graphs
#    - Uses graphs from custom directory (50..100 vertices or more)
#
# Usage:
#   ./run_complete_experiments.sh [OPTIONS]
#
# Options:
#   --generate-graphs, -g    Generate new test graphs
#   --quick, -q              Run quick tests only (smaller graphs, fewer iterations)
#   --skip-baseline          Skip exhaustive and greedy baseline tests
#   --skip-visualize         Skip visualization generation
#   --correctness-only       Run correctness tests (4..20 vertices)
#   --scalability-only       Run scalability tests (50+ vertices)
#                            (both can be combined: --correctness-only --scalability-only)
#   --graphs-dir DIR         Use custom graphs directory
#   --output-dir DIR         Use custom output directory for results
#   --no-vertex-filter       Disable vertex count filtering (use all graphs)
#   --help, -h               Show this help message
#
# =============================================================================

set -e  # Exit on error

# Parse command-line arguments
GENERATE_GRAPHS=false
QUICK_MODE=false
SKIP_BASELINE=false
SKIP_VISUALIZE=false
CORRECTNESS_ONLY=false
SCALABILITY_ONLY=false
CUSTOM_GRAPHS_DIR=""
CUSTOM_OUTPUT_DIR=""
NO_VERTEX_FILTER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --generate-graphs|-g)
            GENERATE_GRAPHS=true
            shift
            ;;
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --skip-visualize)
            SKIP_VISUALIZE=true
            shift
            ;;
        --correctness-only)
            CORRECTNESS_ONLY=true
            shift
            ;;
        --scalability-only)
            SCALABILITY_ONLY=true
            shift
            ;;
        --graphs-dir)
            CUSTOM_GRAPHS_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --generate-graphs, -g    Generate new test graphs"
            echo "  --quick, -q              Run quick tests only (smaller graphs)"
            echo "  --skip-baseline          Skip exhaustive and greedy baseline tests"
            echo "  --skip-visualize         Skip visualization generation"
            echo "  --correctness-only       Run correctness tests (4..20 vertices)"
            echo "  --scalability-only       Run scalability tests (50+ vertices)"
            echo "                           (both can be combined to run all tests)"
            echo "  --graphs-dir DIR         Use custom graphs directory (any format)"
            echo "  --output-dir DIR         Use custom output directory for results"
            echo "  --no-vertex-filter       Disable vertex count filtering (use all graphs)"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
        --no-vertex-filter)
            NO_VERTEX_FILTER=true
            shift
            ;;    
        --output-dir)
            CUSTOM_OUTPUT_DIR="$2"
            shift 2
            ;;    
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SEED=112974
BASE_DIR="experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$QUICK_MODE" = true ]; then
    GRAPHS_DIR="${BASE_DIR}/graphs_complete"  # Use graphs_complete (has all sizes)
    RESULTS_DIR="${BASE_DIR}/results_complete_quick" 
    PLOTS_DIR="${BASE_DIR}/plots_quick"
    MIN_VERTICES=4
    MAX_VERTICES=15
    LARGE_MAX_VERTICES=30
    # Correctness test range (exhaustive feasible)
    CORRECTNESS_MIN=4
    CORRECTNESS_MAX=15
    # Scalability test range
    SCALE_MIN=50
    SCALE_MAX=70
    TIME_LIMIT=10.0
    MAX_ITERATIONS=1000
else
    GRAPHS_DIR="${BASE_DIR}/graphs_complete"
    RESULTS_DIR="${BASE_DIR}/results_complete"
    PLOTS_DIR="${BASE_DIR}/plots"
    MIN_VERTICES=4
    MAX_VERTICES=20
    LARGE_MAX_VERTICES=100
    # Correctness test range (exhaustive feasible)
    CORRECTNESS_MIN=4
    CORRECTNESS_MAX=20
    # Scalability test range
    SCALE_MIN=50
    SCALE_MAX=100
    TIME_LIMIT=30.0
    MAX_ITERATIONS=5000
fi

# Override graphs dir if custom specified
if [ -n "$CUSTOM_GRAPHS_DIR" ]; then
    GRAPHS_DIR="$CUSTOM_GRAPHS_DIR"
fi

# Override output dir if custom specified
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    RESULTS_DIR="$CUSTOM_OUTPUT_DIR"
fi

# Override vertex filters if --no-vertex-filter is set
if [ "$NO_VERTEX_FILTER" = true ]; then
    MIN_VERTICES=1
    MAX_VERTICES=1000000
    LARGE_MAX_VERTICES=1000000
    CORRECTNESS_MIN=1
    CORRECTNESS_MAX=1000000
    SCALE_MIN=1
    SCALE_MAX=1000000
fi

LOG_FILE="${BASE_DIR}/complete_run_${TIMESTAMP}.log"

# All algorithms (15 total)
declare -a ALL_ALGORITHMS=(
    "exhaustive"
    "greedy"
    "random_construction"
    "random_greedy_hybrid"
    "iterative_random_search"
    "monte_carlo"
    "las_vegas"
    "mwc_redu"
    "max_clique_weight"
    "max_clique_dyn_weight"
    "wlmc"
    "tsm_mwc"
    "fast_wclq"
    "scc_walk"
    "mwc_peel"
)

# All algorithms except exhaustive (for scalability tests)
declare -a SCALABILITY_ALGORITHMS=(
    "greedy"
    "random_construction"
    "random_greedy_hybrid"
    "iterative_random_search"
    "monte_carlo"
    "las_vegas"
    "mwc_redu"
    "max_clique_weight"
    "max_clique_dyn_weight"
    "wlmc"
    "tsm_mwc"
    "fast_wclq"
    "scc_walk"
    "mwc_peel"
)

# Algorithm categories (for default mode)
declare -a BASELINE_ALGORITHMS=(
    "exhaustive"
    "greedy"
)

declare -a RANDOM_ALGORITHMS=(
    "random_construction"
    "random_greedy_hybrid"
    "iterative_random_search"
    "monte_carlo"
    "las_vegas"
)

declare -a REDUCTION_ALGORITHMS=(
    "mwc_redu"
    "max_clique_weight"
    "max_clique_dyn_weight"
)

declare -a EXACT_BNB_ALGORITHMS=(
    "wlmc"
    "tsm_mwc"
)

declare -a HEURISTIC_ALGORITHMS=(
    "fast_wclq"
    "scc_walk"
    "mwc_peel"
)

# Logging functions
log_header() {
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════╗${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}║${NC} ${MAGENTA}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}\n" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}▶${NC} $1" | tee -a "$LOG_FILE"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1" | tee -a "$LOG_FILE"
}

log_progress() {
    echo -e "${CYAN}[→]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_section "Checking Prerequisites"
    
    if ! command -v uv &> /dev/null; then
        log_error "uv is not installed. Please install it first."
        exit 1
    fi
    log_success "uv found: $(uv --version)"
    
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        log_error "Python is not installed."
        exit 1
    fi
    log_success "Python found"
    
    # Check if main.py exists
    if [ ! -f "main.py" ]; then
        log_error "main.py not found. Please run from project root."
        exit 1
    fi
    log_success "Project structure verified"
}

# Generate graphs
generate_graphs() {
    log_section "Generating Test Graphs"
    
    mkdir -p "$GRAPHS_DIR"
    
    log_info "Generating graphs with ${MIN_VERTICES}-${LARGE_MAX_VERTICES} vertices..."
    uv run python main.py generate \
        --seed "$SEED" \
        --min-vertices "$MIN_VERTICES" \
        --max-vertices "$LARGE_MAX_VERTICES" \
        --densities 12.5 \
        --densities 25.0 \
        --densities 50.0 \
        --densities 75.0 \
        --output-dir "$GRAPHS_DIR" || {
        log_warning "Failed to generate graphs, using existing if available"
        return 1
    }
    
    local graph_count=$(ls -1 "$GRAPHS_DIR"/*.graphml 2>/dev/null | wc -l)
    log_success "Generated ${graph_count} graphs in ${GRAPHS_DIR}"
}

# Run baseline algorithms (exhaustive + greedy)
run_baseline() {
    log_section "Running Baseline Algorithms (Exhaustive + Greedy)"
    
    # Run exhaustive on small graphs only (vertex-limited)
    local exhaustive_output="${RESULTS_DIR}/exhaustive"
    mkdir -p "$exhaustive_output"
    
    log_progress "Running exhaustive search on graphs (${MIN_VERTICES}..${MAX_VERTICES} vertices)..."
    uv run python main.py benchmark \
        --graphs-dir "$GRAPHS_DIR" \
        --output-dir "$exhaustive_output" \
        --algorithm exhaustive \
        --min-vertices "$MIN_VERTICES" \
        --max-vertices "$MAX_VERTICES" \
        --time-limit "$TIME_LIMIT" \
        --seed "$SEED" \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Exhaustive benchmark had some errors"
    }
    log_success "Exhaustive search completed"
    
    # Run greedy on all graphs (up to LARGE_MAX_VERTICES)
    local greedy_output="${RESULTS_DIR}/greedy"
    mkdir -p "$greedy_output"
    
    log_progress "Running greedy heuristic on graphs (${MIN_VERTICES}..${LARGE_MAX_VERTICES} vertices)..."
    uv run python main.py benchmark \
        --graphs-dir "$GRAPHS_DIR" \
        --output-dir "$greedy_output" \
        --algorithm greedy \
        --min-vertices "$MIN_VERTICES" \
        --max-vertices "$LARGE_MAX_VERTICES" \
        --time-limit "$TIME_LIMIT" \
        --seed "$SEED" \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Greedy benchmark had some errors"
    }
    log_success "Greedy heuristic completed"
    
    log_success "Baseline algorithms completed"
}

# Run a single algorithm benchmark
run_algorithm() {
    local algorithm=$1
    local algorithm_type=$2  # random, reduction, exact, heuristic (for logging only)
    local max_v=$3  # Maximum vertices for this algorithm
    
    local output_path="${RESULTS_DIR}/${algorithm}"
    mkdir -p "$output_path"
    
    log_progress "Testing ${algorithm} (${MIN_VERTICES}..${max_v} vertices)..."
    
    # Determine time limit based on algorithm type
    local alg_time_limit=$TIME_LIMIT
    if [ "$algorithm" = "wlmc" ] || [ "$algorithm" = "tsm_mwc" ]; then
        # Exact BnB algorithms may need more time
        alg_time_limit=60.0
    fi
    
    uv run python main.py benchmark \
        --graphs-dir "$GRAPHS_DIR" \
        --output-dir "$output_path" \
        --algorithm "$algorithm" \
        --min-vertices "$MIN_VERTICES" \
        --max-vertices "$max_v" \
        --time-limit "$alg_time_limit" \
        --max-iterations "$MAX_ITERATIONS" \
        --seed "$SEED" \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_warning "${algorithm} had some errors"
        return 1
    }
    
    log_success "${algorithm} completed"
}

# Generate visualizations
generate_visualizations() {
    log_section "Generating Visualizations"
    
    mkdir -p "$PLOTS_DIR"
    
    # First, generate legacy plots from BenchmarkResult files (if any)
    log_progress "Creating legacy plots from benchmark results..."
    uv run python main.py visualize \
        "$RESULTS_DIR" \
        --output-dir "$PLOTS_DIR/legacy" \
        2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Legacy visualization generation had some errors"
    }
    
    # Generate new individual algorithm charts and pairwise comparisons
    log_progress "Creating individual algorithm and pairwise comparison charts..."
    uv run python main.py visualize-individual \
        "$RESULTS_DIR" \
        --output-dir "$PLOTS_DIR" \
        2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Individual visualization generation had some errors"
    }
    
    # Generate quality comparison plots (if correctness results exist)
    if [ -d "${RESULTS_DIR}/correctness" ]; then
        log_progress "Creating solution quality comparison plots..."
        uv run python main.py compare-quality \
            "${RESULTS_DIR}/correctness" \
            --output-dir "$PLOTS_DIR/quality" \
            2>&1 | tee -a "$LOG_FILE" || {
            log_warning "Quality comparison generation had some errors"
        }
    fi
    
    local plot_count=$(find "$PLOTS_DIR" -name "*.png" 2>/dev/null | wc -l)
    log_success "Generated ${plot_count} plots in ${PLOTS_DIR}"
}

# Print final summary
print_summary() {
    log_header "Experiment Summary"
    
    echo -e "${GREEN}Results Directory:${NC} ${RESULTS_DIR}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}Plots Directory:${NC} ${PLOTS_DIR}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}Log File:${NC} ${LOG_FILE}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Count results
    local result_count
    result_count=$(find "$RESULTS_DIR" -name "benchmark_results.json" 2>/dev/null | wc -l)
    echo -e "${BLUE}Benchmark result files:${NC} ${result_count}" | tee -a "$LOG_FILE"
    
    if [ -d "$PLOTS_DIR" ]; then
        local plot_count
        plot_count=$(ls -1 "$PLOTS_DIR"/*.png 2>/dev/null | wc -l)
        echo -e "${BLUE}Generated plots:${NC} ${plot_count}" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo -e "${CYAN}Algorithms tested:${NC}" | tee -a "$LOG_FILE"
    
    if [ "$CORRECTNESS_ONLY" = true ] && [ "$SCALABILITY_ONLY" = true ]; then
        echo -e "  ${GREEN}Correctness:${NC} ${CORRECTNESS_MIN}..${CORRECTNESS_MAX} vertices (${#ALL_ALGORITHMS[@]} algorithms)" | tee -a "$LOG_FILE"
        echo -e "  ${GREEN}Scalability:${NC} ${SCALE_MIN}..${SCALE_MAX} vertices (${#SCALABILITY_ALGORITHMS[@]} algorithms)" | tee -a "$LOG_FILE"
    elif [ "$CORRECTNESS_ONLY" = true ]; then
        echo -e "  ${GREEN}Correctness:${NC} ${CORRECTNESS_MIN}..${CORRECTNESS_MAX} vertices (${#ALL_ALGORITHMS[@]} algorithms)" | tee -a "$LOG_FILE"
    elif [ "$SCALABILITY_ONLY" = true ]; then
        echo -e "  ${GREEN}Scalability:${NC} ${SCALE_MIN}..${SCALE_MAX} vertices (${#SCALABILITY_ALGORITHMS[@]} algorithms)" | tee -a "$LOG_FILE"
    else
        if [ "$SKIP_BASELINE" = false ]; then
            echo -e "  ${GREEN}Baseline:${NC} exhaustive, greedy" | tee -a "$LOG_FILE"
        fi
        echo -e "  ${GREEN}Randomized:${NC} ${RANDOM_ALGORITHMS[*]}" | tee -a "$LOG_FILE"
        echo -e "  ${GREEN}Reduction:${NC} ${REDUCTION_ALGORITHMS[*]}" | tee -a "$LOG_FILE"
        echo -e "  ${GREEN}Exact BnB:${NC} ${EXACT_BNB_ALGORITHMS[*]}" | tee -a "$LOG_FILE"
        echo -e "  ${GREEN}Heuristic:${NC} ${HEURISTIC_ALGORITHMS[*]}" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}✅ All experiments completed successfully!${NC}" | tee -a "$LOG_FILE"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
}

# Run correctness tests (exhaustive + all algorithms on small graphs)
run_correctness_tests() {
    log_header "CORRECTNESS TESTS (${CORRECTNESS_MIN}..${CORRECTNESS_MAX} vertices)"
    
    local correctness_results="${RESULTS_DIR}/correctness"
    mkdir -p "$correctness_results"
    
    log_info "Running all algorithms on graphs with ${CORRECTNESS_MIN}..${CORRECTNESS_MAX} vertices"
    
    # Run ALL algorithms on these graphs (including exhaustive as ground truth)
    for algorithm in "${ALL_ALGORITHMS[@]}"; do
        log_section "Testing $algorithm (correctness)"
        
        local alg_output="${correctness_results}/${algorithm}"
        mkdir -p "$alg_output"
        
        log_progress "Running $algorithm (${CORRECTNESS_MIN}..${CORRECTNESS_MAX} vertices)..."
        
        uv run python main.py benchmark \
            --graphs-dir "$GRAPHS_DIR" \
            --output-dir "$alg_output" \
            --algorithm "$algorithm" \
            --min-vertices "$CORRECTNESS_MIN" \
            --max-vertices "$CORRECTNESS_MAX" \
            --time-limit "$TIME_LIMIT" \
            --max-iterations "$MAX_ITERATIONS" \
            --seed "$SEED" \
            --verbose \
            2>&1 | tee -a "$LOG_FILE" || {
            log_warning "$algorithm had some errors"
        }
        
        log_success "$algorithm completed"
    done
    
    log_success "Correctness tests completed for all ${#ALL_ALGORITHMS[@]} algorithms"
}

# Run scalability tests (all algorithms except exhaustive on large graphs)
run_scalability_tests() {
    log_header "SCALABILITY TESTS (${SCALE_MIN}..${SCALE_MAX} vertices)"
    
    local scale_results="${RESULTS_DIR}/scalability"
    mkdir -p "$scale_results"
    
    log_info "Running algorithms (except exhaustive) on graphs with ${SCALE_MIN}..${SCALE_MAX} vertices"
    
    # Run all algorithms EXCEPT exhaustive (too slow for large graphs)
    for algorithm in "${SCALABILITY_ALGORITHMS[@]}"; do
        log_section "Testing $algorithm (scalability)"
        
        local alg_output="${scale_results}/${algorithm}"
        mkdir -p "$alg_output"
        
        log_progress "Running $algorithm (${SCALE_MIN}..${SCALE_MAX} vertices)..."
        
        # Use longer time limit for scalability tests
        local scale_time_limit=$TIME_LIMIT
        if [ "$algorithm" = "wlmc" ] || [ "$algorithm" = "tsm_mwc" ]; then
            # Exact BnB algorithms may need more time on large graphs
            scale_time_limit=120.0
        fi
        
        uv run python main.py benchmark \
            --graphs-dir "$GRAPHS_DIR" \
            --output-dir "$alg_output" \
            --algorithm "$algorithm" \
            --min-vertices "$SCALE_MIN" \
            --max-vertices "$SCALE_MAX" \
            --time-limit "$scale_time_limit" \
            --max-iterations "$MAX_ITERATIONS" \
            --seed "$SEED" \
            --verbose \
            2>&1 | tee -a "$LOG_FILE" || {
            log_warning "$algorithm had some errors on scalability test"
        }
        
        log_success "$algorithm scalability test completed"
    done
    
    log_success "Scalability tests completed for all ${#SCALABILITY_ALGORITHMS[@]} algorithms"
}

# Run default tests (original behavior with pairwise comparisons)
run_default_tests() {
    # Generate graphs if requested or if directory is empty
    if [ "$GENERATE_GRAPHS" = true ]; then
        generate_graphs
    elif [ ! -d "$GRAPHS_DIR" ] || [ -z "$(ls -A "$GRAPHS_DIR" 2>/dev/null)" ]; then
        log_warning "Graphs directory is empty. Generating graphs..."
        generate_graphs
    else
        local graph_count
        graph_count=$(find "$GRAPHS_DIR" -name "*.graphml" 2>/dev/null | wc -l)
        log_info "Using existing ${graph_count} graphs in ${GRAPHS_DIR}"
    fi
    
    # Run baseline algorithms
    if [ "$SKIP_BASELINE" = false ]; then
        run_baseline
    else
        log_info "Skipping baseline algorithms"
    fi
    
    # Run randomized algorithms
    log_section "Testing Randomized Algorithms"
    for algorithm in "${RANDOM_ALGORITHMS[@]}"; do
        run_algorithm "$algorithm" "random" "$LARGE_MAX_VERTICES"
    done
    
    # Run reduction algorithms
    log_section "Testing Reduction Algorithms"
    for algorithm in "${REDUCTION_ALGORITHMS[@]}"; do
        run_algorithm "$algorithm" "reduction" "$LARGE_MAX_VERTICES"
    done
    
    # Run exact BnB algorithms
    log_section "Testing Exact Branch & Bound Algorithms"
    for algorithm in "${EXACT_BNB_ALGORITHMS[@]}"; do
        # Use smaller max vertices for exact algorithms (they can be slow)
        local exact_max_v=$MAX_VERTICES
        if [ "$QUICK_MODE" = false ]; then
            exact_max_v=$LARGE_MAX_VERTICES
        fi
        run_algorithm "$algorithm" "exact" "$exact_max_v"
    done
    
    # Run heuristic/local search algorithms
    log_section "Testing Heuristic/Local Search Algorithms"
    for algorithm in "${HEURISTIC_ALGORITHMS[@]}"; do
        run_algorithm "$algorithm" "heuristic" "$LARGE_MAX_VERTICES"
    done
}

# Main execution
main() {
    log_header "Complete MWC Experiments"
    
    echo "Timestamp: ${TIMESTAMP}"
    echo "Mode: $([ "$QUICK_MODE" = true ] && echo "QUICK" || echo "FULL")"
    if [ "$CORRECTNESS_ONLY" = true ] && [ "$SCALABILITY_ONLY" = true ]; then
        echo "Test: CORRECTNESS + SCALABILITY (all tests)"
    elif [ "$CORRECTNESS_ONLY" = true ]; then
        echo "Test: CORRECTNESS (${CORRECTNESS_MIN}..${CORRECTNESS_MAX} vertices)"
    elif [ "$SCALABILITY_ONLY" = true ]; then
        echo "Test: SCALABILITY (${SCALE_MIN}..${SCALE_MAX} vertices)"
    else
        echo "Test: DEFAULT (all algorithms with pairwise comparisons)"
    fi
    echo "Log file: ${LOG_FILE}"
    echo ""
    
    # Initialize log
    mkdir -p "$BASE_DIR"
    echo "Experiment run started at $(date)" > "$LOG_FILE"
    echo "Seed: ${SEED}" >> "$LOG_FILE"
    echo "Mode: $([ "$QUICK_MODE" = true ] && echo "QUICK" || echo "FULL")" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # Check prerequisites
    check_prerequisites
    
    # Create directories
    mkdir -p "$GRAPHS_DIR" "$RESULTS_DIR" "$PLOTS_DIR"
    
    # Decide which tests to run (flags are non-exclusive)
    if [ "$CORRECTNESS_ONLY" = true ] && [ "$SCALABILITY_ONLY" = true ]; then
        # Run both correctness and scalability tests
        run_default_tests
        run_correctness_tests
        run_scalability_tests
    elif [ "$CORRECTNESS_ONLY" = true ]; then
        # Run only correctness tests
        run_correctness_tests
    elif [ "$SCALABILITY_ONLY" = true ]; then
        # Run only scalability tests
        run_scalability_tests
    else
        # Run default tests (original behavior)
        run_default_tests
    fi
    
    # Generate visualizations
    if [ "$SKIP_VISUALIZE" = false ]; then
        generate_visualizations
    else
        log_info "Skipping visualization generation"
    fi
    
    # Print summary
    print_summary
}

# Run main function
main
