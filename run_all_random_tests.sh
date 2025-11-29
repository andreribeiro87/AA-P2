#!/bin/bash
# Comprehensive Test Script for All MWC Algorithms
# Tests all algorithms on graphs of varying sizes
# Limits exhaustive search to <=20 vertices, other algorithms test larger graphs

# Parse command-line arguments
GENERATE_GRAPHS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --generate-graphs|-g)
            GENERATE_GRAPHS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--generate-graphs|-g]"
            exit 1
            ;;
    esac
done

# Don't exit on error - we want to continue testing other algorithms even if one fails
# set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SEED=112974
BASE_DIR="experiments"
SMALL_GRAPHS_DIR="${BASE_DIR}/graphs_small"  # 4-20 vertices for exhaustive
LARGE_GRAPHS_DIR="${BASE_DIR}/graphs_large"  # 20-200 vertices for scalability
RESULTS_DIR="${BASE_DIR}/results_randomized"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${BASE_DIR}/test_run_${TIMESTAMP}.log"

# Separate algorithm categories
declare -a RANDOM_ALGORITHMS=(
    "random_construction"
    "random_greedy_hybrid"
    "iterative_random_search"
    "monte_carlo"
    "las_vegas"
)

# Reduction/exact algorithms (deterministic)
declare -a REDUCTION_ALGORITHMS=(
    "mwc_redu"
    "max_clique_weight"
    "max_clique_dyn_weight"
)

# Exact Branch & Bound algorithms (from literature)
declare -a EXACT_BNB_ALGORITHMS=(
    "wlmc"
    "tsm_mwc"
)

# Heuristic/Local Search algorithms (from literature)
declare -a HEURISTIC_ALGORITHMS=(
    "fast_wclq"
    "scc_walk"
    "mwc_peel"
)

# Function to print colored messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if uv is available
check_uv() {
    if ! command -v uv &> /dev/null; then
        log_error "uv is not installed. Please install it first."
        exit 1
    fi
    log_info "Using uv: $(uv --version)"
}

# Function to generate graphs
generate_graphs() {
    log_info "Generating test graphs..."
    
    # Generate small graphs (4-20 vertices) for exhaustive search comparison
    log_info "Generating small graphs (4-20 vertices) for exhaustive comparison..."
    uv run python main.py generate \
        --seed "$SEED" \
        --min-vertices 4 \
        --max-vertices 20 \
        --densities 12.5 \
        --densities 25.0 \
        --densities 50.0 \
        --densities 75.0 \
        --output-dir "$SMALL_GRAPHS_DIR" || {
        log_warning "Failed to generate small graphs, using existing if available"
    }
    
    # Generate large graphs (20-50 vertices) for randomized algorithms
    log_info "Generating large graphs (20-50 vertices) for randomized algorithms..."
    uv run python main.py generate \
        --seed "$SEED" \
        --min-vertices 20 \
        --max-vertices 200 \
        --densities 12.5 \
        --densities 25.0 \
        --densities 50.0 \
        --densities 75.0 \
        --output-dir "$LARGE_GRAPHS_DIR" || {
        log_warning "Failed to generate large graphs, using existing if available"
    }
    
    log_success "Graph generation completed"
}

# Function to test a single random algorithm on a set of graphs
test_random_algorithm() {
    local algorithm=$1
    local graphs_dir=$2
    local output_subdir=$3
    local max_vertices=$4
    
    log_info "Testing randomized algorithm: ${algorithm}"
    log_info "  Graph directory: ${graphs_dir}"
    log_info "  Output directory: ${output_subdir}"
    log_info "  Max vertices: ${max_vertices}"
    
    local output_path="${RESULTS_DIR}/${output_subdir}"
    mkdir -p "$output_path"
    
    # Prepare algorithm-specific parameters
    local extra_args=""
    local time_limit=30.0
    
    case "$algorithm" in
        "random_construction")
            extra_args="--random-max-iterations 2000"
            if [ -n "$time_limit" ]; then
                extra_args="${extra_args} --random-time-limit ${time_limit}"
            fi
            ;;
        "random_greedy_hybrid")
            # Note: random_greedy_hybrid doesn't accept time_limit
            extra_args="--random-num-starts 20 --random-top-k 5 --random-factor 0.6"
            ;;
        "iterative_random_search")
            extra_args="--random-max-iterations 2000"
            if [ -n "$time_limit" ]; then
                extra_args="${extra_args} --random-time-limit ${time_limit}"
            fi
            ;;
        "monte_carlo")
            extra_args="--random-max-iterations 2000"
            if [ -n "$time_limit" ]; then
                extra_args="${extra_args} --random-time-limit ${time_limit}"
            fi
            ;;
        "las_vegas")
            extra_args="--random-max-iterations 5000"
            if [ -n "$time_limit" ]; then
                extra_args="${extra_args} --random-time-limit ${time_limit}"
            fi
            ;;
    esac
    
    # Run benchmark with proper vertex range - ONLY random algorithm
    local vertex_range="4..${max_vertices}"
    
    log_info "  Running benchmark (randomized algorithm only)..."
    uv run python main.py benchmark \
        --graphs-dir "$graphs_dir" \
        --output-dir "$output_path" \
        --exhaustive none \
        --heuristic none \
        --random \
        --random-range "${vertex_range}" \
        --random-strategy "$algorithm" \
        --random-seed "$SEED" \
        $extra_args \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_error "Failed to run benchmark for ${algorithm}"
        return 1
    }
    
    log_success "Completed testing ${algorithm}"
}

# Function to test a single reduction algorithm on a set of graphs
test_reduction_algorithm() {
    local algorithm=$1
    local graphs_dir=$2
    local output_subdir=$3
    local max_vertices=$4
    
    log_info "Testing reduction algorithm: ${algorithm}"
    log_info "  Graph directory: ${graphs_dir}"
    log_info "  Output directory: ${output_subdir}"
    log_info "  Max vertices: ${max_vertices}"
    
    local output_path="${RESULTS_DIR}/${output_subdir}"
    mkdir -p "$output_path"
    
    # Prepare algorithm-specific parameters
    local extra_args=""
    
    case "$algorithm" in
        "mwc_redu")
            # MWCRedu parameters: reduction rules, solver method
            # Note: aggressive flag defaults to false, can add --redu-aggressive to enable
            extra_args="--redu-reduction-rules domination,isolation,degree --redu-solver-method greedy"
            ;;
        "max_clique_weight")
            # MaxCliqueWeight parameters: variant, color ordering, use reduction
            extra_args="--redu-variant static --redu-color-ordering weight_desc --redu-use-reduction false"
            ;;
        "max_clique_dyn_weight")
            # MaxCliqueDynWeight parameters: color ordering, use reduction
            extra_args="--redu-color-ordering weight_desc --redu-use-reduction false"
            ;;
    esac
    
    # Run benchmark with proper vertex range - ONLY reduction algorithm
    local vertex_range="4..${max_vertices}"
    
    log_info "  Running benchmark (reduction algorithm only)..."
    uv run python main.py benchmark \
        --graphs-dir "$graphs_dir" \
        --output-dir "$output_path" \
        --exhaustive none \
        --heuristic none \
        --reduction \
        --reduction-range "${vertex_range}" \
        --reduction-strategy "$algorithm" \
        $extra_args \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_error "Failed to run benchmark for ${algorithm}"
        return 1
    }
    
    log_success "Completed testing ${algorithm}"
}

# Function to test exact BnB algorithms (WLMC, TSM-MWC)
test_exact_bnb_algorithm() {
    local algorithm=$1
    local graphs_dir=$2
    local output_subdir=$3
    local max_vertices=$4
    
    log_info "Testing exact BnB algorithm: ${algorithm}"
    log_info "  Graph directory: ${graphs_dir}"
    log_info "  Output directory: ${output_subdir}"
    log_info "  Max vertices: ${max_vertices}"
    
    local output_path="${RESULTS_DIR}/${output_subdir}"
    mkdir -p "$output_path"
    
    # These are exact algorithms - use longer time limit
    local time_limit=60.0
    
    # Run benchmark - exact BnB algorithms don't use seed
    local vertex_range="4..${max_vertices}"
    
    log_info "  Running benchmark (exact BnB algorithm)..."
    uv run python main.py benchmark \
        --graphs-dir "$graphs_dir" \
        --output-dir "$output_path" \
        --exhaustive none \
        --heuristic none \
        --random \
        --random-range "${vertex_range}" \
        --random-strategy "$algorithm" \
        --random-time-limit "${time_limit}" \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_error "Failed to run benchmark for ${algorithm}"
        return 1
    }
    
    log_success "Completed testing ${algorithm}"
}

# Function to test heuristic/local search algorithms (FastWClq, SCCWalk, MWCPeel)
test_heuristic_algorithm() {
    local algorithm=$1
    local graphs_dir=$2
    local output_subdir=$3
    local max_vertices=$4
    
    log_info "Testing heuristic/local search algorithm: ${algorithm}"
    log_info "  Graph directory: ${graphs_dir}"
    log_info "  Output directory: ${output_subdir}"
    log_info "  Max vertices: ${max_vertices}"
    
    local output_path="${RESULTS_DIR}/${output_subdir}"
    mkdir -p "$output_path"
    
    # Prepare algorithm-specific parameters
    local extra_args=""
    local time_limit=30.0
    
    case "$algorithm" in
        "fast_wclq")
            extra_args="--random-max-iterations 5000"
            ;;
        "scc_walk")
            extra_args="--random-max-iterations 5000"
            ;;
        "mwc_peel")
            extra_args="--random-max-iterations 2000"
            ;;
    esac
    
    # Run benchmark with proper vertex range
    local vertex_range="4..${max_vertices}"
    
    log_info "  Running benchmark (heuristic algorithm)..."
    uv run python main.py benchmark \
        --graphs-dir "$graphs_dir" \
        --output-dir "$output_path" \
        --exhaustive none \
        --heuristic none \
        --random \
        --random-range "${vertex_range}" \
        --random-strategy "$algorithm" \
        --random-seed "$SEED" \
        --random-time-limit "${time_limit}" \
        $extra_args \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_error "Failed to run benchmark for ${algorithm}"
        return 1
    }
    
    log_success "Completed testing ${algorithm}"
}

# Function to run exhaustive search on small graphs (for comparison)
run_exhaustive_baseline() {
    log_info "Running exhaustive search baseline on small graphs (4-20 vertices)..."
    
    local output_path="${RESULTS_DIR}/exhaustive_baseline"
    mkdir -p "$output_path"
    
    uv run python main.py benchmark \
        --graphs-dir "$SMALL_GRAPHS_DIR" \
        --output-dir "$output_path" \
        --exhaustive "4..20" \
        --heuristic "4..20" \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Exhaustive baseline failed or timed out"
    }
    
    log_success "Exhaustive baseline completed"
}

# Function to run greedy heuristic on all graphs (for comparison)
run_greedy_baseline() {
    log_info "Running greedy heuristic baseline on all graphs..."
    
    local output_path="${RESULTS_DIR}/greedy_baseline"
    mkdir -p "$output_path"
    
    # Test on small graphs
    uv run python main.py benchmark \
        --graphs-dir "$SMALL_GRAPHS_DIR" \
        --output-dir "${output_path}_small" \
        --exhaustive none \
        --heuristic "4..20" \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Greedy baseline (small) failed"
    }
    
    # Test on large graphs
    uv run python main.py benchmark \
        --graphs-dir "$LARGE_GRAPHS_DIR" \
        --output-dir "${output_path}_large" \
        --exhaustive none \
        --heuristic "20..200" \
        --verbose \
        2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Greedy baseline (large) failed"
    }
    
    log_success "Greedy baseline completed"
}

# Main execution
main() {
    echo "=========================================="
    echo "Comprehensive MWC Algorithms Test Suite"
    echo "=========================================="
    echo "Timestamp: ${TIMESTAMP}"
    echo "Log file: ${LOG_FILE}"
    if [ "$GENERATE_GRAPHS" = true ]; then
        echo "Graph generation: ENABLED"
    else
        echo "Graph generation: DISABLED (use --generate-graphs to enable)"
    fi
    echo ""
    
    # Initialize log file
    echo "Test run started at $(date)" > "$LOG_FILE"
    echo "Seed: ${SEED}" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # Check prerequisites
    check_uv
    
    # Create directories
    mkdir -p "$SMALL_GRAPHS_DIR" "$LARGE_GRAPHS_DIR" "$RESULTS_DIR"
    
    # Check if graphs exist, make generation optional
    local graphs_exist_small=false
    local graphs_exist_large=false
    
    if [ -d "$SMALL_GRAPHS_DIR" ] && [ -n "$(ls -A $SMALL_GRAPHS_DIR 2>/dev/null)" ]; then
        graphs_exist_small=true
    fi
    
    if [ -d "$LARGE_GRAPHS_DIR" ] && [ -n "$(ls -A $LARGE_GRAPHS_DIR 2>/dev/null)" ]; then
        graphs_exist_large=true
    fi
    
    # Generate graphs if requested
    if [ "$GENERATE_GRAPHS" = true ]; then
        generate_graphs
    elif [ "$graphs_exist_small" = false ] || [ "$graphs_exist_large" = false ]; then
        log_warning "Some graph directories are missing or empty."
        log_warning "Use --generate-graphs to generate graphs, or provide graphs manually."
        log_warning "Continuing with existing graphs (if any)..."
        
        # Try to use existing graphs directory if available
        if [ ! -d "$SMALL_GRAPHS_DIR" ] || [ "$graphs_exist_small" = false ]; then
            if [ -d "${BASE_DIR}/graphs" ] && [ -n "$(ls -A ${BASE_DIR}/graphs 2>/dev/null)" ]; then
                log_info "Using ${BASE_DIR}/graphs for small graphs"
                SMALL_GRAPHS_DIR="${BASE_DIR}/graphs"
                graphs_exist_small=true
            fi
        fi
        if [ ! -d "$LARGE_GRAPHS_DIR" ] || [ "$graphs_exist_large" = false ]; then
            if [ -d "${BASE_DIR}/graphs" ] && [ -n "$(ls -A ${BASE_DIR}/graphs 2>/dev/null)" ]; then
                log_info "Using ${BASE_DIR}/graphs for large graphs"
                LARGE_GRAPHS_DIR="${BASE_DIR}/graphs"
                graphs_exist_large=true
            fi
        fi
    else
        log_info "Using existing graphs in ${SMALL_GRAPHS_DIR} and ${LARGE_GRAPHS_DIR}"
    fi
    
    # Run baseline algorithms first
    log_info "=========================================="
    log_info "Running Baseline Algorithms"
    log_info "=========================================="
    
    run_exhaustive_baseline
    run_greedy_baseline
    
    # Test randomized algorithms
    log_info "=========================================="
    log_info "Testing Randomized Algorithms"
    log_info "=========================================="
    
    for algorithm in "${RANDOM_ALGORITHMS[@]}"; do
        log_info ""
        log_info "----------------------------------------"
        
        # Test on small graphs first (for comparison with exhaustive)
        test_random_algorithm "$algorithm" "$SMALL_GRAPHS_DIR" "${algorithm}_small" 20
        
        # Test on large graphs
        test_random_algorithm "$algorithm" "$LARGE_GRAPHS_DIR" "${algorithm}_large" 200
        
        log_success "Completed all tests for ${algorithm}"
    done
    
    # Test reduction/exact algorithms
    log_info ""
    log_info "=========================================="
    log_info "Testing Reduction Algorithms"
    log_info "=========================================="
    
    for algorithm in "${REDUCTION_ALGORITHMS[@]}"; do
        log_info ""
        log_info "----------------------------------------"
        
        # Test on small graphs first (for comparison with exhaustive)
        test_reduction_algorithm "$algorithm" "$SMALL_GRAPHS_DIR" "${algorithm}_small" 20
        
        # Test on large graphs
        test_reduction_algorithm "$algorithm" "$LARGE_GRAPHS_DIR" "${algorithm}_large" 100
        
        log_success "Completed all tests for ${algorithm}"
    done
    
    # Test exact BnB algorithms
    log_info ""
    log_info "=========================================="
    log_info "Testing Exact Branch & Bound Algorithms"
    log_info "=========================================="
    
    for algorithm in "${EXACT_BNB_ALGORITHMS[@]}"; do
        log_info ""
        log_info "----------------------------------------"
        
        # Test on small graphs (these are exact, can be slow on large graphs)
        test_exact_bnb_algorithm "$algorithm" "$SMALL_GRAPHS_DIR" "${algorithm}_small" 20
        
        # Test on medium graphs with time limit
        test_exact_bnb_algorithm "$algorithm" "$LARGE_GRAPHS_DIR" "${algorithm}_large" 50
        
        log_success "Completed all tests for ${algorithm}"
    done
    
    # Test heuristic/local search algorithms
    log_info ""
    log_info "=========================================="
    log_info "Testing Heuristic/Local Search Algorithms"
    log_info "=========================================="
    
    for algorithm in "${HEURISTIC_ALGORITHMS[@]}"; do
        log_info ""
        log_info "----------------------------------------"
        
        # Test on small graphs first
        test_heuristic_algorithm "$algorithm" "$SMALL_GRAPHS_DIR" "${algorithm}_small" 20
        
        # Test on large graphs
        test_heuristic_algorithm "$algorithm" "$LARGE_GRAPHS_DIR" "${algorithm}_large" 200
        
        log_success "Completed all tests for ${algorithm}"
    done
    
    # Summary
    log_info ""
    log_info "=========================================="
    log_info "Test Summary"
    log_info "=========================================="
    log_info "All results saved to: ${RESULTS_DIR}"
    log_info "Log file: ${LOG_FILE}"
    log_info ""
    log_info "Tested randomized algorithms:"
    for algorithm in "${RANDOM_ALGORITHMS[@]}"; do
        log_info "  - ${algorithm}"
    done
    log_info ""
    log_info "Tested reduction algorithms:"
    for algorithm in "${REDUCTION_ALGORITHMS[@]}"; do
        log_info "  - ${algorithm}"
    done
    log_info ""
    log_info "Tested exact BnB algorithms:"
    for algorithm in "${EXACT_BNB_ALGORITHMS[@]}"; do
        log_info "  - ${algorithm}"
    done
    log_info ""
    log_info "Tested heuristic/local search algorithms:"
    for algorithm in "${HEURISTIC_ALGORITHMS[@]}"; do
        log_info "  - ${algorithm}"
    done
    log_info ""
    log_success "All tests completed!"
    log_info ""
    log_info "To visualize results, run:"
    log_info "  uv run python main.py visualize ${RESULTS_DIR}"
}

# Run main function
main "$@"