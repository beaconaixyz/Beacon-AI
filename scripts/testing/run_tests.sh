#!/bin/bash

# BEACON Test Runner Script

# Default values
TEST_DIR="tests"
COVERAGE_DIR="htmlcov"
REPORT_DIR="test_reports"
PARALLEL_JOBS=4
VERBOSE=false
COVERAGE=false
BENCHMARK=false
SKIP_SLOW=false
FAIL_FAST=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--test-dir)
            TEST_DIR="$2"
            shift
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -b|--benchmark)
            BENCHMARK=true
            shift
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --skip-slow)
            SKIP_SLOW=true
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p "$REPORT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Build pytest command
PYTEST_CMD="pytest"

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add parallel execution
if [ "$PARALLEL_JOBS" -gt 1 ]; then
    PYTEST_CMD="$PYTEST_CMD -n $PARALLEL_JOBS"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=beacon --cov-report=html:$COVERAGE_DIR --cov-report=xml:$REPORT_DIR/coverage.xml"
fi

# Add benchmark
if [ "$BENCHMARK" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --benchmark-only --benchmark-json=$REPORT_DIR/benchmark.json"
fi

# Skip slow tests
if [ "$SKIP_SLOW" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
fi

# Fail fast
if [ "$FAIL_FAST" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -x"
fi

# Add test directory
PYTEST_CMD="$PYTEST_CMD $TEST_DIR"

# Print configuration
echo "Test Configuration:"
echo "- Test Directory: $TEST_DIR"
echo "- Coverage: $COVERAGE"
echo "- Benchmark: $BENCHMARK"
echo "- Parallel Jobs: $PARALLEL_JOBS"
echo "- Skip Slow Tests: $SKIP_SLOW"
echo "- Fail Fast: $FAIL_FAST"
echo "- Command: $PYTEST_CMD"
echo

# Run tests
echo "Running tests..."
if $PYTEST_CMD; then
    TEST_EXIT_CODE=$?
else
    TEST_EXIT_CODE=$?
fi

# Generate test summary
echo -e "\nTest Summary:"
if [ "$COVERAGE" = true ]; then
    echo "Coverage report generated in: $COVERAGE_DIR"
    echo "Coverage XML report: $REPORT_DIR/coverage.xml"
fi

if [ "$BENCHMARK" = true ]; then
    echo "Benchmark results: $REPORT_DIR/benchmark.json"
    # Generate benchmark comparison if previous results exist
    if [ -f "$REPORT_DIR/benchmark_prev.json" ]; then
        pytest-benchmark compare "$REPORT_DIR/benchmark_prev.json" "$REPORT_DIR/benchmark.json"
    fi
    # Save current benchmark as previous for next run
    cp "$REPORT_DIR/benchmark.json" "$REPORT_DIR/benchmark_prev.json"
fi

# Check for test failures
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "\nAll tests passed successfully!"
else
    echo -e "\nSome tests failed. Exit code: $TEST_EXIT_CODE"
fi

# Generate test report
echo -e "\nGenerating test report..."
python -c "
import json
import datetime

report = {
    'timestamp': datetime.datetime.now().isoformat(),
    'exit_code': $TEST_EXIT_CODE,
    'coverage_enabled': $COVERAGE,
    'benchmark_enabled': $BENCHMARK,
    'skip_slow': $SKIP_SLOW,
    'parallel_jobs': $PARALLEL_JOBS
}

with open('$REPORT_DIR/test_summary.json', 'w') as f:
    json.dump(report, f, indent=4)
"

exit $TEST_EXIT_CODE 