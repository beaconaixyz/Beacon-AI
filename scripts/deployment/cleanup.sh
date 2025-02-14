#!/bin/bash

# BEACON Cleanup Script

# Default values
LOG_DIR="/var/log/beacon"
DATA_DIR="/data/beacon"
MODEL_DIR="/models/beacon"
CACHE_DIR="/tmp/beacon"
DAYS_TO_KEEP=30
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -l|--log-dir)
            LOG_DIR="$2"
            shift
            shift
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift
            shift
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
            shift
            shift
            ;;
        -c|--cache-dir)
            CACHE_DIR="$2"
            shift
            shift
            ;;
        -k|--days-to-keep)
            DAYS_TO_KEEP="$2"
            shift
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to cleanup directory
cleanup_directory() {
    local dir=$1
    local days=$2
    local pattern=$3

    echo "Cleaning up $dir..."
    if [ ! -d "$dir" ]; then
        echo "Directory $dir does not exist. Skipping..."
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "Would delete the following files (dry run):"
        find "$dir" -type f -name "$pattern" -mtime +"$days" -print
    else
        find "$dir" -type f -name "$pattern" -mtime +"$days" -delete
        echo "Deleted files older than $days days in $dir"
    fi
}

# Function to cleanup empty directories
cleanup_empty_dirs() {
    local dir=$1
    
    echo "Cleaning up empty directories in $dir..."
    if [ ! -d "$dir" ]; then
        echo "Directory $dir does not exist. Skipping..."
        return
    }

    if [ "$DRY_RUN" = true ]; then
        echo "Would delete the following empty directories (dry run):"
        find "$dir" -type d -empty -print
    else
        find "$dir" -type d -empty -delete
        echo "Deleted empty directories in $dir"
    fi
}

# Cleanup log files
cleanup_directory "$LOG_DIR" "$DAYS_TO_KEEP" "*.log"
cleanup_directory "$LOG_DIR" "$DAYS_TO_KEEP" "*.log.*"

# Cleanup cached data
cleanup_directory "$CACHE_DIR" "7" "*"  # Cache files older than 7 days

# Cleanup old model files
cleanup_directory "$MODEL_DIR" "$DAYS_TO_KEEP" "*.pt"
cleanup_directory "$MODEL_DIR" "$DAYS_TO_KEEP" "*.pth"
cleanup_directory "$MODEL_DIR" "$DAYS_TO_KEEP" "*.ckpt"

# Cleanup temporary data files
cleanup_directory "$DATA_DIR/temp" "1" "*"  # Temp files older than 1 day

# Cleanup empty directories
cleanup_empty_dirs "$LOG_DIR"
cleanup_empty_dirs "$CACHE_DIR"
cleanup_empty_dirs "$MODEL_DIR"
cleanup_empty_dirs "$DATA_DIR"

# Cleanup Docker images if Docker is installed
if command -v docker &> /dev/null; then
    echo "Cleaning up Docker resources..."
    if [ "$DRY_RUN" = true ]; then
        echo "Would cleanup Docker resources (dry run)"
    else
        # Remove unused Docker images
        docker image prune -f
        # Remove stopped containers
        docker container prune -f
        # Remove unused volumes
        docker volume prune -f
    fi
fi

# Print summary
echo "Cleanup completed!"
echo "Directories processed:"
echo "- Logs: $LOG_DIR"
echo "- Data: $DATA_DIR"
echo "- Models: $MODEL_DIR"
echo "- Cache: $CACHE_DIR"

# Check disk usage
echo -e "\nCurrent disk usage:"
df -h 