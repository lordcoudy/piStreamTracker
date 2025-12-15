#!/usr/bin/env bash
# Optimized Human Tracker for Raspberry Pi
# Usage: ./run_tracker.sh [options]

set -e

# Check for virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -d "venv" ]]; then
        source venv/bin/activate
    elif [[ -d ".venv" ]]; then
        source .venv/bin/activate
    fi
fi

# Performance optimizations for Raspberry Pi
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Use performance governor if available (requires root)
if [[ $EUID -eq 0 ]] && command -v cpufreq-set &> /dev/null; then
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$cpu" 2>/dev/null || true
    done
fi

# Default settings optimized for Pi 3B (use --pi5 for better settings)
PI_VERSION=""
DETECTION_INTERVAL=10
PROCESS_SCALE=0.4
MOVENET_THREADS=2

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pi3)
            PI_VERSION="pi3"
            DETECTION_INTERVAL=12
            PROCESS_SCALE=0.35
            MOVENET_THREADS=2
            shift
            ;;
        --pi5)
            PI_VERSION="pi5"
            DETECTION_INTERVAL=6
            PROCESS_SCALE=0.5
            MOVENET_THREADS=4
            shift
            ;;
        *)
            break
            ;;
    esac
done

echo "==================================="
echo "Optimized Human Tracker"
if [[ -n "$PI_VERSION" ]]; then
    echo "Mode: Raspberry Pi ${PI_VERSION#pi}"
fi
echo "Detection interval: $DETECTION_INTERVAL"
echo "Process scale: $PROCESS_SCALE"
echo "Threads: $MOVENET_THREADS"
echo "==================================="

# Run the optimized tracker
exec python3 tracker_optimized.py \
    --detection-interval "$DETECTION_INTERVAL" \
    --process-scale "$PROCESS_SCALE" \
    --movenet-threads "$MOVENET_THREADS" \
    "$@"
