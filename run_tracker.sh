#!/usr/bin/env bash
#
# Run piStreamTracker - Tracker on Pi 5
#
# Usage:
#   sudo ./run_tracker.sh           # Default Pi 5 settings
#   sudo ./run_tracker.sh --fast    # Higher FPS, lower quality
#   sudo ./run_tracker.sh --quality # Better detection accuracy
#   sudo ./run_tracker.sh --web     # Web interface (headless)
#

set -e

# Check root
if [[ $EUID -ne 0 ]]; then
    echo "Please run as root: sudo $0" >&2
    exit 1
fi

# Configure network
echo "Configuring network interface..."
ip addr flush dev eth0 2>/dev/null || true
ip addr add 192.168.100.2/24 dev eth0
ip link set eth0 up
echo "Network: 192.168.100.2/24"

# Activate virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -d "venv" ]]; then
        source venv/bin/activate
    elif [[ -d ".venv" ]]; then
        source .venv/bin/activate
    fi
fi

# Performance optimizations
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Set CPU governor to performance (if available)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu" 2>/dev/null || true
done

# Default settings for Pi 5 (primary tracker)
DETECTION_INTERVAL=6
PROCESS_SCALE=0.5
MOVENET_THREADS=4
RUN_WEB=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pi5)
            # Already defaults to Pi 5 settings
            echo "Mode: Raspberry Pi 5"
            shift
            ;;
        --fast)
            echo "Mode: Fast (lower quality, higher FPS)"
            DETECTION_INTERVAL=10
            PROCESS_SCALE=0.35
            shift
            ;;
        --quality)
            echo "Mode: High Quality"
            DETECTION_INTERVAL=4
            PROCESS_SCALE=0.6
            shift
            ;;
        --web)
            RUN_WEB=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

echo "=========================================="
echo "piStreamTracker"
echo "=========================================="
echo "Detection interval: $DETECTION_INTERVAL"
echo "Process scale:      $PROCESS_SCALE"
echo "MoveNet threads:    $MOVENET_THREADS"
echo "=========================================="

if [[ "$RUN_WEB" == true ]]; then
    echo "Starting web interface..."
    exec python3 web.py "$@"
else
    echo "Starting tracker..."
    exec python3 tracker.py \
        --detection-interval "$DETECTION_INTERVAL" \
        --process-scale "$PROCESS_SCALE" \
        --movenet-threads "$MOVENET_THREADS" \
        "$@"
fi
