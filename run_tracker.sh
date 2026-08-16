#!/usr/bin/env bash
#
# Run piStreamTracker - Tracker on Pi 5
#
# Usage:
#   ./run_tracker.sh           # Default Pi 5 settings
#   ./run_tracker.sh --fast    # Higher FPS, lower quality
#   ./run_tracker.sh --quality # Better detection accuracy
#   ./run_tracker.sh --web     # Web interface (headless)
#

set -e

# Optional one-shot network config (do not use if SSH is on eth0)
if [[ "${1:-}" == "--configure-network" ]]; then
    shift
    if [[ $EUID -ne 0 ]]; then
        echo "Network config requires root: sudo $0 --configure-network" >&2
        exit 1
    fi
    echo "Configuring eth0 as 192.168.100.2/24 (no flush of other addresses)..."
    ip addr add 192.168.100.2/24 dev eth0 2>/dev/null || true
    ip link set eth0 up
fi

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

# Set CPU governor to performance (if available; needs write access)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu" 2>/dev/null || true
done

PRESET=pi5
RUN_WEB=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pi5)
            PRESET=pi5
            echo "Mode: Raspberry Pi 5"
            shift
            ;;
        --fast)
            PRESET=fast
            echo "Mode: Fast (lower quality, higher FPS)"
            shift
            ;;
        --quality)
            PRESET=quality
            echo "Mode: High Quality"
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
echo "Preset: $PRESET"
echo "=========================================="

if [[ "$RUN_WEB" == true ]]; then
    echo "Starting web interface..."
    exec python3 web.py --preset "$PRESET" "$@"
else
    echo "Starting tracker..."
    exec python3 tracker.py --preset "$PRESET" "$@"
fi
