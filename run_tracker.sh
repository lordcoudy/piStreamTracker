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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

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
    echo "Network configured. Re-run $0 as a normal user to start the tracker."
    exit 0
fi

if [[ $EUID -eq 0 ]]; then
    echo "Refusing to run the tracker as root." >&2
    exit 1
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
    if [[ -w "$cpu" ]]; then
        echo performance > "$cpu" 2>/dev/null || true
    fi
done

PRESET=pi5
RUN_WEB=false
EXTRA_ARGS=()

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
            EXTRA_ARGS+=("$1")
            shift
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
    exec python3 web.py --preset "$PRESET" "${EXTRA_ARGS[@]}"
else
    echo "Starting tracker..."
    exec python3 tracker.py --preset "$PRESET" "${EXTRA_ARGS[@]}"
fi
