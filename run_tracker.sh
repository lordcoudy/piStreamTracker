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

if [[ "${1:-}" == "--configure-network" ]]; then
    shift
    if [[ $EUID -eq 0 ]]; then
        echo "Refusing to run as root. Use: $0 --configure-network" >&2
        echo "The installer will sudo only for nmcli/networkd." >&2
        exit 1
    fi
    PY="$SCRIPT_DIR/venv/bin/python"
    if [[ ! -x "$PY" ]]; then
        PY="$SCRIPT_DIR/.venv/bin/python"
    fi
    if [[ ! -x "$PY" ]]; then
        echo "No venv. Run ./setup.sh --tracker first." >&2
        exit 1
    fi
    exec "$PY" -m pistream.install network --role tracker "$@"
fi

if [[ $EUID -eq 0 ]]; then
    echo "Refusing to run the tracker as root." >&2
    exit 1
fi

if command -v systemctl >/dev/null 2>&1; then
    state=$(systemctl is-active pitracker 2>/dev/null || true)
    if [[ "$state" == "active" || "$state" == "activating" ]]; then
        echo "pitracker.service is already running."
        echo "  UI:     http://192.168.100.2:5000  (IPs in config.yaml)"
        echo "  Status: sudo systemctl status pitracker"
        echo "  Logs:   journalctl -u pitracker -e"
        echo "  Manual: sudo systemctl stop pitracker && $0 $*"
        exit 0
    fi
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
