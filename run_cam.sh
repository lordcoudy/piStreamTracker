#!/usr/bin/env bash
#
# Run piStreamTracker - Camera Pi
#
# Usage:
#   ./run_cam.sh
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
        echo "No venv. Run ./setup.sh --camera first." >&2
        exit 1
    fi
    exec "$PY" -m pistream.install network --role camera "$@"
fi

if [[ $EUID -eq 0 ]]; then
    echo "Refusing to run the camera server as root." >&2
    exit 1
fi

if command -v systemctl >/dev/null 2>&1; then
    state=$(systemctl is-active pitracker 2>/dev/null || true)
    if [[ "$state" == "active" || "$state" == "activating" ]]; then
        echo "pitracker.service is already running and holds the camera."
        echo "  Stream: http://192.168.100.1:8000/stream  (IPs in config.yaml)"
        echo "  Status: sudo systemctl status pitracker"
        echo "  Logs:   journalctl -u pitracker -e"
        echo "  Manual: sudo systemctl stop pitracker && $0"
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

echo "=========================================="
echo "piStreamTracker - Camera Server"
echo "=========================================="
echo "Starting camera server..."

exec python3 camera.py "$@"
