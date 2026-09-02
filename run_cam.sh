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
