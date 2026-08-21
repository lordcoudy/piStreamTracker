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

# Optional one-shot network config (do not use if SSH is on eth0)
if [[ "${1:-}" == "--configure-network" ]]; then
    shift
    if [[ $EUID -ne 0 ]]; then
        echo "Network config requires root: sudo $0 --configure-network" >&2
        exit 1
    fi
    echo "Configuring eth0 as 192.168.100.1/24 (no flush of other addresses)..."
    ip addr add 192.168.100.1/24 dev eth0 2>/dev/null || true
    ip link set eth0 up
    echo "Network configured. Re-run $0 as a normal user to start the camera server."
    exit 0
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
