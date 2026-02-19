#!/usr/bin/env bash
#
# Run piStreamTracker - Camera Pi
#
# Usage:
#   sudo ./run_cam.sh
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
ip addr add 192.168.100.1/24 dev eth0
ip link set eth0 up
echo "Network: 192.168.100.1/24"

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
