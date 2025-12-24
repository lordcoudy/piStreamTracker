#!/usr/bin/env bash
# usage: ./run_cam.sh
if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (e.g.,: sudo $0)" >&2
  exit 1
fi

echo "==================================="
echo "Cleaning and setting up network interface eth0"
ip addr flush dev eth0
echo "Assigning IP address to eth0: 192.168.100.1/24"
ip addr add 192.168.100.1/24 dev eth0
echo "Bringing up eth0"
ip link set eth0 up
echo "==================================="

echo "Starting camera application..."
set -e
exec python "camera.py"