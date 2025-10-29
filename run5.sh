#!/usr/bin/env bash
# run eth init and cam server
if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (e.g.,: sudo $0)" >&2
  exit 1
fi

# Read from config.yaml
CONFIG_FILE="config.yaml"
IP_ADDR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['network']['tracker_pi']['ip'])")
DEVICE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['network']['tracker_pi']['device'])")

ip addr flush dev $DEVICE
ip addr add $IP_ADDR dev $DEVICE
ip link set $DEVICE up
exec python "tracker/main.py"