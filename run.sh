#!/usr/bin/env bash
# run eth init and cam server
if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (e.g.,: sudo $0)" >&2
  exit 1
fi

ip addr flush dev eth0
ip addr add 192.168.100.1/24 dev eth0
ip link set eth0 up
exec python "cams/main.py"