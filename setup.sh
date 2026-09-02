#!/usr/bin/env bash
#
# piStreamTracker bootstrap + installer
#
# Usage:
#   ./setup.sh --camera
#   ./setup.sh --tracker
#   ./setup.sh --camera --skip-network
#   ./setup.sh --tracker --dry-run
#   ./setup.sh status
#   ./setup.sh network --role camera
#

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

show_help() {
    cat <<'EOF'
Usage: ./setup.sh [OPTIONS] [COMMAND] [ARGS...]

Roles:
  --camera     Pi 3B+ camera server (192.168.100.1 from config.yaml)
  --tracker    Pi 5 tracker + EV3 (192.168.100.2 from config.yaml)

Commands (default: install):
  install      Full automatic: network + udev (tracker) + systemd enable --now
  network      Configure ethernet IPv4 via NetworkManager (or systemd-networkd)
  service      Write / enable pitracker.service
  udev         EV3 USB udev rule
  status       Show IPs, venv, udev, service

Flags (forwarded to python -m pistream.install):
  --skip-network --skip-service --skip-udev --dry-run --interface IFACE

Examples:
  ./setup.sh --camera
  ./setup.sh --tracker
  ./setup.sh --camera --skip-network
  ./setup.sh --tracker --dry-run
  ./setup.sh status
  ./setup.sh network --role camera

Do not run as root. The installer uses sudo only for nmcli/tee/systemctl/usermod.
EOF
}

detect_pi() {
    if [[ -f /proc/device-tree/model ]]; then
        local model
        model=$(tr -d '\0' < /proc/device-tree/model)
        if [[ "$model" == *"Pi 5"* ]]; then
            echo "pi5"
        elif [[ "$model" == *"Pi 4"* ]]; then
            echo "pi4"
        elif [[ "$model" == *"Pi 3"* ]]; then
            echo "pi3"
        else
            echo "unknown"
        fi
    else
        echo "unknown"
    fi
}

check_platform() {
    local arch
    arch=$(uname -m)
    [[ "$arch" == "aarch64" || "$arch" == "armv7l" ]]
}

install_system_deps() {
    local role=$1
    print_step "Installing system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv \
        libopenblas-dev libhdf5-dev libhdf5-serial-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        libv4l-dev libxvidcore-dev libx264-dev \
        libfontconfig1-dev libcairo2-dev libgdk-pixbuf-2.0-dev \
        libpango1.0-dev libgtk-3-dev \
        python3-dev \
        ffmpeg \
        network-manager
    if [[ "$role" == "camera" ]]; then
        sudo apt-get install -y -qq \
            libcamera-apps \
            libcamera-dev \
            python3-libcamera \
            python3-picamera2 || true
    fi
    print_step "System dependencies installed"
}

setup_venv() {
    print_step "Setting up Python virtual environment..."
    if [[ ! -d "venv" ]]; then
        python3 -m venv --system-site-packages venv
    fi
    print_step "Virtual environment ready"
}

install_python_deps() {
    local role=$1
    # shellcheck disable=SC1091
    source venv/bin/activate
    pip install --upgrade pip -q
    if [[ "$role" == "camera" ]]; then
        print_step "Installing Camera Pi Python deps (no OpenCV)..."
        pip install -q -e . --no-deps
        pip install -q 'pyyaml>=6.0'
    else
        print_step "Installing Tracker Python deps..."
        pip install -q -e .
    fi
}

download_model() {
    print_step "Downloading MoveNet model..."
    mkdir -p models
    if [[ ! -f "models/movenet_lightning.tflite" ]]; then
        wget -q -O models/movenet_lightning.tflite \
            "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/lightning/tflite/float16/4.tflite" \
            || print_warning "Model download failed - will auto-download on first run"
    fi
    print_step "Model ready"
}

array_contains() {
    local needle=$1
    shift
    local item
    for item in "$@"; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

ROLE=""
COMMAND=""
DRY_RUN=0
FORWARD=()

if [[ $# -eq 0 ]]; then
    COMMAND="install"
    ROLE="auto"
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --camera)
            ROLE="camera"
            shift
            ;;
        --tracker)
            ROLE="tracker"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            FORWARD+=("$1")
            shift
            ;;
        install|network|service|udev|status)
            if [[ -z "$COMMAND" ]]; then
                COMMAND="$1"
            else
                FORWARD+=("$1")
            fi
            shift
            ;;
        *)
            FORWARD+=("$1")
            shift
            ;;
    esac
done

COMMAND=${COMMAND:-install}

if [[ $EUID -eq 0 ]]; then
    print_error "Do not run setup as root; it uses sudo only for the required system changes."
    exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    ver=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)
    print_error "Python 3.11+ required (found ${ver}). Raspberry Pi OS Bookworm's python3 is 3.11."
    exit 1
fi

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    piStreamTracker Setup                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

pi_model=$(detect_pi)
if [[ "$pi_model" != "unknown" ]]; then
    print_step "Detected: Raspberry Pi ${pi_model#pi}"
fi

pip_role="$ROLE"
if [[ -z "$pip_role" || "$pip_role" == "auto" ]]; then
    if [[ "$pi_model" == "pi3" ]]; then
        pip_role="camera"
    else
        pip_role="tracker"
    fi
fi

if check_platform; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        print_warning "Dry-run: skipping apt-get"
    else
        install_system_deps "$pip_role"
    fi
fi

setup_venv
if [[ "$DRY_RUN" -eq 1 ]]; then
    print_warning "Dry-run: skipping pip install (venv must already import pistream)"
else
    install_python_deps "$pip_role"
    if [[ "$pip_role" != "camera" ]]; then
        download_model
    fi
fi

PY="$SCRIPT_DIR/venv/bin/python"
if [[ ! -x "$PY" ]]; then
    print_error "venv python missing"
    exit 1
fi

PY_ARGS=("$COMMAND")
if [[ "$COMMAND" == "install" || "$COMMAND" == "network" || "$COMMAND" == "service" ]]; then
    if [[ -n "$ROLE" ]] && ! array_contains --role "${FORWARD[@]+"${FORWARD[@]}"}"; then
        PY_ARGS+=(--role "$ROLE")
    elif [[ -z "$ROLE" && "$COMMAND" == "install" ]]; then
        PY_ARGS+=(--role auto)
    fi
fi
if [[ ${#FORWARD[@]} -gt 0 ]]; then
    PY_ARGS+=("${FORWARD[@]}")
fi

print_step "Running python -m pistream.install ${PY_ARGS[*]}"
exec "$PY" -m pistream.install "${PY_ARGS[@]}"
