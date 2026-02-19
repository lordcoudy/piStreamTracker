#!/usr/bin/env bash
#
# piStreamTracker Setup Script
# One-command installation for Raspberry Pi
#
# Usage:
#   ./setup.sh              # Full setup (auto-detects Pi type)
#   ./setup.sh --camera     # Setup for Camera Pi only
#   ./setup.sh --tracker    # Setup for Tracker Pi only
#   ./setup.sh --help       # Show help
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    piStreamTracker Setup                     ║"
    echo "║     Human Tracking with MoveNet + EV3 Motor Control          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

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
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --camera     Setup Pi 3B+ as camera server"
    echo "  --tracker    Setup Pi 5 as tracker + EV3 controller"
    echo "  --help       Show this help message"
    echo ""
    echo "Hardware Setup:"
    echo "  Pi 3B+ (Camera):  192.168.100.1  - Runs camera.py"
    echo "  Pi 5 (Tracker):   192.168.100.2  - Runs tracker.py/web.py + EV3"
    echo ""
    echo "Connect both Pis via Ethernet cable."
}

# Detect Raspberry Pi model
detect_pi() {
    if [[ -f /proc/device-tree/model ]]; then
        model=$(cat /proc/device-tree/model)
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

# Check if running on Raspberry Pi
check_platform() {
    if [[ "$(uname -m)" == "aarch64" ]] || [[ "$(uname -m)" == "armv7l" ]]; then
        return 0
    fi
    return 1
}

# Install system dependencies
install_system_deps() {
    print_step "Installing system dependencies..."

    sudo apt-get update -qq

    # Common dependencies
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv \
        libatlas-base-dev libhdf5-dev libhdf5-serial-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        libv4l-dev libxvidcore-dev libx264-dev \
        libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev \
        libpango1.0-dev libgtk-3-dev libopenblas-dev \
        python3-dev

    # Camera dependencies (if available)
    if [[ "$1" == "camera" ]] || [[ "$1" == "full" ]]; then
        sudo apt-get install -y -qq libcamera-apps python3-picamera2 || true
    fi

    print_step "System dependencies installed"
}

# Create and activate virtual environment
setup_venv() {
    print_step "Setting up Python virtual environment..."

    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi

    source venv/bin/activate
    pip install --upgrade pip -q

    print_step "Virtual environment ready"
}

# Install Python dependencies for Camera Pi
install_camera_deps() {
    print_step "Installing Camera Pi dependencies..."

    pip install -q pyyaml picamera2

    print_step "Camera dependencies installed"
}

# Install Python dependencies for Tracker Pi
install_tracker_deps() {
    print_step "Installing Tracker Pi dependencies..."

    # Core dependencies
    pip install -q \
        "opencv-contrib-python>=4.8.0,<4.10.0" \
        "numpy>=1.20.0,<2.0.0" \
        pyyaml \
        ev3-dc

    # TFLite runtime
    if check_platform; then
        pip install -q tflite-runtime || pip install -q tensorflow
    else
        pip install -q tensorflow
    fi

    print_step "Tracker dependencies installed"
}

# Install web interface dependencies
install_web_deps() {
    print_step "Installing web interface dependencies..."

    pip install -q flask

    print_step "Web dependencies installed"
}

# Download MoveNet model
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

# Configure network interface
setup_network() {
    local role=$1
    local ip=""

    if [[ "$role" == "camera" ]]; then
        ip="192.168.100.1/24"
    elif [[ "$role" == "tracker" ]]; then
        ip="192.168.100.2/24"
    else
        return
    fi

    print_step "Configuring network (${ip})..."

    # Create systemd network config
    sudo tee /etc/systemd/network/10-eth0.network > /dev/null << EOF
[Match]
Name=eth0

[Network]
Address=${ip}
EOF

    print_step "Network configured"
    print_warning "Reboot or run 'sudo systemctl restart systemd-networkd' to apply"
}

# Create systemd service for auto-start
create_service() {
    local role=$1
    local script=""
    local desc=""

    if [[ "$role" == "camera" ]]; then
        script="camera.py"
        desc="piStreamTracker Camera Server"
    elif [[ "$role" == "tracker" ]]; then
        script="web.py"
        desc="piStreamTracker Web Interface"
    else
        return
    fi

    print_step "Creating systemd service..."

    local work_dir=$(pwd)

    sudo tee /etc/systemd/system/pitracker.service > /dev/null << EOF
[Unit]
Description=${desc}
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${work_dir}
ExecStart=${work_dir}/venv/bin/python ${work_dir}/${script}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload

    print_step "Service created (pitracker.service)"
    echo "  Enable with: sudo systemctl enable pitracker"
    echo "  Start with:  sudo systemctl start pitracker"
}

# Make scripts executable
setup_permissions() {
    chmod +x *.sh 2>/dev/null || true
    chmod +x *.py 2>/dev/null || true
}

# Print final instructions
print_instructions() {
    local role=$1

    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Setup Complete!                           ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    if [[ "$role" == "camera" ]]; then
        echo "Pi 3B+ Camera Setup Complete!"
        echo ""
        echo "To start the camera server:"
        echo "  sudo ./run_cam.sh"
        echo ""
        echo "Stream URL: http://192.168.100.1:8000/stream"
        echo ""
        echo "Next: Setup Pi 5 with ./setup.sh --tracker"

    elif [[ "$role" == "tracker" ]]; then
        echo "Pi 5 Tracker Setup Complete!"
        echo ""
        echo "To start tracking:"
        echo "  sudo ./run_tracker.sh          # With display"
        echo "  sudo ./run_tracker.sh --web    # Web interface only"
        echo ""
        echo "Web interface: http://192.168.100.2:5000"
        echo ""
        echo "Make sure Pi 3B+ camera is running first!"

    else
        echo "Full Setup Complete!"
        echo ""
        echo "For Raspberry Pi deployment:"
        echo "  Pi 3B+ (Camera):  ./setup.sh --camera"
        echo "  Pi 5 (Tracker):   ./setup.sh --tracker"
    fi

    echo ""
    echo "Configuration: config.yaml"
    echo ""
}

# Main setup function
main() {
    local mode="full"
    local include_web=true

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --camera)
                mode="camera"
                include_web=false
                shift
                ;;
            --tracker)
                mode="tracker"
                shift
                ;;
            --web)
                include_web=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    print_header

    # Detect Pi model
    pi_model=$(detect_pi)
    if [[ "$pi_model" != "unknown" ]]; then
        print_step "Detected: Raspberry Pi ${pi_model#pi}"
    fi

    # Install system dependencies
    if check_platform; then
        install_system_deps "$mode"
    fi

    # Setup virtual environment
    setup_venv

    # Install Python dependencies based on mode
    case $mode in
        camera)
            install_camera_deps
            ;;
        tracker)
            install_tracker_deps
            if [[ "$include_web" == true ]]; then
                install_web_deps
            fi
            download_model
            ;;
        full)
            install_tracker_deps
            install_web_deps
            download_model
            ;;
    esac

    # Setup permissions
    setup_permissions

    # Optional: setup network and service
    if check_platform && [[ "$mode" != "full" ]]; then
        read -p "Configure network interface? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            setup_network "$mode"
        fi

        read -p "Create systemd service for auto-start? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            create_service "$mode"
        fi
    fi

    # Print instructions
    print_instructions "$mode"
}

# Run main
main "$@"
