# piStreamTracker

Real-time human tracking with MoveNet pose detection and EV3 motor control.

## Hardware Setup

| Device | Role | IP Address |
|--------|------|------------|
| **Raspberry Pi 3B+** | Camera streaming | 192.168.100.1 |
| **Raspberry Pi 5** | Detection + EV3 control | 192.168.100.2 |
| **EV3 Brick** | Motor control (USB to Pi 5) | - |

Connect the two Pis via Ethernet cable (direct connection or switch).

## Features

- **MoveNet Lightning** pose detection on Pi 5
- **EV3 motor control** for pan/tilt camera tracking
- **Web interface** for remote control and monitoring
- **MJPEG streaming** from Pi 3B+ to Pi 5
- **One-command setup** with automatic configuration

## Quick Start

### Setup

**On Pi 3B+ (Camera):**
```bash
git clone https://github.com/yourusername/piStreamTracker.git
cd piStreamTracker
./setup.sh --camera
```

**On Pi 5 (Tracker):**
```bash
git clone https://github.com/yourusername/piStreamTracker.git
cd piStreamTracker
./setup.sh --tracker
```

### Run

**1. Start Camera (Pi 3B+):**
```bash
sudo ./run_cam.sh
```

**2. Start Tracker (Pi 5):**
```bash
# With display
sudo ./run_tracker.sh --pi5

# Or with web interface (headless)
sudo ./run_tracker.sh --web
```

## Architecture

```
┌─────────────────┐  Ethernet   ┌─────────────────┐
│  Pi 3B+ Camera  │   MJPEG     │    Pi 5         │
│  192.168.100.1  │────────────►│  192.168.100.2  │
│                 │             │                 │
│  • Pi Camera    │             │  • MoveNet      │
│  • Stream Server│             │  • OpenCV       │
└─────────────────┘             │  • Web UI       │
                                └────────┬────────┘
                                         │ USB
                                ┌────────▼────────┐
                                │    EV3 Brick    │
                                │  Port A: Pan    │
                                │  Port B: Tilt   │
                                └─────────────────┘
```

## Configuration

All settings in `config.yaml`:

```yaml
network:
  camera_ip: "192.168.100.1"
  tracker_ip: "192.168.100.2"

camera:
  port: 8000
  resolution: {width: 1280, height: 960}

tracker:
  detection:
    interval: 8         # Frames between detections
    scale: 0.5          # Processing scale (lower = faster)
    confidence: 0.5     # Detection threshold

ev3:
  enabled: true
  deadzone: {x: 90, y: 90}
  max_speed: 50
  ports: {pan: "a", tilt: "b"}

web:
  port: 5000
```

## Web Interface

Access at `http://192.168.100.2:5000`

- Live video stream with overlay
- Start/stop tracking
- Recording controls
- EV3 motor adjustments
- Detection tuning

## Command-Line Options

```bash
python tracker.py [OPTIONS]

# Stream
--url URL              Video stream URL
--config FILE          Config file path

# Detection
--detection-interval N   Frames between detections (default: 8)
--process-scale N        Scale factor 0.2-1.0 (default: 0.5)
--confidence N           Confidence threshold (default: 0.5)
--movenet-threads N      Inference threads

# Display
--no-display            Headless mode
--auto-record           Auto-start recording
--no-ev3                Disable EV3
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Toggle recording |
| `s` | Screenshot |
| `d` | Reset detection |
| `e` | Toggle EV3 |

## Performance Tuning (Pi 5)

**Default (balanced):**
```bash
sudo ./run_tracker.sh
```

**Fast mode (higher FPS):**
```bash
sudo ./run_tracker.sh --fast
```

**Quality mode (better accuracy):**
```bash
sudo ./run_tracker.sh --quality
```

**Headless (maximum FPS):**
```bash
python tracker.py --no-display
```

## Troubleshooting

**Low FPS:** Reduce `--process-scale` or increase `--detection-interval`

**No detection:** Lower `--confidence` threshold

**EV3 not connecting:** Check USB connection, ensure `ev3-dc` is installed

**Network issues:** Verify IPs with `ping 192.168.100.1`

## Project Structure

```
piStreamTracker/
├── config.yaml       # All settings
├── tracker.py        # Main tracking application
├── camera.py         # Camera streaming server
├── web.py            # Web interface
├── ev3_usb.py        # EV3 communication wrapper
├── setup.sh          # One-command setup
├── run_tracker.sh    # Run tracker
├── run_cam.sh        # Run camera server
├── requirements.txt  # Python dependencies
└── models/           # MoveNet model (auto-downloads)
```

## License

MIT
