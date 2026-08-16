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
git clone https://github.com/lordcoudy/piStreamTracker.git
cd piStreamTracker
./setup.sh --camera
```

**On Pi 5 (Tracker):**
```bash
git clone https://github.com/lordcoudy/piStreamTracker.git
cd piStreamTracker
./setup.sh --tracker
```

Assign the static Ethernet IPs once (systemd-networkd during setup, or a one-shot `sudo ./run_cam.sh --configure-network` / `sudo ./run_tracker.sh --configure-network`). Do **not** run the Python processes as root.

### Run

**1. Start Camera (Pi 3B+):**
```bash
./run_cam.sh
```

**2. Start Tracker (Pi 5):**
```bash
# With display
./run_tracker.sh --pi5

# Or with web interface (headless)
./run_tracker.sh --web
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
  host: null            # bind tracker_ip; use 0.0.0.0 to listen on all interfaces
  port: 5000
  overlay: true         # false = proxy raw camera MJPEG (lowest preview delay)
  preview_quality: 70
  preview_max_edge: 640
  preview_max_fps: 15
```

By default the camera server and web UI bind only to `camera_ip` / `tracker_ip`. Set `host: "0.0.0.0"` if you need them on Wi-Fi as well (anyone on that LAN can then start motors and delete recordings).

## Web Interface

Access at `http://192.168.100.2:5000`

- Live video stream with overlay (or raw camera proxy)
- Start/stop tracking
- Recording controls
- EV3 motor adjustments
- Detection tuning

Turn **Tracking overlay** off to watch the camera MJPEG directly (no JPEG re-encode on the Pi 5). That is the lowest-delay preview. Overlay preview is capped at `preview_max_fps` and `preview_max_edge`.

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
./run_tracker.sh
```

**Fast mode (higher FPS):**
```bash
./run_tracker.sh --fast
```

**Quality mode (better accuracy):**
```bash
./run_tracker.sh --quality
```

**Headless (maximum FPS):**
```bash
python tracker.py --no-display --preset fast
```

`--fast` / `--quality` / `--pi5` on `run_tracker.sh` select the matching `presets:` block in `config.yaml`. `--web` forwards `--preset` to `web.py`.

## Tests

```bash
python3 -m pytest
# or
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

Most tests do not need a Pi or EV3. The package/web template test needs Flask. CI runs ruff + pytest on Python 3.12.

## Troubleshooting

**Preview lag:** Turn off Tracking overlay, or lower `web.preview_max_edge` / `web.preview_quality`.

**Low FPS:** Reduce `--process-scale` or increase `--detection-interval`

**No detection:** Lower `--confidence` threshold

**EV3 not connecting:** Check USB connection, ensure `ev3-dc` is installed

**Network issues:** Verify IPs with `ping 192.168.100.1`. Scripts no longer flush `eth0` on start.

**EV3 permission denied:** Re-plug the brick after setup so the udev rule applies, or add your user to `plugdev`.

## Project Structure

```
piStreamTracker/
├── config.yaml         # All settings
├── tracker.py          # CLI entry (display tracker)
├── web.py              # CLI entry (Flask UI)
├── camera.py           # Camera Pi stream server
├── pistream/           # Application package
│   ├── config.py
│   ├── capture.py
│   ├── detect.py
│   ├── motors.py
│   ├── record.py
│   ├── track.py
│   ├── web_app.py
│   ├── templates/index.html
│   └── static/{app.js,style.css}
├── tests/
├── setup.sh
├── run_tracker.sh
├── run_cam.sh
└── models/             # MoveNet (auto-download, gitignored)
```

## License

MIT
