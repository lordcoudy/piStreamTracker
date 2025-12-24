# piStreamTracker

Real-time human tracking using MoveNet pose estimation with EV3 motor control, optimized for Raspberry Pi 3B and 5.

## Architecture

The system consists of two Raspberry Pi devices connected via Ethernet:
- **Camera Pi** (`192.168.100.1`): Streams MJPEG video via HTTP
- **Tracker Pi** (`192.168.100.2`): Processes stream, runs pose detection, controls EV3 motors

## Performance Targets

| Device | Target FPS | Detection Scale | Detection Interval |
|--------|-----------|-----------------|-------------------|
| Pi 3B  | 20-24 FPS | 0.35            | 12 frames         |
| Pi 5   | 30+ FPS   | 0.5             | 6 frames          |


## Installation

### Requirements

- Python 3.12+
- Raspberry Pi 3B/5 (aarch64 or armv7l)

### Tracker Pi Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Camera Pi Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/activate

# Install picamera2 (Pi camera library)
pip install picamera2
```

### MoveNet Model

The model downloads automatically on first run from TensorFlow Hub. Manual download:

```bash
mkdir -p models
wget -O models/movenet_lightning.tflite \
    "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/lightning/tflite/float16/4.tflite"
```

## Usage

### Camera Pi (Streaming)

```bash
# Start MJPEG stream server (requires root for network setup)
sudo ./run_cam.sh
```

The camera streams at `http://192.168.100.1:8000/stream` (1280x960 resolution).

### Tracker Pi (Detection & Control)

```bash
# Raspberry Pi 3B (optimized for lower resources)
sudo ./run_tracker.sh --pi3

# Raspberry Pi 5 (higher quality detection)
sudo ./run_tracker.sh --pi5

# Custom settings
sudo python3 tracker.py --url http://192.168.100.1:8000/stream
```

> **Note**: Root privileges are required to configure the network interface.

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | `http://192.168.100.1:8000/stream` | Video stream URL |
| `--output-dir` | `recordings` | Output directory for recordings |
| `--detection-interval` | `8` | Run detection every N frames (higher=faster) |
| `--process-scale` | `0.5` | Frame scale for detection (0.2-1.0) |
| `--confidence-threshold` | `0.5` | Detection confidence threshold |
| `--keypoint-threshold` | `0.3` | Keypoint visibility threshold |
| `--movenet-model` | auto | Path to MoveNet TFLite model |
| `--movenet-threads` | auto | Inference threads (default: half CPU cores) |
| `--no-display` | false | Headless mode (no video window) |
| `--no-auto-record` | false | Disable auto-recording on start |
| `--verbose` | false | Enable verbose logging |

### EV3 Motor Control Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ev3-deadzone-x` | `90` | Horizontal deadzone (pixels) |
| `--ev3-deadzone-y` | `90` | Vertical deadzone (pixels) |
| `--ev3-speed-factor` | `1.0` | Motor speed multiplier (0.1-2.0) |
| `--ev3-max-speed` | `50` | Maximum motor speed (1-100) |
| `--ev3-invert-x` | false | Invert horizontal direction |
| `--ev3-invert-y` | false | Invert vertical direction |
| `--ev3-cooldown` | `0.5` | Motor command cooldown (seconds) |

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Toggle recording |
| `s` | Take screenshot |
| `d` | Force detection reset |
| `e` | Toggle EV3 connection |

## Performance Tuning

### For Higher FPS (less accurate tracking)

```bash
python3 tracker.py \
    --detection-interval 15 \
    --process-scale 0.3 \
    --keypoint-threshold 0.4
```

### For Better Accuracy (lower FPS)

```bash
python3 tracker.py \
    --detection-interval 4 \
    --process-scale 0.6 \
    --confidence-threshold 0.6
```

### Headless Mode (Maximum FPS)

```bash
python3 tracker.py --no-display --no-auto-record
```

## EV3 Motor Setup

Motors should be connected to the EV3 brick:
- **Port A**: Horizontal (pan) motor
- **Port B**: Vertical (tilt) motor

The EV3 brick connects to the Tracker Pi via USB.

## Troubleshooting

### Low FPS

1. Reduce `--process-scale` (try 0.3)
2. Increase `--detection-interval` (try 15)
3. Run with `--no-display` to check if display is bottleneck
4. Ensure CPU governor is set to `performance`:
   ```bash
   sudo cpufreq-set -g performance
   ```

### Detection Not Working

1. Ensure MoveNet model exists in `models/` directory (auto-downloads on first run)
2. Lower `--confidence-threshold` (try 0.4)
3. Verify stream URL is accessible: `curl http://192.168.100.1:8000/stream`

### EV3 Not Connecting

1. Check USB connection between EV3 and Tracker Pi
2. Ensure EV3 is powered on
3. Verify `ev3-dc` is installed: `pip show ev3-dc`
4. Check motor ports (A=horizontal, B=vertical)

### Network Issues

1. Verify both Pis are on the same network segment
2. Check IP assignments: Camera Pi should be `192.168.100.1`, Tracker Pi should be `192.168.100.2`
3. Test connectivity: `ping 192.168.100.1` from Tracker Pi

## Logging

The tracker logs to both console and `tracker.log` file. Position shifts are logged to a separate file with batched writes for performance.

