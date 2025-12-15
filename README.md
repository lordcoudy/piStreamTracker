# Optimized Human Tracker for Raspberry Pi

Real-time human tracking using MoveNet pose estimation, optimized for Raspberry Pi 3B and 5.

## Performance Targets

| Device | Target FPS | Detection Scale | Detection Interval |
|--------|-----------|-----------------|-------------------|
| Pi 3B  | 20-24 FPS | 0.35            | 12 frames         |
| Pi 5   | 30+ FPS   | 0.5             | 6 frames          |

## Key Optimizations

1. **Threaded Video Capture**: Non-blocking frame reads eliminate I/O stalls
2. **Pre-allocated Buffers**: Reduces garbage collection pressure
3. **MOSSE Tracker**: Fastest OpenCV tracker for frame-to-frame tracking
4. **Single-pass Preprocessing**: Combines resize and color conversion
5. **Batched Logging**: Reduces file I/O overhead by 50x
6. **Optimized TFLite**: Uses tflite-runtime (faster than full TensorFlow)

## Installation

### Raspberry Pi

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install opencv-contrib-python numpy ev3-dc

# Install TFLite runtime (Pi-specific wheel)
pip install tflite-runtime

# For camera streaming (camera Pi only)
pip install picamera2
```

### Download MoveNet Model

The model downloads automatically on first run, or manually:

```bash
mkdir -p models
wget -O models/movenet_lightning.tflite \
    https://storage.googleapis.com/movenet/SinglePoseLightning.tflite
```

## Usage

### Basic Usage

```bash
# Raspberry Pi 3B
./run_tracker.sh --pi3

# Raspberry Pi 5
./run_tracker.sh --pi5

# Custom settings
python3 tracker_optimized.py --url http://192.168.100.1:8000/stream
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | `http://192.168.100.1:8000/stream` | Video stream URL |
| `--detection-interval` | `8` | Run detection every N frames (higher=faster) |
| `--process-scale` | `0.5` | Scale factor for detection (0.2-1.0) |
| `--confidence-threshold` | `0.5` | Detection confidence threshold |
| `--keypoint-threshold` | `0.3` | Keypoint visibility threshold |
| `--movenet-threads` | auto | Inference threads |
| `--no-display` | false | Headless mode (no video window) |
| `--no-auto-record` | false | Disable auto-recording |

### EV3 Motor Control Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ev3-deadzone-x` | `90` | Horizontal deadzone (pixels) |
| `--ev3-deadzone-y` | `90` | Vertical deadzone (pixels) |
| `--ev3-speed-factor` | `1.0` | Motor speed multiplier |
| `--ev3-max-speed` | `50` | Maximum motor speed (1-100) |
| `--ev3-invert-x` | false | Invert horizontal direction |
| `--ev3-invert-y` | false | Invert vertical direction |
| `--ev3-cooldown` | `0.1` | Command rate limit (seconds) |

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
python3 tracker_optimized.py \
    --detection-interval 15 \
    --process-scale 0.3 \
    --keypoint-threshold 0.4
```

### For Better Accuracy (lower FPS)

```bash
python3 tracker_optimized.py \
    --detection-interval 4 \
    --process-scale 0.6 \
    --confidence-threshold 0.6
```

### Headless Mode (Maximum FPS)

```bash
python3 tracker_optimized.py --no-display --no-auto-record
```

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

1. Ensure MoveNet model exists in `models/` directory
2. Check `--confidence-threshold` (try 0.4)
3. Verify stream URL is accessible

### EV3 Not Connecting

1. Check USB connection
2. Ensure EV3 is powered on
3. Verify `ev3-dc` is installed
4. Check motor ports (A=horizontal, B=vertical)

