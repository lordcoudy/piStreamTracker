# piStreamTracker

Lightweight tracker for face (or body) tracking using Raspberry Pi camera, OpenCV and Lego EV3.

## Requirements

- Raspberry Pi (or any Linux system) for tracker
- Raspberry Pi with camera for video feed
- Lego EV3 brick with motors (at least 2)
- Python 3.8+
- Network connection between RPIs

## Installation

```bash
git clone <repo-url> piStreamTracker
cd piStreamTracker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to apply new settings.

## Usage

```bash
./run.sh # on camera Pi
./run5.sh # on tracker Pi
```
