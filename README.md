# piStreamTracker

Follow-cam for a lecturer: a Pi 3B+ streams MJPEG, a Pi 5 runs MoveNet + OpenCV tracking and drives an EV3 pan/tilt head. Control it from a local web UI or a display on the tracker Pi.

Do not run the Python processes as root. Assign Ethernet IPs once, then start the apps as a normal user (EV3 via udev).

## Hardware

| Device | Role | Address |
|--------|------|---------|
| Raspberry Pi 3B+ | Camera + MJPEG (and optional H.264 record) | 192.168.100.1 |
| Raspberry Pi 5 | Detection, web UI, EV3 | 192.168.100.2 |
| EV3 brick (USB) | Port A pan, port B tilt | — |

Direct Ethernet (or a switch) between the two Pis.

```
Pi 3B+  --MJPEG /record/*-->  Pi 5  --USB-->  EV3 (A=pan, B=tilt)
192.168.100.1                 192.168.100.2
camera.py                     tracker.py  or  web.py
```

## Features

- MoveNet Lightning (async) + MOSSE/KCF/CSRT between detects
- Aims at the upper half of the box (head/torso)
- EV3 pan/tilt with deadzone, home-hold after Reset, D-pad
- Web UI: overlay or raw-camera proxy, zoom, horizon preview, recordings
- Local H.264 (ffmpeg, hardware when available) or camera-side record
- Stream reconnects after a camera reboot; status shows `stream_lost`
- Optional Bearer token on every `/record/*` operation; cap on concurrent `/stream` clients
- Camera-mode recordings remain downloadable/deletable from the tracker UI; local fallback is automatic

## Setup

Install from this git repo (not a `.deb` or PyPI package). `pip install -e .` only
installs Python deps; it does not configure ethernet, udev, or systemd.

**Pi 3B+**

```bash
git clone https://github.com/lordcoudy/piStreamTracker.git
cd piStreamTracker
./setup.sh --camera
```

**Pi 5**

```bash
git clone https://github.com/lordcoudy/piStreamTracker.git
cd piStreamTracker
./setup.sh --tracker
```

`./setup.sh --camera` / `--tracker` is non-interactive: apt + venv + pip, ethernet
IPs from `config.yaml` via NetworkManager (Bookworm/Trixie), EV3 udev on the tracker,
and `pitracker.service` enabled. Wi-Fi is left alone; ethernet is `ipv4.never-default`
so SSH/internet on wlan0 keep the default route.

```bash
./setup.sh --camera --skip-network
./setup.sh --tracker --skip-service --skip-udev
./setup.sh --tracker --dry-run
./setup.sh status
./setup.sh network --role camera
```

For a custom configuration, pass `--config "/absolute/path/site config.yaml"` to
the installer. Generated services pass that resolved path to either `camera.py`
or `web.py`; relative output directories still use the checkout as their base.

Configure persistent Ethernet once, or repair it later, from the project directory
on each Pi without wrapping the whole script in sudo:

```bash
./run_cam.sh --configure-network      # 192.168.100.1 from config.yaml
./run_tracker.sh --configure-network  # 192.168.100.2 from config.yaml
```

These commands save the static address in the network backend's configuration;
with NetworkManager they also enable automatic connection at boot and when the
Ethernet cable is plugged in. A manual `ip address replace ...` / `ip link set ...`
only changes the running system and is lost at reboot. After rebooting each Pi,
check `ip -4 address show dev eth0` for its address above.

The installer sudoes only `nmcli` / networkd / systemd / udev. Do not run Python as root.

Scripts accept `venv/` (what `setup.sh` creates) or `.venv/`, can be launched from any
working directory, and accept tracker flags in any order.

## Run

Camera Pi:

```bash
./run_cam.sh
```

Stream: `http://192.168.100.1:8000/stream`

Tracker Pi:

```bash
./run_tracker.sh --pi5     # OpenCV window
./run_tracker.sh --web     # http://192.168.100.2:5000
./run_tracker.sh --fast    # preset: fast
./run_tracker.sh --quality # preset: quality
```

`--web` forwards `--preset` (default `pi5`) to `web.py`. Presets live in `config.yaml`.

Laptop (no Pi / no EV3):

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,desktop]"
.venv/bin/python scripts/fake_mjpeg.py          # :8000
.venv/bin/python web.py --host 127.0.0.1 --port 5000 --no-ev3 \
  --url http://127.0.0.1:8000/stream
```

The `desktop` extra installs LiteRT 2.3 on supported desktops, or TensorFlow 2.16
on Intel macOS. The detector tries TFLite Runtime, LiteRT, then TensorFlow. Pi
installation keeps the existing TFLite Runtime dependency. MoveNet downloads on
first use; the synthetic stream exercises controls and transport, not pose quality.

## Web UI

`http://192.168.100.2:5000` (or the `--host` you passed)

| Control | What it does |
|---------|----------------|
| Start / Stop | Connects to the camera stream, then tracks. Start fails if the stream is down. |
| Reset | Clears the tracker and homes the EV3; auto-track is held for `ev3.home_hold` seconds |
| Record / Screenshot | Local files under `tracker.output_dir`, or camera Pi if `recording_mode: camera`; an unreachable camera recorder falls back locally |
| Tracking overlay | Off = proxy the camera MJPEG (lowest delay). On = annotated preview at `preview_max_fps` / `preview_max_edge` |
| Auto-Level | Rotate the **preview** to level shoulders/hips; motors still use the raw box |
| D-pad / Zoom | Manual pan/tilt; digital zoom on the preview only |
| Recordings | Lists, downloads, and deletes `.mp4` / `.avi` / stills from both Pis. Active recordings cannot be deleted. |

The tracking status dot turns yellow while the stream is reconnecting (`stream_lost`).

Stop immediately releases tracking resources and makes one camera-stop request.
If the camera does not return JSON `recording: false`, the UI keeps a warning and
shows **Retry camera stop**. Start stays disabled until confirmation; each click
retries once, and status polling never retries. A failed **Stop Rec** while tracking
continues can also be retried with **Stop Rec**. Keep the web process running to
retain this recovery state; it is held in memory.

Local encoder failures clear the recording indicator and remain visible until a
new recording starts successfully. Files stay protected from deletion until the
writer closes, including while tracking is stopping.

By default the UI and camera server bind to `tracker_ip` / `camera_ip`. Set `host: "0.0.0.0"` only if you need another NIC; anyone who can reach the port can move motors and delete recordings.
The web server rejects cross-origin browser control requests and sends clickjacking/MIME-sniffing
protections, but it is still a trusted-LAN control surface rather than an Internet-facing service.

## Configuration

All defaults are in `config.yaml`. CLI flags override a named `--preset`, which overrides the file.

```yaml
network:
  camera_ip: "192.168.100.1"
  tracker_ip: "192.168.100.2"

camera:
  port: 8000
  framerate: 30
  resolution: {width: 1280, height: 960}
  token: null              # Bearer for /record/* ; empty = open LAN
  max_stream_clients: 4

tracker:
  recording_mode: local    # or camera (files on the 3B+)
  recording_encoder: auto  # h264_v4l2m2m | libx264 | mjpg
  recording_fps: 30        # capped to camera.framerate
  detection:
    interval: 10
    scale: 0.4
    confidence: 0.5

ev3:
  deadzone: {x: 90, y: 90}
  max_speed: 50
  home_hold: 3.0
  ports: {pan: "a", tilt: "b"}

web:
  host: null               # null = tracker_ip
  port: 5000
  overlay: true
  preview_quality: 70
  preview_max_edge: 640
  preview_max_fps: 15

presets:
  pi5:     {detection_interval: 10, process_scale: 0.4, movenet_threads: 4}
  fast:    {detection_interval: 10, process_scale: 0.35, movenet_threads: 4}
  quality: {detection_interval: 4,  process_scale: 0.6,  movenet_threads: 4}
```

Configuration is validated at startup; unsafe values such as a zero detection interval, invalid
ports, unknown recording backends, or malformed section types fail immediately with a clear error.
Set `PISTREAM_CAMERA_TOKEN` to override `camera.token` without storing the secret in `config.yaml`.

## CLI

```bash
python tracker.py --help
python web.py --help
python -m pistream.install --help
```

Shared flags: `--config`, `--url`, `--output-dir`, `--detection-interval`, `--process-scale`, `--confidence`, `--movenet-threads`, `--no-ev3`, `--preset`.

Tracker-only: `--no-display`, `--auto-record`.

Web-only: `--host`, `--port`.

Camera: `--config` (loaded before recording directories and server settings are created).

### OpenCV window keys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Toggle recording |
| `s` | Screenshot |
| `d` | Reset tracker + home EV3 |
| `e` | Connect / disconnect EV3 |

Keyboard controls remain available during a stream outage. Screenshots use the
last displayed frame; before the first frame, the CLI reports that none is available.
After an unconfirmed camera stop, tracking, capture and motors stop while the
window stays open for **R** to retry once or **Q** to force quit. Headless terminals
accept `retry` or `quit`; an unavailable OpenCV window also falls back to terminal
input. Confirmation exits normally; force quit, terminal EOF,
or headless operation without a terminal exits nonzero with recovery instructions.
An underlying tracking error also exits nonzero after recovery.

## Tests and CI

```bash
.venv/bin/pip install -e ".[dev,desktop]"
.venv/bin/pytest
.venv/bin/ruff check pistream tests tracker.py web.py camera.py ev3_usb.py
```

No Pi or EV3 required. GitHub Actions runs the same on Python 3.11 and 3.12.
Raspberry Pi OS Bookworm's system `python3` is 3.11.

Optional browser checks use an existing Node/Playwright installation: run
`.venv/bin/python tests/web_ui_fixture.py` and, in another terminal,
`node tests/browser_ui.cjs`. `PLAYWRIGHT_MODULE` and `PLAYWRIGHT_EXECUTABLE` can
select an existing installation and Chromium executable. The fixture uses mocked
hardware and checks manual retry, warnings, controls and responsive layout.

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| Preview lag | Turn **Tracking overlay** off, or lower `web.preview_max_edge` / `preview_quality` / `preview_max_fps` |
| Low FPS | `--preset fast`, or raise `detection.interval` / lower `detection.scale` |
| No detection | Lower `confidence` |
| Stream dies mid-talk | Wait: capture reconnects with backoff. Yellow status = `stream_lost`. Do not need a second Start unless you clicked Stop |
| EV3 not found | USB cable, `ev3-dc` installed, udev rule; re-plug the brick / re-login after `plugdev` |
| EV3 permission denied | `setup.sh` udev (`idVendor=0694`); user in `plugdev` |
| Camera recordings unavailable | Check the camera server and token. The UI still shows local screenshots/fallback recordings and reports that camera files are unavailable. |
| Camera stop unconfirmed | Restore camera connectivity/token, then explicitly retry Stop in the web UI or R / `retry` in CLI recovery. A forced exit leaves the stop unresolved; confirm manually with `/record/stop`. |
| Local recording failed | Check encoder logs, storage and ffmpeg availability. The error remains visible until a successful recording restart. |
| Bind / “address already in use” | Something else on 5000/8000, or `host` is an IP that is not assigned yet. After setup, `pitracker.service` may already own the port: `sudo systemctl status pitracker` |
| Camera busy / pipeline handler in use | Another process holds libcamera. After `./setup.sh --camera`, that is `pitracker.service` (stream already at `http://192.168.100.1:8000/stream`). `sudo systemctl status pitracker`. To run `./run_cam.sh` yourself: `sudo systemctl stop pitracker` |
| Network | `ping 192.168.100.1`. `./setup.sh status`. Bookworm/Trixie uses NetworkManager; install `network-manager` or pass `--skip-network`. Remove leftover `/etc/systemd/network/10-eth0.network` from older setup.sh if it fights NM. |

## Layout

```
piStreamTracker/
├── config.yaml          # Single source of settings
├── tracker.py           # Display / CLI entry
├── web.py               # Flask entry
├── camera.py            # Camera Pi server
├── ev3_usb.py           # Shim → pistream.ev3_usb
├── setup.sh             # Bootstrap apt/venv; exec python -m pistream.install
├── run_cam.sh
├── run_tracker.sh
├── pistream/            # Application package
│   ├── install.py       # Installer CLI (network, udev, systemd)
│   ├── capture.py       # Threaded MJPEG capture + reconnect
│   ├── detect.py        # MoveNet + async worker
│   ├── track.py         # HumanTracker, MOSSE, CLI
│   ├── motors.py        # EV3 pan/tilt + home hold
│   ├── record.py        # ffmpeg / MJPG writer
│   ├── web_app.py       # Routes
│   ├── recordings.py    # Listing + path safety
│   ├── preview.py       # Preview gate / scale / fps cap
│   ├── templates/index.html
│   └── static/
├── tests/
├── scripts/fake_mjpeg.py
└── models/              # MoveNet (downloaded, gitignored)
```

## License

MIT
