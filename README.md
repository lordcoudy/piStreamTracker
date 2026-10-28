# piStreamTracker

Follow-cam for a lecturer: run an IMX219 CSI camera, MoveNet + OpenCV tracking, a web UI and two SG90 pan/tilt servos on one Raspberry Pi 4, or use the Pi 3B+ IMX219 camera → Pi 5 tracker → USB EV3 setup.

Run application Python processes as an ordinary user. The standalone setup keeps existing Ethernet/Wi-Fi settings; the two-Pi setup assigns its Ethernet addresses once (EV3 access uses udev).

## Two-Pi hardware

| Device | Role | Address |
|--------|------|---------|
| Raspberry Pi 3B+ | IMX219 camera + MJPEG (and optional H.264 record) | 192.168.100.1 |
| Raspberry Pi 5 | Detection, web UI, EV3 | 192.168.100.2 |
| EV3 brick (USB) | Port A pan, port B tilt | — |

Direct Ethernet (or a switch) between the two Pis.
The Pi 3B+ and Pi 4 use the same IMX219 camera module and share the
`tuning/imx219-160-no-alsc.json` profile through `camera.tuning_file` in
`config.yaml` and `config.pi4.yaml`. The profile preserves the camera correction
from the supplied Pi 4 project.

```
Pi 3B+  --MJPEG /record/*-->  Pi 5  --USB-->  EV3 (A=pan, B=tilt)
192.168.100.1                 192.168.100.2
camera.py                     tracker.py  or  web.py
```

## Features

- MoveNet Lightning (async) + MOSSE/KCF/CSRT between detects
- Aims at the upper half of the box (head/torso)
- EV3 pan/tilt with deadzone, home-hold after Reset, D-pad
- GPIO SG90 commands through local `pigpiod`, independent calibration and mechanical limits, speed caps, and a fresh-frame watchdog
- Web UI: overlay or raw-camera proxy, zoom, horizon preview, recordings
- Local H.264 (ffmpeg, hardware when available) or camera-side H.264/MP4 via Picamera2/PyavOutput
- Stream reconnects after a camera reboot; status shows `stream_lost`
- Optional Bearer token on every `/record/*` operation; cap on concurrent `/stream` clients
- Camera-mode recordings remain downloadable/deletable from the tracker UI; local fallback is automatic

## One Raspberry Pi 4: assembly and first run

Use **Raspberry Pi OS Bookworm 64-bit with its system Python 3.11**, one CSI
camera and two positional SG90 servos. The current `config.pi4.yaml` combines
the supplied IMX219 160-degree camera profile with settings tuned on the Pi 4. Other
sensors, including OV5647, can use their automatic tuning by setting
`camera.tuning_file: null`. The separately
prepared `piStreamTracker_Pi4_AD5X.3mf` supplies the mechanism; software limits must
be calibrated against its printed stops and the actual CSI cable routing.

`camera.py` exclusively owns the CSI camera and serves `127.0.0.1:8000`.
`web.py` receives `http://127.0.0.1:8000/stream`; raw preview, recording controls,
file listing and downloads use the same local camera address. Open the UI from
another LAN device at `http://<Pi-LAN-IP>:5000`. No second Pi or static
`192.168.100.1/2` addresses are needed.

### Wiring

Shut down the Pi and disconnect both power supplies before connecting or moving
the CSI cable or GPIO wiring. Use the Pi 4's 15-pin **CAMERA** connector and leave
enough cable slack for both axes. Follow the
[Raspberry Pi camera connection guide](https://www.raspberrypi.com/documentation/accessories/camera.html#install-a-raspberry-pi-camera).

| Connection | BCM identifier in YAML | Physical header pin |
|------------|------------------------|---------------------|
| Pan SG90 signal | GPIO18 | 12 |
| Tilt SG90 signal | GPIO19 | 35 |
| Common ground | GND | 6, for example |
| Both servo power inputs | Separate regulated 5 V supply | No GPIO signal pin |

Power the Pi through its own USB-C supply. Connect the servo supply ground,
both servo grounds and Pi ground together. Size the servo supply and wiring for
both servos moving together; do not power them from a GPIO signal or the 3.3 V
rail, and do not join the separate supply's positive output to Pi power.
Check connector polarity on the actual servos. BCM numbering differs from
physical pin numbering; GPIO uses 3.3 V logic. See the
[official GPIO pinout](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#40-pin-gpio-header).

### Install

From the checkout, as the user who will run the applications:

```bash
./setup.sh --standalone --config config.pi4.yaml
```

Pi 4 auto-detection also selects `standalone`. Setup creates a
`--system-site-packages` `venv/`, installs apt camera/libcamera bindings and
`rpicam-apps`, installs `pigpiod` and the Python client, and installs project
dependencies. It preserves the selected YAML file on repeat installations and
does not reconfigure networking or install EV3 udev rules in this role.
The applications run as the invoking ordinary user; the privileged daemon uses
`pigpiod -g -l -f` (foreground, localhost connections, FIFO disabled).
[Upstream daemon options](https://github.com/joan2937/pigpio/blob/master/pigpiod.c).

The target dependency combination is TFLite Runtime **2.14.0**, NumPy **<2** and
OpenCV contrib **>=4.8,<4.10**. The
[published TFLite wheel](https://pypi.org/project/tflite-runtime/2.14.0/#files)
includes CPython 3.11 Linux ARM64; the
[OpenCV 4.9 release](https://pypi.org/project/opencv-contrib-python/4.9.0.80/#files)
also supplies Linux ARM64 wheels. The NumPy bound avoids the
[NumPy 2 binary compatibility break](https://numpy.org/doc/stable/user/troubleshooting-importerror.html#downstream-importerror-attributeerror-or-c-api-abi-incompatibility).
Picamera2/libcamera come from the OS and remain visible through the venv, as
described in the [Picamera2 manual, virtual environments](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).
Setup checks imports and loads the MoveNet model/interpreter, downloading the
model if absent; an unavailable model/runtime fails setup. This still requires
camera capture and inference checks on the installed Pi image. Log out/in or reboot after group changes
before using the manual launcher.

Setup enables three services: `pigpiod`, `pistream-camera`, and
`pistream-tracker`. The tracker starts after and requires the camera and daemon.
Both applications receive the same absolute config path. A previous
`pitracker.service` on this device is stopped and disabled to avoid a second
camera owner. The web service opens the control panel immediately and starts
tracking with the configured motors **120 seconds after startup**, including
after a reboot. Manual **Start** or **Stop** cancels this pending one-time start.
Set `web.autostart_delay: null` to require manual Start instead; a numeric value
sets the delay in seconds (0–86400). The same behavior applies to foreground
web launches. Automatic startup does not start a recording.

`config.pi4.yaml` uses 1280×720 at 25 camera/recording frames/s, JPEG quality 80,
one MoveNet thread and at most 10 annotated-preview frames/s. The original
45-frame/s setting reached 82–84°C with only a passive radiator during the live
test. Reducing the stream to 25 frames/s lowered camera-process CPU usage from
about 107% to 65% of one core; it did not establish sustained thermal stability.
Check cooling under the intended workload before prolonged operation.
Camera recordings go to `camera-recordings/`; screenshots and
fallback local recordings go to `recordings/`, avoiding duplicate list entries.

The **Camera settings** panel below the preview changes and saves capture format,
FPS, white balance and exposure on the camera Pi. **Save & apply** writes the
camera service's active YAML atomically; changes survive service restarts and
reboots. It offers 1280×720/1920×1080 at 16:9 and 1280×960 at 4:3, with 25 or
30 camera frames/s. Format/FPS changes briefly stop and resume an active tracker;
recording must be stopped first. Other tracker settings remain session-only.

The supplied Pi 4 was measured at 25.000 capture frames/s (39,999 µs frame
duration). IMX219 uses the binned 1640×1232/10-bit sensor mode for 720p and
1280×960, preserving full sensor width; automatic 720p selection had cropped the
view to a 1920×1080 readout. Its room-light white-card calibration is red gain 1.15644, blue gain
2.24688; this manual balance is saved in `config.pi4.yaml`. Select **Auto** when
the lighting changes. Autoexposure with 50 Hz flicker correction (when supported)
is retained: in the measured dim scene it chose about 30 ms, maximum analogue
gain and some digital gain. Fixed 10/20 ms exposures were substantially darker.
Manual exposure is available in milliseconds with separate analogue gain; its
upper limit follows both the sensor and frame period. The panel reports actual
exposure/gain/frame rate separately from the tracker's processing FPS.

Both Pi camera configurations load `tuning/imx219-160-no-alsc.json`, preserving the camera
correction from the supplied project. The JSON omits `rpi.alsc`; it is specific
to this IMX219 setup. `camera.tuning_file` accepts an absolute path or a path
relative to the directory containing `camera.py`, independent of the working
directory or the configuration file's location. Null selects Picamera2's normal
sensor tuning; an explicitly selected profile that cannot be loaded fails camera
startup instead of falling back to a different image configuration.

### Calibrate before automatic motion

The current Pi configuration preserves its `gpio_servo.calibrated: true`
approval and mechanical limits, with pan −90…90° at 25°/s and tilt −45…45° at
15°/s. Both axes use `invert: true`, as verified by small manual movements on
the supplied mechanism. Gain is 6, reduced from 30; the previous 500/300°/s speed
caps were also reduced after the camera moved away from its target. These are
settings for that mechanism. For a new or changed assembly, set
`gpio_servo.calibrated: false` before starting services and commission it below.
The built-in servo defaults remain uncalibrated with smaller working ranges and
lower speeds. `calibrated: false` blocks automatic motor tracking; video
tracking, manual movement and Home remain available. **Start with the servo
horns detached from the mechanism**: the first pulse uses the configured Home
reference, and the application cannot know the actual shaft position.
Speed limits constrain subsequent command changes, not an unknown initial
physical movement.

Stop the tracker service so it releases GPIO; the camera service may remain on:

```bash
sudo systemctl stop pistream-tracker
cp config.pi4.yaml config.pi4.backup.yaml
venv/bin/python -m pistream.calibrate --config config.pi4.yaml
```

In the calibration prompt, first use a small working range:

```text
set pan.min_angle -5
set pan.max_angle 5
set tilt.min_angle -5
set tilt.max_angle 5
home
status
pan 1
pan -1
tilt 1
tilt -1
save
```

Wait for each small move and check its direction. `set pan.invert true` or
`set tilt.invert true` reverses that axis if required. Every `set` disables PWM,
clears calibration approval and requires another explicit `home` or manual
command. `stop`, `quit`, Ctrl-C and normal termination disable pulses and
release the controller; use `connect` after correcting an error that disconnected
it.

Calibrate each servo's angle-to-pulse map on the unloaded servo:
`pan.calibration.min_angle`, `max_angle`, `min_pulse_us`, `max_pulse_us`, and the
corresponding tilt settings. The default -90…90° / 1000…2000 µs map is provisional;
check the actual angles and pulse range of the installed units. Keep pulses
within the configured calibration limits. Set each axis's `home` and `max_speed`
(degrees/s) independently. The utility validates changes before enabling PWM.

Return to Home, quit, remove servo power and fit the mechanism at its neutral
position without turning the shafts. Re-enter calibration with narrow ranges.
Increase `pan.min_angle`/`max_angle` and `tilt.min_angle`/`max_angle` gradually,
one small move at a time, stopping before a printed part touches or the CSI cable
tightens. These **working limits** must remain inside the calibration angle
range; changing working limits does not rescale the pulse map. Do not assume
180° is safe in the assembled platform. Reduce ranges or speed if the servos
strain or oscillate.

Only when both directions, Home, pulse calibration and assembled limits have
been checked, enter:

```text
confirm
quit
```

`confirm` saves `calibrated: true` atomically to the chosen YAML. `save` preserves
the current approval state: after editing a setting this remains `false`; an
already confirmed, unchanged configuration remains approved. Other sections are
preserved, though YAML formatting/comments are rewritten. The UI shows
**commanded** and target angles in degrees, working limits, calibration status
and controller errors; no SG90 position feedback is available.

### Start, stop and recover

```bash
sudo systemctl start pistream-tracker
```

Open `http://<Pi-LAN-IP>:5000` and wait for the two-minute automatic start, or
press **Start** to start immediately. Check the image and use small D-pad steps.
**Stop** also cancels a pending automatic start, leaving tracking stopped until
the next manual Start or web service restart. **Home / Reset** clears detection
and returns to saved Home while holding automatic tracking for
`gpio_servo.home_hold`; manual motion uses
`manual_hold`. Stop cancels either motion immediately and disables pulses.
Losing the person cancels automatic motion and holds the last issued position.
The independent worker cancels automatic motion after `frame_timeout` seconds
without a fresh frame, even when frame processing stops. There is no search
rotation. A daemon connection failure remains an explicit controller error.

For a foreground run, stop both application services first; keep `pigpiod`
running:

```bash
sudo systemctl stop pistream-tracker pistream-camera
./run_standalone.sh --config config.pi4.yaml
# Ctrl-C stops both child applications.
sudo systemctl start pistream-camera pistream-tracker
```

The launcher rejects an active application service or another launcher.
Host-wide GPIO ownership locks also exclude a second tracker/calibration
session using the same pins. To run
without motion, append `--no-motors` to the launcher. Configuration changes take
effect in a new process; restart the relevant services after editing the file.

```bash
venv/bin/python -m pistream.install --config config.pi4.yaml status
sudo systemctl status pigpiod pistream-camera pistream-tracker
sudo journalctl -u pistream-camera -u pistream-tracker -u pigpiod -b --no-pager -n 100
curl http://127.0.0.1:8000/record/status
```

If `camera.token` is set, add the matching `Authorization: Bearer ...` header
to diagnostic recording requests.

For a camera-only diagnostic, release CSI ownership first:

```bash
sudo systemctl stop pistream-tracker pistream-camera
rpicam-hello --list-cameras
rpicam-still --nopreview --timeout 1000 --output /tmp/pistream-camera-test.jpg
sudo systemctl start pistream-camera pistream-tracker
```

Bookworm uses the `rpicam-*` commands; see
[Raspberry Pi camera software](https://www.raspberrypi.com/documentation/computers/camera_software.html).
If the camera is absent, power down before reseating its cable and check the
IMX219 entry (or the installed sensor) with `--list-cameras`. If the UI reports `pigpiod` unavailable, check
its service log, run `sudo systemctl restart pigpiod` and
`sudo systemctl start pistream-tracker`, then reload the UI and wait for automatic
startup or press Start immediately.
For a transient client error while the service stays up, reconnect using
**Motors Enabled** or Stop/Start the tracking session. There is no software-PWM
fallback. Errors about unconfirmed PWM shutdown stay visible after Stop; an
unknown PWM state is not confirmation that pulses are off. Remove servo power
before touching the mechanism; a lost daemon connection cannot confirm PWM
shutdown. Repeated import/ABI failures warrant checking the venv's Python and
NumPy versions and rerunning setup with the supported OS/runtime combination.

### Hardware acceptance checklist

Automated tests use fake GPIO, camera and EV3 components. They do not establish
physical motion, CSI capture, electrical stability or Pi 4 processing speed.
The remote Pi 4 / IMX219 / SG90 check on 2026-10-28 verified the camera image,
manual direction changes, Home and Stop. The operator confirmed horizontal
following, seated vertical following and motion stopping after leaving the frame.
A stationary run at 25 camera frames/s and two inference threads had 68/68
positive detections over 20 seconds, with fixed pan commands and no motor errors;
processing FPS had a median of 15.9. Tilt stayed at its −45° limit in that run,
so it did not demonstrate vertical centering of a standing person. Reposition the
camera or person before assessing vertical centering; these measurements do not
justify widening the mechanical limits. The current one-thread setting was
selected from an inference comparison under load: about 108 ms versus 78 ms
with two threads. Repeat the full checklist for the final assembly and workload.
A subsequent one-thread live test recorded 24 positive samples over 7.76 seconds,
median processing FPS 16.32, no motor errors or stream loss, and holding the last
angles after exit. A four-second camera-side H.264 MP4 at 1280×720 was checked
with ffprobe and a decoded frame; a 10.48-second recording through the web UI
also passed. The tracker and motors became active about
121.5 seconds after service startup with the 120-second delay selected.
A short one-thread post-start sample was 73.5°C, with no current thermal flags
and camera/tracker CPU around 53%/126% of one core; it does not establish
sustained thermal stability or complete mechanical acceptance. A later reading
after the web recording was 79.3°C, still without active thermal flags.

1. Verify supply polarity/common ground and unloaded Home; commission both
   calibrated axes and inspect the cable/printed stops through their full
   **working** ranges.
2. Reboot, verify all three services are active, open the UI from another LAN
   device and press Start. Confirm annotated and raw preview.
3. Check D-pad, Home/Reset and Stop during manual/Home motion; repeat Start/Stop
   and verify no second camera/GPIO owner appears.
4. Follow a person in both axes, verify limits, remove the person, and interrupt
   the camera stream. Confirm hold/cancellation and controlled recovery without
   a stale movement.
5. Capture a screenshot and a recording, stop recording, and download/play both
   files. Check camera and local fallback directories and available storage.
6. Measure actual processing FPS, preview latency, CPU temperature, power and
   tracking quality with recording enabled before tuning detection/preview
   settings; a short tracking sample is not a sustained-performance guarantee.
7. Reboot again and repeat image/manual/tracking/recording checks. Confirm the
   selected config, calibration and existing network settings persisted.

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
The web interface starts tracking and connects the configured motor backend
after `web.autostart_delay` seconds (120 by default). Manual Start/Stop cancels
the pending start; `null` disables automatic startup. Disabled motor backends and
the SG90 calibration gate still apply.

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
| Start / Stop | Starts/stops tracking and cancels a pending automatic start. Start connects to the camera stream and fails if it is down. |
| Home / Reset | Clears detection and homes the selected controller; auto-track holds for its configured `home_hold` |
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

Use `config.yaml` for the original two-Pi setup and `config.pi4.yaml` for the
standalone build. Missing settings use built-in defaults. CLI flags override a
named `--preset`, which overrides the file. The built-in `pi4` performance preset
still supplies conservative 640×480 at 15 camera frames/s; selecting it explicitly
overrides the current Pi file's 1280×720 at 25 frames/s. Standalone services and
`run_standalone.sh` pass the shared config without a preset, so they keep the
file's camera settings in both processes.

```yaml
network:
  camera_ip: "192.168.100.1"
  tracker_ip: "192.168.100.2"

camera:
  port: 8000
  framerate: 30
  resolution: {width: 1280, height: 960}
  tuning_file: tuning/imx219-160-no-alsc.json  # shared Pi3B+/Pi4 IMX219 profile
  awb_mode: auto           # auto | indoor | daylight | cloudy | manual
  colour_gains: [1.5, 1.5] # red, blue; used in manual WB
  exposure_mode: auto      # auto | manual
  exposure_time_us: 20000  # used in manual exposure, must fit one frame
  analogue_gain: 1.0       # used in manual exposure
  token: null              # Bearer for /record/* and /camera/settings; empty = open LAN
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
  autostart_delay: 120      # seconds; null disables the one-time automatic start
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

`motors.backend` accepts `ev3`, `gpio_servo`, or `none`. Its default `null` keeps
old files working: `ev3.enabled: true` selects EV3 and `false` selects no motors.
An explicit backend overrides that legacy flag. `--motor-backend` overrides the
file; `--no-ev3` disables only EV3, including an explicitly selected EV3 backend,
and still permits GPIO servos. `--no-motors` wins over both and disables all
actuators. GPIO mode never connects to EV3; hardware libraries are loaded only
by their selected controller.

SG90 settings live under `gpio_servo`. `deadzone` is a fraction of half the frame
dimension; `gain` sets the angular correction at full normalized offset;
`min_step` suppresses tiny changes. `max_dt` limits credited elapsed time after
a delayed worker cycle; each axis also has its own `max_speed` in degrees/s.
`tracker.detection.result_timeout` rejects inference results that are too old.

## CLI

```bash
python tracker.py --help
python web.py --help
python -m pistream.install --help
```

Shared flags: `--config`, `--url`, `--output-dir`, `--detection-interval`, `--process-scale`, `--confidence`, `--movenet-threads`, `--motor-backend ev3|gpio_servo|none`, `--no-motors`, `--no-ev3`, `--preset`.

Tracker-only: `--no-display`, `--auto-record`.

Web-only: `--host`, `--port`.

Camera: `--config` (loaded before recording directories and server settings are created).

### OpenCV window keys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Toggle recording |
| `s` | Screenshot |
| `d` | Reset tracker + home selected controller |
| `e` | Connect / disconnect selected controller |

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

No Pi, GPIO or EV3 required. GitHub Actions runs the same on Python 3.11 and 3.12.
Raspberry Pi OS Bookworm's system `python3` is 3.11.

Optional browser checks use an existing Node/Playwright installation: run
`.venv/bin/python tests/web_ui_fixture.py` and, in another terminal,
`node tests/browser_ui.cjs`. `PLAYWRIGHT_MODULE` and `PLAYWRIGHT_EXECUTABLE` can
select an existing installation and Chromium executable. The fixture uses mocked
hardware and checks manual retry, warnings, GPIO calibration/angle/error display,
manual command requests and responsive layout.

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
├── config.yaml          # Two-Pi settings
├── config.pi4.yaml      # Standalone camera/tracker/GPIO settings
├── tracker.py           # Display / CLI entry
├── web.py               # Flask entry
├── camera.py            # Camera Pi server
├── tuning/imx219-160-no-alsc.json  # Current Pi4 camera correction profile
├── ev3_usb.py           # Shim → pistream.ev3_usb
├── setup.sh             # Bootstrap apt/venv; exec python -m pistream.install
├── run_cam.sh
├── run_tracker.sh
├── run_standalone.sh    # Two local application processes, shared config
├── pistream/            # Application package
│   ├── install.py       # Installer CLI (network, udev, systemd)
│   ├── capture.py       # Threaded MJPEG capture + reconnect
│   ├── detect.py        # MoveNet + async worker
│   ├── track.py         # HumanTracker, MOSSE, CLI
│   ├── motors.py        # EV3 pan/tilt + home hold
│   ├── motion.py        # Controller interface/factory and no-motor backend
│   ├── servo.py         # Local pigpio SG90 commands and watchdog
│   ├── calibrate.py     # Interactive GPIO calibration and YAML saving
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
