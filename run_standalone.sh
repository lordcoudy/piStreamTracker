#!/usr/bin/env bash
# Run the existing camera and web applications together on one Pi.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ $EUID -eq 0 ]]; then
    echo "Run as an ordinary user; setup grants camera and motor access." >&2
    exit 1
fi

CONFIG="$SCRIPT_DIR/config.pi4.yaml"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            [[ $# -ge 2 ]] || { echo "--config needs a path" >&2; exit 2; }
            CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#--config=}"; shift ;;
        --help|-h)
            echo "Usage: $0 [--config CONFIG] [web.py options]"
            echo "Default: config.pi4.yaml (GPIO servos). Use config.pi4-ev3.yaml for EV3 USB."
            exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
[[ -f "$CONFIG" ]] || { echo "Missing configuration: $CONFIG" >&2; exit 2; }

if command -v systemctl >/dev/null 2>&1; then
    for unit in pitracker pistream-camera pistream-tracker; do
        state=$(systemctl is-active "$unit" 2>/dev/null || true)
        if [[ "$state" == "active" || "$state" == "activating" ]]; then
            echo "$unit.service is already running. Stop it before launching manually:" >&2
            echo "sudo systemctl stop pistream-tracker pistream-camera pitracker" >&2
            exit 1
        fi
    done
fi

PY="$SCRIPT_DIR/venv/bin/python"
[[ -x "$PY" ]] || PY="$SCRIPT_DIR/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Run ./setup.sh --standalone first." >&2; exit 1; }
CONFIG=$("$PY" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$CONFIG")
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# A checkout lock also excludes two simultaneous manual launchers.
exec 9>"$SCRIPT_DIR/.standalone.lock"
if command -v flock >/dev/null 2>&1; then
    flock -n 9 || { echo "Standalone launcher is already running." >&2; exit 1; }
fi
CAM_PID=""
WEB_PID=""
cleanup() {
    trap - EXIT INT TERM
    for pid in "$WEB_PID" "$CAM_PID"; do
        [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
    done
    # Bound shutdown without leaving a timeout process behind.
    local attempt
    for ((attempt=0; attempt<60; attempt++)); do
        if ! kill -0 "$WEB_PID" 2>/dev/null && ! kill -0 "$CAM_PID" 2>/dev/null; then
            break
        fi
        sleep 0.25
    done
    for pid in "$WEB_PID" "$CAM_PID"; do
        [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
    done
    [[ -z "$WEB_PID" ]] || wait "$WEB_PID" 2>/dev/null || true
    [[ -z "$CAM_PID" ]] || wait "$CAM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$PY" "$SCRIPT_DIR/camera.py" --config "$CONFIG" &
CAM_PID=$!
"$PY" "$SCRIPT_DIR/web.py" --config "$CONFIG" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
WEB_PID=$!
echo "Standalone started. Stop with Ctrl-C. Config: $CONFIG"
while kill -0 "$CAM_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
    sleep 0.25
done
set +e
if ! kill -0 "$CAM_PID" 2>/dev/null; then
    wait "$CAM_PID"
else
    wait "$WEB_PID"
fi
result=$?
echo "An application exited (status $result); shutting down standalone." >&2
[[ $result -ne 0 ]] || result=1
exit "$result"
