# import necessary libraries
import asyncio

from vidgear.gears import PiGear
from vidgear.gears.asyncio import StreamGear

# --- Camera Configuration ---
# Adjust resolution and framerate as needed
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FRAMERATE = 30
# --------------------------

async def main():
    """
    Initializes the camera using PiGear and streams it via RTSP using StreamGear.
    """
    print("Starting RTSP video streaming server...")

    # Configure PiGear
    options = {
        "framerate": FRAMERATE,
        "resolution": (VIDEO_WIDTH, VIDEO_HEIGHT),
    }
    stream = PiGear(logging=True, **options).start()

    # Configure StreamGear for RTSP output
    # The output URL will be rtsp://<your-pi-ip>:8000/live
    stream_params = {
        "-input_framerate": stream.framerate,
        "-f": "rtsp",
        "-rtsp_transport": "tcp",
        "-vcodec": "copy", # No re-encoding
    }
    
    # Instantiate StreamGear
    server = StreamGear(
        output="rtsp://0.0.0.0:8000/live",
        format="rtsp",
        logging=True,
        **stream_params
    )

    # Start streaming
    server.stream(stream)

    print("RTSP server is running at rtsp://<your-pi-ip>:8000/live")
    print("Press Ctrl+C to stop.")

    # Keep the server running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        print("Stopping server...")
        server.close()
        stream.stop()
        print("Server stopped.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
