import argparse

import yaml

from tracker.tracker import Tracker


def main():
    # Load config to override argparse defaults
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tracker_config = config.get('tracker', {})
    detection_config = tracker_config.get('detection', {})

    parser = argparse.ArgumentParser(description='Human Tracker for Raspberry Pi 5 with EV3 Control')
    parser.add_argument('--url', default=tracker_config.get('stream_url', 'http://192.168.100.1:8000/stream'),
                        help='Stream URL')
    parser.add_argument('--output-dir', default=tracker_config.get('output_dir', 'recordings'),
                        help='Output directory for recordings')
    parser.add_argument('--detection-type', default=detection_config.get('type', 'face'), help='[face, body]')
    parser.add_argument('--confidence-threshold', type=float, default=detection_config.get('confidence_threshold', 0.7),
                        help='Confidence threshold for human detection')
    parser.add_argument('--detection-interval', type=int, default=detection_config.get('interval', 5),
                        help='Run detection every N frames (higher = faster but less responsive)')
    parser.add_argument('--process-scale', type=float, default=detection_config.get('process_scale', 0.3),
                        help='Scale factor for detection processing (lower = faster)')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without displaying video (saves CPU)')
    parser.add_argument('--no-auto-record', action='store_true',
                        help='Do not start recording automatically')

    args = parser.parse_args()

    # Update config with any command-line arguments
    tracker_config['stream_url'] = args.url
    tracker_config['output_dir'] = args.output_dir
    detection_config['type'] = args.detection_type
    detection_config['confidence_threshold'] = args.confidence_threshold
    detection_config['interval'] = args.detection_interval
    detection_config['process_scale'] = args.process_scale
    
    # Write the potentially updated config back
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)

    tracker = Tracker()

    tracker.run(
        display_video=not args.no_display,
        auto_record=not args.no_auto_record
    )

if __name__ == "__main__":
    main()
