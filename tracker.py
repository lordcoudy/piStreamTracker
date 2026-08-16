#!/usr/bin/env python3
"""Display / CLI entry point for the tracker."""

import logging

from pistream.track import main

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)

if __name__ == '__main__':
    main()
