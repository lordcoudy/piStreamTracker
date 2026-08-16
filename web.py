#!/usr/bin/env python3
"""Web UI entry point."""

import logging

from pistream.web_app import main

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

if __name__ == '__main__':
    main()
