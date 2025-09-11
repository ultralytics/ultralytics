#!/usr/bin/env python3
"""
Debug version of the YOLO CLI command.
This script replicates the functionality of the 'yolo' command but allows for debugging.
"""

import sys
from ultralytics.cfg import entrypoint

if __name__ == '__main__':
    # Simulate the yolo command behavior
    sys.argv[0] = 'yolo'  # Set the script name to 'yolo' for consistency
    sys.exit(entrypoint())
