"""Shared constants, logger, and utility functions used across all modules."""

import logging
from pathlib import Path

logger = logging.getLogger("yolo_bench")

# TFLite models are compiled with fixed 640x640 input.
TFLITE_IMG_SIZE = 640

# Minimum duration (seconds) for predicted benchmark videos.
# The actual frame count is computed as ceil(fps × duration).
# If the source video is shorter, the entire video is used.
BENCH_MIN_DURATION_S = 24


def file_size_mb(path: Path) -> float | None:
    """Return file size in MB, or None if the file does not exist."""
    return round(path.stat().st_size / 1_048_576, 2) if path.exists() else None
