"""LiAgent - Local intelligent agent powered by MLX."""
import os
import warnings

__version__ = "0.1.0"


if os.environ.get("LIAGENT_SUPPRESS_RESOURCE_TRACKER_WARNING", "1").strip().lower() in {"1", "true", "yes"}:
    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be \d+ leaked semaphore objects to clean up at shutdown",
        category=UserWarning,
    )
