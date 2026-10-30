"""Toy physical environments for perceptual coherence experiments."""

import logging
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "perception-env-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

__version__ = "0.1.0"

