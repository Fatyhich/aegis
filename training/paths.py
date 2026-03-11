"""
Path constants for the training package.

Single source of truth for locating the segmast3r submodule and its
vendored dependencies (mast3r_src / dust3r / croco).

Usage in any training module::

    from training.paths import setup_segmast3r_path
    setup_segmast3r_path()   # call once before importing from src.* / mast3r_src.*
"""

from pathlib import Path

# Absolute paths — resolved at import time, not relative to cwd.
AEGIS_ROOT: Path = Path(__file__).resolve().parents[1]
SEGMAST3R_ROOT: Path = AEGIS_ROOT / "third_party" / "segmast3r"


def setup_segmast3r_path() -> None:
    """Add segmast3r and its vendored deps to sys.path (idempotent)."""
    import sys

    for path in [
        SEGMAST3R_ROOT,
        SEGMAST3R_ROOT / "mast3r_src" / "dust3r",
        SEGMAST3R_ROOT / "mast3r_src" / "dust3r" / "croco",
    ]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
