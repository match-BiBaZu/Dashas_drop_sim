"""Offline snapshot of the public BiBaZu coarse STL workpiece catalogue."""

from __future__ import annotations

from pathlib import Path


SOURCE_REPOSITORY = "https://github.com/match-BiBaZu/bibazu_geometry_to_pose"
SOURCE_COMMIT = "02d3fbcfcdc86fcf09ddbe40f39e89a936130447"


def catalog_directory() -> Path:
    return Path(__file__).resolve().parent / "assets" / "catalog"


def catalog_models() -> dict[str, Path]:
    """Return the bundled STL files keyed by their workpiece names."""
    directory = catalog_directory()
    return {path.stem: path for path in sorted(directory.iterdir(), key=lambda item: item.stem.casefold())
            if path.is_file() and path.suffix.lower() == ".stl"}
