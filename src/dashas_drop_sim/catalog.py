"""Offline snapshot of the public BiBaZu coarse STL workpiece catalogue."""

from __future__ import annotations

from pathlib import Path
import os


SOURCE_REPOSITORY = "https://github.com/match-BiBaZu/bibazu_geometry_to_pose"
SOURCE_COMMIT = "02d3fbcfcdc86fcf09ddbe40f39e89a936130447"


def catalog_directory() -> Path:
    if root := os.environ.get("DASHAS_DROP_SIM_CATALOG_DIR"):
        return Path(root).expanduser().resolve()
    return Path(__file__).resolve().parent / "assets" / "catalog"


def catalog_models() -> dict[str, Path]:
    """Return the bundled STL files keyed by their workpiece names."""
    directory = catalog_directory()
    paths = list(directory.iterdir())
    if os.environ.get("DASHAS_DROP_SIM_CATALOG_DIR"):
        paths.extend(directory.glob("*/*"))
    models = {}
    for path in sorted(paths, key=lambda item: item.stem.casefold()):
        if path.is_file() and path.suffix.lower() == ".stl":
            if path.stem in models:
                raise ValueError(f"Doppeltes Werkstück im Katalog: {path.stem}")
            models[path.stem] = path
    return models
