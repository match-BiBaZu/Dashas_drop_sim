"""Repository-local CAD pairs available after installation or relocation."""

from pathlib import Path


def geometry_directory() -> Path:
    return Path(__file__).resolve().parent / "assets" / "geometry"


def cad_directory() -> Path:
    return geometry_directory() / "cad"


def local_geometry_path(value: str | Path) -> Path:
    """Use the exact bundled copy of a formerly shared geometry path, if present."""
    path = Path(value)
    parts = str(value).replace("\\", "/").split("/")
    if "bibazu_geometry_to_pose" not in parts:
        return path
    suffix = parts[parts.index("bibazu_geometry_to_pose") + 1:]
    roots = {"Werkstücke_STL_grob": cad_directory()}
    if suffix and suffix[0] in roots:
        candidate = roots[suffix[0]].joinpath(*suffix[1:]).resolve()
        if candidate.is_relative_to(geometry_directory()) and candidate.exists():
            return candidate
    return path
