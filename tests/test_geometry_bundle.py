"""CAD export and geometry matching survive moving the application's bundle."""

import shutil
from pathlib import Path

import pytest
from dashas_drop_sim.catalog import catalog_models
from dashas_drop_sim.catalog_export import model_pair
from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.pose_matching import PoseResolver

from dashas_drop_sim import resources


def test_all_catalog_models_have_byte_identical_local_cad_pairs():
    for mesh in catalog_models().values():
        stl, step = model_pair(mesh)
        assert stl.parent == resources.cad_directory()
        assert stl.read_bytes() == mesh.read_bytes()
        assert step.is_file()


def test_geometry_matching_after_bundle_relocation(tmp_path, monkeypatch):
    pytest.importorskip("chute_pose")
    bundle = tmp_path / "independent" / "geometry"
    shutil.copytree(resources.geometry_directory(), bundle)
    monkeypatch.setattr(resources, "geometry_directory", lambda: bundle)
    monkeypatch.chdir(tmp_path)
    mesh = bundle / "cad" / "Df1a.STL"
    resolver = PoseResolver(mesh)
    assert resolver.symmetry_available
    assert not resolver.known_pose_ids
    assert resolver.cluster_unmatched([[0, 0, 0, 1], [0, 0, 0, -1]]) == [0, 0]
    assert not (bundle / "roadmaps").exists()
    stl, step = model_pair(mesh)
    assert stl.parent == step.parent == bundle / "cad"


def test_old_config_paths_use_exact_local_copies():
    prefix = Path("Z:/archive/bibazu_geometry_to_pose")
    config = RunConfig(
        mesh_path=prefix / "Werkstücke_STL_grob/Df1a.STL",
        catalog_cad_dir=prefix / "Werkstücke_STL_grob",
    )
    config.validate()
    assert config.mesh_path == resources.cad_directory() / "Df1a.STL"
    assert config.catalog_cad_dir == resources.cad_directory()
    missing = prefix / "Poses_Found_Robust/missing.yaml"
    assert resources.local_geometry_path(missing) == missing


def test_retired_roadmap_input_is_ignored_in_saved_configs():
    config = RunConfig.from_dict({
        "mesh_path": str(resources.cad_directory() / "Df1a.STL"),
        "roadmap_path": "Z:/removed/roadmap.yaml",
    })
    config.validate()
    assert "roadmap_path" not in config.to_dict()
