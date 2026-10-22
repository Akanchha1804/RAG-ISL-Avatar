"""Path resolution guards, in particular the VRM avatar fallback.

Regression guard for the Docker image. `ENV DATA_DIR=/data` plus a compose
mount carrying only ISL_MediaPipe/ used to resolve /avatar/model.vrm to a file
nothing ever mounts, so the container served no avatar and the viewport
rendered nothing. The image now also carries the model at
backend/avatar/model_isl.vrm and paths.py falls back to it.
"""

import paths
from paths import BACKEND_DIR, SOURCE_BACKEND_FALLBACK, SOURCE_DATA_DIR


def test_vrm_prefers_the_data_dir_layout(tmp_path):
    vrm = tmp_path / "ISL-3D Avatar" / "Assets" / "model_isl.vrm"
    vrm.parent.mkdir(parents=True)
    vrm.write_bytes(b"glTF")

    resolved = paths.resolve_paths({"DATA_DIR": str(tmp_path)})

    assert resolved.vrm_model == vrm
    assert resolved.vrm_model_source == SOURCE_DATA_DIR


def test_vrm_falls_back_when_data_dir_carries_no_avatar(tmp_path):
    # Exactly the container: DATA_DIR exists but holds no ISL-3D Avatar/.
    resolved = paths.resolve_paths({"DATA_DIR": str(tmp_path)})

    assert resolved.vrm_model == BACKEND_DIR / "avatar" / "model_isl.vrm"
    assert resolved.vrm_model_source == SOURCE_BACKEND_FALLBACK


def test_health_reports_where_the_vrm_came_from(tmp_path):
    described = paths.resolve_paths({"DATA_DIR": str(tmp_path)}).describe()

    assert described["vrm_model"]["path"].endswith("model_isl.vrm")
    assert described["vrm_model"]["source"] == SOURCE_BACKEND_FALLBACK


def test_image_carries_the_vrm_at_the_fallback_path():
    """The fallback file exists only inside the built image, so the COPY that
    creates it is the other half of the fix and must not be dropped."""
    dockerfile = (BACKEND_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "ISL-3D Avatar/Assets/model_isl.vrm" in dockerfile
