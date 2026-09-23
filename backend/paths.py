"""
Central path resolution for the backend.

DATA_DIR mechanism (preserved from the original implementation):
  - The environment variable DATA_DIR overrides the default project-root layout.
  - Each asset first tries the DATA_DIR/project layout, then a backend-local
    fallback (so the backend also works when deployed with only backend/ copied).

Resolution is explicit and testable:
  - resolve_paths() records which candidate was selected for every asset
    (source = "data_dir" | "backend_fallback") so misconfiguration is visible
    in /api/health instead of being silently swallowed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = BACKEND_DIR.parent

SOURCE_DATA_DIR = "data_dir"
SOURCE_BACKEND_FALLBACK = "backend_fallback"


@dataclass(frozen=True)
class AppPaths:
    data_dir: Path
    landmarks_dir: Path
    landmarks_dir_source: str
    mapping_file: Path
    mapping_file_source: str
    isl_corpus_csv: Path
    isl_corpus_csv_source: str
    cislr_csv: Path
    cislr_csv_source: str
    index_dir: Path
    model_dir: Path

    def describe(self) -> dict:
        """Structured path report for /api/health (never raises)."""

        def entry(path: Path, source: str) -> dict:
            return {"path": str(path), "exists": path.exists(), "source": source}

        return {
            "data_dir": str(self.data_dir),
            "landmarks_dir": entry(self.landmarks_dir, self.landmarks_dir_source),
            "mapping_file": entry(self.mapping_file, self.mapping_file_source),
            "isl_corpus_csv": entry(self.isl_corpus_csv, self.isl_corpus_csv_source),
            "cislr_csv": entry(self.cislr_csv, self.cislr_csv_source),
            "index_dir": entry(self.index_dir, "backend"),
            "model_dir": entry(self.model_dir, "backend"),
        }


def _pick(primary: Path, fallback: Path) -> tuple[Path, str]:
    if primary.exists():
        return primary, SOURCE_DATA_DIR
    return fallback, SOURCE_BACKEND_FALLBACK


def resolve_paths(environ: Optional[dict] = None) -> AppPaths:
    env = os.environ if environ is None else environ
    data_dir = Path(env.get("DATA_DIR") or DEFAULT_PROJECT_DIR)

    landmarks_dir, landmarks_src = _pick(
        data_dir / "ISL_MediaPipe" / "output_landmarks",
        BACKEND_DIR / "landmarks",
    )
    mapping_file, mapping_src = _pick(
        data_dir / "ISL_MediaPipe" / "sentence_mapping.json",
        BACKEND_DIR / "sentence_mapping.json",
    )
    isl_corpus_csv, isl_src = _pick(
        data_dir
        / "Dataset"
        / "data"
        / "isl_csltr"
        / "ISL_CSLRT_Corpus"
        / "ISL_CSLRT_Corpus"
        / "corpus_csv_files"
        / "ISL Corpus sign glosses.csv",
        BACKEND_DIR / "ISL Corpus sign glosses.csv",
    )
    cislr_csv, cislr_src = _pick(
        data_dir / "Dataset" / "data" / "cislr" / "dataset.csv",
        BACKEND_DIR / "dataset.csv",
    )

    return AppPaths(
        data_dir=data_dir,
        landmarks_dir=landmarks_dir,
        landmarks_dir_source=landmarks_src,
        mapping_file=mapping_file,
        mapping_file_source=mapping_src,
        isl_corpus_csv=isl_corpus_csv,
        isl_corpus_csv_source=isl_src,
        cislr_csv=cislr_csv,
        cislr_csv_source=cislr_src,
        index_dir=BACKEND_DIR / "index",
        model_dir=BACKEND_DIR / "models" / "isl_gloss_t5",
    )


@lru_cache(maxsize=1)
def get_paths() -> AppPaths:
    return resolve_paths()


def reset_paths_cache() -> None:
    """Test hook: clear the cached paths (call after changing DATA_DIR)."""
    get_paths.cache_clear()
