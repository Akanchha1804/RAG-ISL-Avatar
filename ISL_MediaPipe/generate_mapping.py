"""
Generate sentence_mapping.json from ISL-CSLTR CSV + processed landmark files.

batch_process.py already writes the mapping (with detection statistics), so
this script is a repair tool. It therefore:

  * validates that a landmark file actually CONTAINS hand landmarks instead
    of trusting that the file exists - an empty file is marked
    "no_landmarks", not "ready";
  * preserves any richer fields already recorded by batch_process.py
    (detection_rate, video_used, trimmed_start, ...);
  * mirrors the result to backend/sentence_mapping.json so the fallback copy
    the backend can use stays in sync.

Run this AFTER batch_process.py has generated the landmark JSON files.
"""

import csv
import json
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = PROJECT_DIR / "ISL_MediaPipe" / "output_landmarks"
ISL_CSV = (
    PROJECT_DIR / "Dataset" / "data" / "isl_csltr" / "ISL_CSLRT_Corpus"
    / "ISL_CSLRT_Corpus" / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
)
OUTPUT_FILE = PROJECT_DIR / "ISL_MediaPipe" / "sentence_mapping.json"
BACKEND_COPY = PROJECT_DIR / "backend" / "sentence_mapping.json"


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def file_has_hands(path: Path) -> tuple[bool, int]:
    """Return (has_hands, frames_with_hands) for a landmark JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, 0
    frames = data.get("frames") or []
    with_hands = sum(1 for f in frames if f.get("hands"))
    return with_hands > 0, with_hands


def main():
    if not ISL_CSV.exists():
        raise SystemExit(f"ISL corpus CSV not found: {ISL_CSV}")
    if not LANDMARKS_DIR.is_dir():
        raise SystemExit(f"Landmarks directory not found: {LANDMARKS_DIR}")

    gloss_mapping = {}
    with open(ISL_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sentence = row["Sentence"].strip()
            gloss_mapping[_normalize(sentence)] = {
                "sentence": sentence,
                "glosses": row["SIGN GLOSSES"].strip(),
            }

    print(f"Loaded {len(gloss_mapping)} gloss entries from CSV.")

    landmark_files = {f.name: f for f in LANDMARKS_DIR.glob("*_landmarks.json")}
    print(f"Found {len(landmark_files)} landmark files.")

    # Keep the per-sentence stats batch_process.py already computed.
    previous = {}
    if OUTPUT_FILE.exists():
        try:
            previous = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}

    mapping = {}
    matched = 0
    empty = 0

    for _, info in gloss_mapping.items():
        sentence = info["sentence"]
        sentence_lower = sentence.lower()
        safe_name = sentence.replace(" ", "_").replace("'", "").replace(",", "")
        landmark_filename = f"{safe_name}_landmarks.json"

        entry = dict(previous.get(sentence_lower, {}))
        entry["sentence"] = sentence
        entry["glosses"] = info["glosses"]

        path = landmark_files.get(landmark_filename)
        has_hands, with_hands = file_has_hands(path) if path else (False, 0)

        if has_hands:
            entry["landmark_file"] = landmark_filename
            entry["landmark_size_bytes"] = path.stat().st_size
            entry.setdefault("hands_frames", with_hands)
            entry["status"] = "ready"
            matched += 1
        else:
            entry["landmark_file"] = ""
            entry["status"] = "no_landmarks"
            empty += 1

        mapping[sentence_lower] = entry

    # Also keep any sentence that exists only in the previous mapping.
    for key, value in previous.items():
        if key not in mapping:
            mapping[key] = value

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    if BACKEND_COPY.parent.exists():
        shutil.copyfile(OUTPUT_FILE, BACKEND_COPY)

    print(f"\nSentence mapping created: {OUTPUT_FILE}")
    print(f"Ready (file exists AND contains hands): {matched}")
    print(f"no_landmarks (missing or empty file):   {empty}")
    print(f"Total: {matched + empty}")
    if BACKEND_COPY.exists():
        print(f"Synced backend copy: {BACKEND_COPY}")


if __name__ == "__main__":
    main()
