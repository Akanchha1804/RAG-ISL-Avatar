"""
Generate sentence_mapping.json from ISL-CSLTR CSV + processed landmark files.
Run this AFTER batch_process.py has generated the landmark JSON files.
"""

import json
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = PROJECT_DIR / "ISL_MediaPipe" / "output_landmarks"
ISL_CSV = PROJECT_DIR / "Dataset" / "data" / "isl_csltr" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
OUTPUT_FILE = PROJECT_DIR / "ISL_MediaPipe" / "sentence_mapping.json"


def main():
    gloss_mapping = {}
    if ISL_CSV.exists():
        with open(ISL_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sentence = row["Sentence"].strip()
                glosses = row["SIGN GLOSSES"].strip()
                gloss_mapping[sentence.lower()] = {
                    "sentence": sentence,
                    "glosses": glosses
                }

    print(f"Loaded {len(gloss_mapping)} gloss entries from CSV.")

    landmark_files = {f.name: f for f in LANDMARKS_DIR.glob("*_landmarks.json")}
    print(f"Found {len(landmark_files)} landmark files.")

    mapping = {}
    matched = 0
    unmatched = 0

    for sentence_lower, info in gloss_mapping.items():
        safe_name = sentence_lower.replace(" ", "_").replace("'", "").replace(",", "")
        landmark_filename = f"{safe_name}_landmarks.json"

        if landmark_filename in landmark_files:
            landmark_size = landmark_files[landmark_filename].stat().st_size
            mapping[sentence_lower] = {
                "sentence": info["sentence"],
                "glosses": info["glosses"],
                "landmark_file": landmark_filename,
                "landmark_size_bytes": landmark_size,
                "status": "ready"
            }
            matched += 1
        else:
            mapping[sentence_lower] = {
                "sentence": info["sentence"],
                "glosses": info["glosses"],
                "landmark_file": "",
                "status": "no_landmarks"
            }
            unmatched += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nSentence mapping created: {OUTPUT_FILE}")
    print(f"Matched (ready): {matched}")
    print(f"Unmatched: {unmatched}")
    print(f"Total: {matched + unmatched}")


if __name__ == "__main__":
    main()
