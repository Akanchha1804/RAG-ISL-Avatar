
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output_landmarks"

cleaned_files = list(OUTPUT_DIR.glob("*_cleaned.json"))

if not cleaned_files:
    print("Cleaned JSON file not found.")
    exit()

FILE_PATH = cleaned_files[0]

with open(FILE_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

frames = data["frames"]

frame_map = {
    frame["frame_index"]: frame
    for frame in frames
}

missing_indices = [
    index
    for index, frame in frame_map.items()
    if not frame["hands"]
]

print("--- Internal Gap Analysis ---")

for index in missing_indices:
    previous_frame = frame_map.get(index - 1)
    next_frame = frame_map.get(index + 1)

    print(f"\nMissing frame: {index}")

    if previous_frame:
        print(
            "Previous frame:",
            index - 1,
            "| Hands:",
            len(previous_frame["hands"])
        )
    else:
        print("Previous frame unavailable.")

    if next_frame:
        print(
            "Next frame:",
            index + 1,
            "| Hands:",
            len(next_frame["hands"])
        )
    else:
        print("Next frame unavailable.")

print("\nAnalysis completed.")