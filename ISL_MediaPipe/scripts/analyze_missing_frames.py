
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output_landmarks"

json_files = list(OUTPUT_DIR.glob("*_landmarks.json"))

if not json_files:
    print("No landmark JSON files found.")
    exit()

FILE_PATH = json_files[0]

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

frames = data["frames"]

total_frames = len(frames)

detected_frames = [
    frame["frame_index"]
    for frame in frames
    if frame["hands"]
]

missing_frames = [
    frame["frame_index"]
    for frame in frames
    if not frame["hands"]
]

print("--- Detection Quality Analysis ---")

print("Video:", data["video_name"])
print("FPS:", data["fps"])
print("Total frames:", total_frames)
print("Frames with hands:", len(detected_frames))
print("Missing frames:", len(missing_frames))

coverage = (len(detected_frames) / total_frames) * 100

print(f"Detection coverage: {coverage:.2f}%")

# Find continuous missing-frame intervals

missing_ranges = []

if missing_frames:

    start = missing_frames[0]
    previous = missing_frames[0]

    for current in missing_frames[1:]:

        if current == previous + 1:

            previous = current

        else:

            missing_ranges.append((start, previous))

            start = current
            previous = current

    missing_ranges.append((start, previous))

print("\n--- Missing Frame Ranges ---")

for start, end in missing_ranges:

    count = end - start + 1

    print(
        f"Frames {start}-{end} | "
        f"Count: {count}"
    )

print("\nAnalysis completed.")