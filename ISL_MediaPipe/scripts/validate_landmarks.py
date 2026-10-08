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

landmark_counts = {}
hands_checked = 0

for frame in data["frames"]:
    for hand in frame["hands"]:
        hands_checked += 1

        count = len(hand["landmarks"])

        landmark_counts[count] = landmark_counts.get(count, 0) + 1

print("--- Landmark Structure Validation ---")
print("Total hands checked:", hands_checked)

for count, occurrences in landmark_counts.items():
    print(f"Landmarks per hand: {count} | Occurrences: {occurrences}")

if list(landmark_counts.keys()) == [21]:
    print("All hands contain 21 landmarks.")
else:
    print("Review landmark counts above.")