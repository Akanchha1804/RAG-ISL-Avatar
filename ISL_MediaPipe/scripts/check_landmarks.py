
import json
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output_landmarks"

# Find landmark JSON files
json_files = list(OUTPUT_DIR.glob("*_landmarks.json"))

if not json_files:
    print("No landmark JSON files found.")
    print("Checked:", OUTPUT_DIR)
    exit()

# Select the first JSON file
FILE_PATH = json_files[0]

print("Reading:", FILE_PATH)

# Load JSON
with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

total_frames = len(data["frames"])
frames_with_hands = 0
total_hands = 0

for frame in data["frames"]:
    hands = frame["hands"]

    if hands:
        frames_with_hands += 1
        total_hands += len(hands)

print("\n--- Landmark Analysis ---")
print("Total frames:", total_frames)
print("Frames with hands:", frames_with_hands)
print("Frames without hands:", total_frames - frames_with_hands)
print("Total detected hands:", total_hands)

if frames_with_hands > 0:
    print("Hand landmarks detected successfully.")
else:
    print("No hand landmarks detected.")