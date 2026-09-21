
import json
from pathlib import Path


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output_landmarks"

json_files = list(OUTPUT_DIR.glob("*_landmarks.json"))

if not json_files:
    print("No landmark JSON files found.")
    exit()

INPUT_FILE = json_files[0]

OUTPUT_FILE = OUTPUT_DIR / (
    INPUT_FILE.stem.replace("_landmarks", "")
    + "_cleaned.json"
)


# ==========================================
# 2. LOAD DATA
# ==========================================

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

frames = data["frames"]


# ==========================================
# 3. FIND ACTIVE FRAME RANGE
# ==========================================

active_indices = [
    index
    for index, frame in enumerate(frames)
    if frame["hands"]
]

if not active_indices:
    print("No detected hands found.")
    exit()

first_active = active_indices[0]
last_active = active_indices[-1]

cleaned_frames = frames[first_active:last_active + 1]


# ==========================================
# 4. UPDATE METADATA
# ==========================================

data["frames"] = cleaned_frames

data["original_frame_count"] = len(frames)

data["cleaned_frame_count"] = len(cleaned_frames)

data["trimmed_start_frames"] = first_active

data["trimmed_end_frames"] = (
    len(frames) - last_active - 1
)

data["preprocessing_note"] = (
    "Leading and trailing frames without detected "
    "hands were removed. Internal missing frames "
    "were preserved."
)


# ==========================================
# 5. SAVE CLEANED JSON
# ==========================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=2)

print("--- Preprocessing Completed ---")
print(f"Input file: {INPUT_FILE.name}")
print(f"Output file: {OUTPUT_FILE.name}")
print(f"Original frames: {len(frames)}")
print(f"Cleaned frames: {len(cleaned_frames)}")
print(f"Removed beginning frames: {first_active}")
print(f"Removed ending frames: {len(frames) - last_active - 1}")
print("Internal missing frames preserved.")