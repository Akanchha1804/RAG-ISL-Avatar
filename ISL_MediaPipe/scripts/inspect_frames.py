
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input_videos"
OUTPUT_DIR = PROJECT_ROOT / "debug_frames"

OUTPUT_DIR.mkdir(exist_ok=True)

video_files = list(INPUT_DIR.glob("*"))

if not video_files:
    print("No video found.")
    exit()

VIDEO_PATH = video_files[0]

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

selected_frames = [
    0, 20, 36, 37, 38, 39,
    100, 200, 268, 269, 290, 310, 329
]

for frame_index in selected_frames:
    if frame_index >= total_frames:
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()

    if success:
        output_path = OUTPUT_DIR / f"frame_{frame_index}.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

cap.release()

print("\nFrame inspection completed.")
print(f"Open this folder: {OUTPUT_DIR}")