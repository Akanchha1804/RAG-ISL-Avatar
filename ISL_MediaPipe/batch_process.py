"""
Batch process all ISL-CSLTR sentence videos through MediaPipe.
Extracts hand landmarks, cleans, and interpolates gaps.
"""

import cv2
import json
import time
import csv
from pathlib import Path

# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
DATASET_DIR = Path(r"F:\FY PROJECT\Dataset\data\isl_csltr\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus")
SENTENCES_DIR = DATASET_DIR / "Videos_Sentence_Level"
GLOSS_CSV = DATASET_DIR / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
OUTPUT_DIR = PROJECT_DIR / "output_landmarks"
MAPPING_FILE = PROJECT_DIR / "sentence_mapping.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. MEDIAPIPE SETUP
# =====================================================

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".MOV"}


# =====================================================
# 3. LOAD GLOSS MAPPING
# =====================================================

def load_gloss_mapping():
    mapping = {}
    if GLOSS_CSV.exists():
        with open(GLOSS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sentence = row["Sentence"].strip().lower()
                glosses = row["SIGN GLOSSES"].strip()
                mapping[sentence] = glosses
    return mapping


# =====================================================
# 4. PROCESS SINGLE VIDEO
# =====================================================

def process_video(video_path, landmarker):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_data = []
    frame_index = 0
    previous_timestamp_ms = -1

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_index * 1000 / fps)

        if timestamp_ms <= previous_timestamp_ms:
            timestamp_ms = previous_timestamp_ms + 1
        previous_timestamp_ms = timestamp_ms

        try:
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            frame_data = {
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "hands": []
            }
            frames_data.append(frame_data)
            frame_index += 1
            continue

        frame_data = {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "hands": []
        }

        for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
            landmarks = []
            for landmark in hand_landmarks:
                landmarks.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })

            hand_info = {
                "hand_index": hand_index,
                "landmarks": landmarks
            }

            if hand_index < len(results.handedness):
                handedness = results.handedness[hand_index]
                if handedness:
                    category = handedness[0]
                    hand_info["handedness"] = category.category_name
                    hand_info["handedness_score"] = category.score

            frame_data["hands"].append(hand_info)

        frames_data.append(frame_data)
        frame_index += 1

    cap.release()

    return {
        "video_name": video_path.name,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "processed_frames": len(frames_data),
        "frames": frames_data
    }


# =====================================================
# 5. CLEAN FRAMES (trim leading/trailing empty)
# =====================================================

def clean_frames(data):
    frames = data["frames"]
    active_indices = [i for i, f in enumerate(frames) if f["hands"]]

    if not active_indices:
        return data, 0, 0

    first_active = active_indices[0]
    last_active = active_indices[-1]

    data["frames"] = frames[first_active:last_active + 1]
    data["cleaned_frame_count"] = len(data["frames"])

    return data, first_active, len(frames) - last_active - 1


# =====================================================
# 6. INTERPOLATE GAPS
# =====================================================

def interpolate_gaps(data):
    frames = data["frames"]
    frame_map = {f["frame_index"]: f for f in frames}

    missing_indices = [idx for idx, f in frame_map.items() if not f["hands"]]

    for index in missing_indices:
        prev_frame = frame_map.get(index - 1)
        next_frame = frame_map.get(index + 1)

        if not prev_frame or not next_frame:
            continue

        if len(prev_frame["hands"]) != len(next_frame["hands"]):
            continue

        interpolated_hands = []
        for prev_hand, next_hand in zip(prev_frame["hands"], next_frame["hands"]):
            if prev_hand["hand_index"] != next_hand["hand_index"]:
                continue

            interp_landmarks = []
            for p_landmark, n_landmark in zip(prev_hand["landmarks"], next_hand["landmarks"]):
                interp_landmarks.append({
                    "x": (p_landmark["x"] + n_landmark["x"]) / 2,
                    "y": (p_landmark["y"] + n_landmark["y"]) / 2,
                    "z": (p_landmark["z"] + n_landmark["z"]) / 2
                })

            hand_info = {
                "hand_index": prev_hand["hand_index"],
                "landmarks": interp_landmarks
            }
            if "handedness" in prev_hand:
                hand_info["handedness"] = prev_hand["handedness"]
            if "handedness_score" in prev_hand:
                hand_info["handedness_score"] = prev_hand["handedness_score"]

            interpolated_hands.append(hand_info)

        if interpolated_hands:
            frame_map[index]["hands"] = interpolated_hands

    data["frames"] = list(frame_map.values())
    return data


# =====================================================
# 7. MAIN BATCH PROCESSING
# =====================================================

def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    gloss_mapping = load_gloss_mapping()
    print(f"Loaded {len(gloss_mapping)} gloss mappings.")

    sentence_folders = sorted([
        d for d in SENTENCES_DIR.iterdir() if d.is_dir()
    ])
    print(f"Found {len(sentence_folders)} sentence folders.")

    sentence_to_landmarks = {}
    processed = 0
    failed = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for folder in sentence_folders:
            sentence = folder.name
            print(f"\n[{processed + 1}/{len(sentence_folders)}] {sentence}")

            video_files = [
                f for f in folder.iterdir()
                if f.is_file() and f.suffix in VIDEO_EXTENSIONS
            ]

            if not video_files:
                print(f"  No videos found, skipping.")
                failed += 1
                continue

            video_path = video_files[0]
            print(f"  Using: {video_path.name}")

            start_time = time.time()
            try:
                data = process_video(video_path, landmarker)
            except Exception as e:
                print(f"  ERROR processing video: {e}")
                failed += 1
                continue

            if data is None:
                failed += 1
                continue

            elapsed = time.time() - start_time
            print(f"  Extracted {data['processed_frames']} frames in {elapsed:.1f}s")

            data, trimmed_start, trimmed_end = clean_frames(data)
            data = interpolate_gaps(data)

            safe_name = sentence.replace(" ", "_").replace("'", "").replace(",", "")
            output_filename = f"{safe_name}_landmarks.json"
            output_path = OUTPUT_DIR / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            glosses = gloss_mapping.get(sentence.lower(), "")

            sentence_to_landmarks[sentence] = {
                "landmark_file": output_filename,
                "glosses": glosses,
                "video_count": len(video_files),
                "frame_count": data["processed_frames"],
                "trimmed_start": trimmed_start,
                "trimmed_end": trimmed_end
            }

            processed += 1
            print(f"  Saved: {output_filename}")

    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(sentence_to_landmarks, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Batch processing complete!")
    print(f"Processed: {processed}/{len(sentence_folders)}")
    print(f"Failed: {failed}")
    print(f"Mapping saved: {MAPPING_FILE}")


if __name__ == "__main__":
    main()
