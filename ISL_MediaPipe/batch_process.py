"""
Batch process all ISL-CSLTR sentence videos through MediaPipe.
Extracts hand landmarks, cleans, and interpolates gaps.

Two correctness rules this script must obey:

1. MediaPipe VIDEO-mode landmarker timestamps must be monotonically
   increasing across *every* detect_for_video() call on a given landmarker
   instance - not merely within one video. The original implementation
   restarted timestamps at 0 for each video while reusing a single
   landmarker, so every frame after the first video raised
   "Input timestamp must be monotonically increasing". The exception was
   swallowed and an empty frame was written instead, which is why 97 of 102
   output files contained no hands at all. This version carries a running
   timestamp offset across videos.

2. A sentence is only marked "ready" when the written file actually
   contains hand landmarks, not merely because a file exists.
"""

import csv
import json
import sys
import time
from pathlib import Path

# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
OUTPUT_DIR = PROJECT_DIR / "output_landmarks"
MAPPING_FILE = PROJECT_DIR / "sentence_mapping.json"

# The dataset can live in several places depending on the machine. Try an
# explicit override first, then the known layouts, then give a clear error.
_CANDIDATE_ROOTS = [
    Path(r"F:\FY PROJECT\Dataset\data\isl_csltr\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"),
    PROJECT_DIR.parent
    / "Dataset" / "data" / "isl_csltr" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus",
    Path.home() / "FY PROJECT" / "Dataset" / "data" / "isl_csltr"
    / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus",
]


def resolve_dataset_dir() -> Path:
    override = __import__("os").environ.get("ISL_CSLTR_DIR")
    if override:
        p = Path(override)
        if p.is_dir():
            return p
        raise SystemExit(f"ISL_CSLTR_DIR is set but not a directory: {p}")
    for candidate in _CANDIDATE_ROOTS:
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        "Could not locate the ISL-CSLTR corpus. Set ISL_CSLTR_DIR to the "
        "folder containing Videos_Sentence_Level/.\nTried:\n  "
        + "\n  ".join(str(c) for c in _CANDIDATE_ROOTS)
    )


DATASET_DIR = resolve_dataset_dir()
SENTENCES_DIR = DATASET_DIR / "Videos_Sentence_Level"
GLOSS_CSV = DATASET_DIR / "corpus_csv_files" / "ISL Corpus sign glosses.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. MEDIAPIPE SETUP
# =====================================================

import cv2
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
    min_tracking_confidence=0.5,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".MOV"}

# Stop searching a sentence folder once this share of frames has hands.
DETECTION_TARGET = 0.70
# Never read more than this many videos per sentence (keeps runtime bounded).
MAX_VIDEOS_PER_SENTENCE = 5
# Minimum share of frames with hands for a file to be recorded as usable.
USABLE_DETECTION_RATE = 0.05

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
    else:
        print(f"WARNING: gloss CSV not found: {GLOSS_CSV}")
    return mapping


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def lookup_gloss(sentence: str, mapping: dict) -> str:
    """Gloss lookup tolerant of filesystem folder mangling.

    Corpus folders are created from sentence text, so a sentence containing
    punctuation (e.g. "which college/school are you from") loses that
    character in the folder name. Fall back to an alphanumeric-only
    comparison before giving up.
    """
    key = sentence.lower()
    if key in mapping:
        return mapping[key]
    wanted = _normalize(key)
    for candidate, glosses in mapping.items():
        if _normalize(candidate) == wanted:
            return glosses
    return ""


# =====================================================
# 4. PROCESS SINGLE VIDEO
# =====================================================


def process_video(video_path, landmarker, ts_offset_ms):
    """Run detection on one video.

    Returns (data, frames_with_hands, next_ts_offset_ms). Timestamps start at
    ts_offset_ms so they stay strictly increasing across the whole batch.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return None, 0, ts_offset_ms

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_data = []
    frame_index = 0
    previous_timestamp_ms = ts_offset_ms - 1
    frames_with_hands = 0
    errors = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(ts_offset_ms + frame_index * 1000 / fps)

        if timestamp_ms <= previous_timestamp_ms:
            timestamp_ms = previous_timestamp_ms + 1
        previous_timestamp_ms = timestamp_ms

        try:
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  detection error at frame {frame_index}: {e}")
            frames_data.append({
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "hands": [],
            })
            frame_index += 1
            continue

        frame_data = {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "hands": [],
        }

        for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
            landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks
            ]
            hand_info = {"hand_index": hand_index, "landmarks": landmarks}

            if hand_index < len(results.handedness):
                handedness = results.handedness[hand_index]
                if handedness:
                    category = handedness[0]
                    hand_info["handedness"] = category.category_name
                    hand_info["handedness_score"] = category.score

            frame_data["hands"].append(hand_info)

        if frame_data["hands"]:
            frames_with_hands += 1

        frames_data.append(frame_data)
        frame_index += 1

    cap.release()

    if errors and errors == frame_index:
        print(f"  ERROR: every frame failed for {video_path.name}")
        return None, 0, previous_timestamp_ms + 1

    data = {
        "video_name": video_path.name,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "processed_frames": len(frames_data),
        "frames": frames_data,
    }
    return data, frames_with_hands, previous_timestamp_ms + 1


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

            interp_landmarks = [
                {
                    "x": (p["x"] + n["x"]) / 2,
                    "y": (p["y"] + n["y"]) / 2,
                    "z": (p["z"] + n["z"]) / 2,
                }
                for p, n in zip(prev_hand["landmarks"], next_hand["landmarks"])
            ]

            hand_info = {
                "hand_index": prev_hand["hand_index"],
                "landmarks": interp_landmarks,
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
# 7. VIDEO SELECTION
# =====================================================


def candidate_videos(folder):
    """Videos to try for a sentence, cheapest/most-specific first.

    Smaller portrait clips in this corpus are the tightly cropped sign
    recordings and are far faster to process than the 1080p masters.
    """
    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix in VIDEO_EXTENSIONS
    ]
    files.sort(key=lambda f: f.stat().st_size)
    return files


def detect_sentence(folder, landmarker, ts_offset_ms):
    """Pick the best video for a sentence folder.

    Returns (data, hands_frames, video_name, ts_offset_ms) or
    (None, 0, None, ts_offset_ms) when nothing usable was found.
    """
    best = None  # (frames_with_hands, data, video_name)

    for attempt, video_path in enumerate(candidate_videos(folder)):
        if attempt >= MAX_VIDEOS_PER_SENTENCE:
            break

        print(f"  [{attempt + 1}] {video_path.name}")
        start = time.time()
        data, hands, ts_offset_ms = process_video(video_path, landmarker, ts_offset_ms)
        elapsed = time.time() - start

        if data is None:
            continue

        total = data["processed_frames"] or 1
        rate = hands / total
        print(
            f"      frames={data['processed_frames']} hands_frames={hands} "
            f"({rate:.0%}) in {elapsed:.1f}s"
        )

        if best is None or hands > best[0]:
            best = (hands, data, video_path.name)

        if rate >= DETECTION_TARGET:
            break

    if best is None:
        return None, 0, None, ts_offset_ms

    hands, data, name = best
    return data, hands, name, ts_offset_ms


# =====================================================
# 8. MAIN BATCH PROCESSING
# =====================================================


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return 1

    gloss_mapping = load_gloss_mapping()
    print(f"Loaded {len(gloss_mapping)} gloss mappings.")
    print(f"Dataset: {DATASET_DIR}")

    if not SENTENCES_DIR.is_dir():
        print(f"ERROR: Sentence video dir not found: {SENTENCES_DIR}")
        return 1

    sentence_folders = sorted(d for d in SENTENCES_DIR.iterdir() if d.is_dir())
    print(f"Found {len(sentence_folders)} sentence folders.")

    sentence_to_landmarks = {}
    processed = 0
    no_videos = 0
    no_hands = 0
    timestamp_offset_ms = 0  # keeps timestamps increasing across videos

    with HandLandmarker.create_from_options(options) as landmarker:
        for index, folder in enumerate(sentence_folders, start=1):
            sentence = folder.name
            print(f"\n[{index}/{len(sentence_folders)}] {sentence}")

            videos = candidate_videos(folder)
            if not videos:
                print("  No videos found, skipping.")
                no_videos += 1
                continue

            data, hands, video_name, timestamp_offset_ms = detect_sentence(
                folder, landmarker, timestamp_offset_ms
            )

            if data is None:
                no_hands += 1
                safe_name = safe_filename(sentence)
                sentence_to_landmarks[sentence] = {
                    "landmark_file": "",
                    "glosses": lookup_gloss(sentence, gloss_mapping),
                    "video_count": len(videos),
                    "frame_count": 0,
                    "hands_frames": 0,
                    "detection_rate": 0.0,
                    "status": "no_landmarks",
                }
                print("  No hands detected in any candidate video.")
                continue

            data, trimmed_start, trimmed_end = clean_frames(data)
            data = interpolate_gaps(data)

            hands_after = sum(1 for f in data["frames"] if f["hands"])
            total_after = len(data["frames"]) or 1
            detection_rate = round(hands_after / total_after, 4)

            safe_name = safe_filename(sentence)
            output_filename = f"{safe_name}_landmarks.json"
            output_path = OUTPUT_DIR / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            usable = hands_after > 0
            status = "ready" if usable else "no_landmarks"

            sentence_to_landmarks[sentence] = {
                "landmark_file": output_filename if usable else "",
                "glosses": lookup_gloss(sentence, gloss_mapping),
                "video_count": len(videos),
                "video_used": video_name,
                "frame_count": data["processed_frames"],
                "hands_frames": hands_after,
                "detection_rate": detection_rate,
                "trimmed_start": trimmed_start,
                "trimmed_end": trimmed_end,
                "status": status,
            }

            if usable:
                processed += 1
                print(
                    f"  Saved: {output_filename} "
                    f"({hands_after}/{total_after} frames with hands, "
                    f"{detection_rate:.0%})"
                )
            else:
                no_hands += 1
                print("  Saved file contains no hands; marked no_landmarks.")

    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(sentence_to_landmarks, f, indent=2)

    print("\n" + "=" * 50)
    print("Batch processing complete!")
    print(f"Sentences with usable landmarks: {processed}/{len(sentence_folders)}")
    print(f"Folders with no videos:          {no_videos}")
    print(f"Folders with no usable hands:    {no_hands}")
    print(f"Mapping saved: {MAPPING_FILE}")
    return 0


def safe_filename(sentence: str) -> str:
    return sentence.replace(" ", "_").replace("'", "").replace(",", "")


if __name__ == "__main__":
    sys.exit(main())
