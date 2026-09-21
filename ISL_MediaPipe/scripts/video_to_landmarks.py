
import cv2
import mediapipe as mp
import json
import time
from pathlib import Path


# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"

INPUT_DIR = PROJECT_DIR / "input_videos"

OUTPUT_DIR = PROJECT_DIR / "output_landmarks"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. MEDIAPIPE TASKS API
# =====================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


# =====================================================
# 3. CONFIGURATION
# =====================================================

options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=2,

    # Slightly more sensitive detection
    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)

# =====================================================
# 4. VIDEO PROCESSING FUNCTION
# =====================================================

def process_video(video_path, landmarker):

    print(f"\nProcessing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print(f"ERROR: Cannot open {video_path}")

        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:

        fps = 30.0

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frames_data = []

    frame_index = 0

    previous_timestamp_ms = -1

    start_time = time.time()

    while True:

        success, frame = cap.read()

        if not success:

            break

        # ---------------------------------------------
        # Convert BGR to RGB
        # ---------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ---------------------------------------------
        # Create MediaPipe Image
        # ---------------------------------------------

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame
        )

        # ---------------------------------------------
        # Calculate video timestamp
        # ---------------------------------------------

        timestamp_ms = int(
            frame_index * 1000 / fps
        )

        # Ensure strictly increasing timestamps

        if timestamp_ms <= previous_timestamp_ms:

            timestamp_ms = previous_timestamp_ms + 1

        previous_timestamp_ms = timestamp_ms

        # ---------------------------------------------
        # Detect hands
        # ---------------------------------------------

        results = landmarker.detect_for_video(

            mp_image,

            timestamp_ms
        )

        # ---------------------------------------------
        # Store current frame
        # ---------------------------------------------

        frame_data = {

            "frame_index": frame_index,

            "timestamp_ms": timestamp_ms,

            "hands": []

        }

        # ---------------------------------------------
        # Extract detected hands
        # ---------------------------------------------

        for hand_index, hand_landmarks in enumerate(
            results.hand_landmarks
        ):

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

            # Add handedness if available

            if hand_index < len(results.handedness):

                handedness = results.handedness[
                    hand_index
                ]

                if handedness:

                    category = handedness[0]

                    hand_info["handedness"] = (
                        category.category_name
                    )

                    hand_info["handedness_score"] = (
                        category.score
                    )

            frame_data["hands"].append(hand_info)

        frames_data.append(frame_data)

        frame_index += 1

        # ---------------------------------------------
        # Progress display
        # ---------------------------------------------

        if frame_index % 30 == 0:

            print(
                f"Processed {frame_index} frames",
                end="\r"
            )

    cap.release()

    # =================================================
    # 5. SAVE JSON
    # =================================================

    output_data = {

        "video_name": video_path.name,

        "fps": fps,

        "frame_count": frame_count,

        "width": width,

        "height": height,

        "processed_frames": len(frames_data),

        "frames": frames_data

    }

    output_path = OUTPUT_DIR / (
        video_path.stem + "_landmarks.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_data,
            f,
            indent=2
        )

    elapsed_time = time.time() - start_time

    print(
        f"\nSaved: {output_path.name}"
    )

    print(
        f"Frames processed: {len(frames_data)}"
    )

    print(
        f"Processing time: {elapsed_time:.2f} seconds"
    )


# =====================================================
# 6. MAIN EXECUTION
# =====================================================

def main():

    if not MODEL_PATH.exists():

        print(
            f"ERROR: Model not found: {MODEL_PATH}"
        )

        return

    video_extensions = {

        ".mp4",

        ".avi",

        ".mov",

        ".mkv"

    }

    video_files = [

        file for file in INPUT_DIR.iterdir()

        if file.is_file()
        and file.suffix.lower() in video_extensions

    ]

    if not video_files:

        print(
            "No video files found in input_videos."
        )

        return

    print(
        f"Found {len(video_files)} video(s)."
    )

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        for video_path in video_files:

            process_video(
                video_path,

                landmarker
            )

    print("\nAll videos processed successfully.")


if __name__ == "__main__":

    main()