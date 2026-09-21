
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output_landmarks"

cleaned_files = list(OUTPUT_DIR.glob("*_cleaned.json"))

if not cleaned_files:
    print("Cleaned JSON file not found.")
    exit()

INPUT_FILE = cleaned_files[0]

OUTPUT_FILE = OUTPUT_DIR / (
    INPUT_FILE.stem.replace("_cleaned", "")
    + "_interpolated.json"
)


with open(INPUT_FILE, "r", encoding="utf-8") as file:
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


def interpolate_landmarks(previous, next_frame):
    interpolated = []

    for prev_landmark, next_landmark in zip(
        previous, next_frame
    ):
        interpolated.append({
            "x": (prev_landmark["x"] + next_landmark["x"]) / 2,
            "y": (prev_landmark["y"] + next_landmark["y"]) / 2,
            "z": (prev_landmark["z"] + next_landmark["z"]) / 2
        })

    return interpolated


for index in missing_indices:

    previous_frame = frame_map.get(index - 1)
    next_frame = frame_map.get(index + 1)

    if not previous_frame or not next_frame:
        continue

    if (
        len(previous_frame["hands"]) !=
        len(next_frame["hands"])
    ):
        print(
            f"Skipped frame {index}: "
            "different hand counts."
        )
        continue

    interpolated_hands = []

    for prev_hand, next_hand in zip(
        previous_frame["hands"],
        next_frame["hands"]
    ):

        if (
            prev_hand["hand_index"] !=
            next_hand["hand_index"]
        ):
            continue

        hand_info = {
            "hand_index": prev_hand["hand_index"],
            "landmarks": interpolate_landmarks(
                prev_hand["landmarks"],
                next_hand["landmarks"]
            )
        }

        if "handedness" in prev_hand:
            hand_info["handedness"] = prev_hand["handedness"]

        if "handedness_score" in prev_hand:
            hand_info["handedness_score"] = (
                prev_hand["handedness_score"]
            )

        interpolated_hands.append(hand_info)

    if interpolated_hands:
        frame_map[index]["hands"] = interpolated_hands
        print(f"Interpolated frame: {index}")


data["frames"] = list(frame_map.values())

data["interpolation_note"] = (
    "Missing frames with compatible neighboring "
    "hand structures were linearly interpolated."
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=2)

print("\nInterpolation completed.")
print(f"Output file: {OUTPUT_FILE.name}")