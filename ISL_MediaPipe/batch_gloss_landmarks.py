"""
Build per-gloss landmark clips for the avatar.

For each target gloss key, processes that gloss's CISLR video(s) through
MediaPipe (same pipeline as batch_process.py) and writes a trimmed,
gap-interpolated landmark clip:

    output_landmarks/gloss_<key>.json

plus the lookup table:

    gloss_landmarks.json   { gloss_key: {file, uid, ...} }

Only clips with real hand motion are kept (hands_frames >= 5 and
detection_rate >= 0.10); the rest are recorded as "no_landmarks" so the
avatar skips those tokens instead of playing an empty clip.

Resume-safe: existing outputs listed in the mapping are skipped unless
--force is given. Chunked runs: --limit N --offset M.

    .venv\\Scripts\\python.exe batch_gloss_landmarks.py --targets ..\\gloss_targets.txt
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

OUTPUT_DIR = PROJECT_DIR / "output_landmarks"
MAPPING_FILE = PROJECT_DIR / "gloss_landmarks.json"
CISLR_CSV = (
    PROJECT_DIR.parent / "Dataset" / "data" / "cislr" / "dataset.csv"
)
VIDEOS_DIR = (
    PROJECT_DIR.parent / "Dataset" / "data" / "cislr" / "CISLR_v1.5-a_videos"
)

MIN_HANDS_FRAMES = 5
MIN_DETECTION_RATE = 0.10
MAX_VIDEOS_PER_GLOSS = 2

VIDEO_EXTENSIONS = {".mp4", ".MP4"}


def safe_key(key: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in key.lower()).strip("_")


def load_gloss_videos():
    """gloss key -> [(uid, duration)] sorted shortest-first."""
    videos = {}
    with open(CISLR_CSV, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            gloss = (row.get("gloss") or "").strip().lower()
            uid = (row.get("uid") or "").strip()
            if not gloss or not uid:
                continue
            try:
                duration = float((row.get("duration") or "0").strip())
            except ValueError:
                duration = 0.0
            videos.setdefault(gloss, []).append((uid, duration))
    for gloss in videos:
        videos[gloss].sort(key=lambda t: (t[1] or 1e9, t[0]))
    return videos


def load_mapping():
    if MAPPING_FILE.exists():
        try:
            return json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def valid_clip_file(path: Path) -> bool:
    """An existing clip is reusable when it parses with real frames."""
    try:
        if not path.exists() or path.stat().st_size < 1024:
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data.get("frames") if isinstance(data, dict) else None
        return bool(frames) and any(f.get("hands") for f in frames)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="file with one gloss key per line")
    ap.add_argument("--limit", type=int, default=0, help="max glosses this run (0 = all)")
    ap.add_argument("--offset", type=int, default=0, help="skip first N targets")
    ap.add_argument("--force", action="store_true", help="reprocess existing outputs")
    args = ap.parse_args()

    targets = [
        line.strip().lower()
        for line in Path(args.targets).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.offset:
        targets = targets[args.offset:]
    if args.limit:
        targets = targets[: args.limit]
    print(f"targets this run: {len(targets)}")

    from batch_process import (
        HandLandmarker,
        clean_frames,
        interpolate_gaps,
        options,
        process_video,
    )

    gloss_videos = load_gloss_videos()
    mapping = load_mapping()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts_offset_ms = 0
    done = skipped = failed = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for index, key in enumerate(targets, start=1):
            fname = f"gloss_{safe_key(key)}.json"
            # File-aware resume: adopt usable clips from interrupted runs
            # instead of reprocessing them (the mapping is only written
            # at the end of a run, so killed runs leave orphans).
            if not args.force and valid_clip_file(OUTPUT_DIR / fname):
                entry = mapping.get(key, {})
                entry.update({"status": "ready", "file": fname})
                mapping[key] = entry
                skipped += 1
                continue
            if key in mapping and mapping[key].get("status") == "ready" and not args.force:
                out = OUTPUT_DIR / mapping[key].get("file", "")
                if out.exists():
                    skipped += 1
                    continue

            cands = gloss_videos.get(key, [])
            if not cands:
                mapping[key] = {"status": "no_video", "file": ""}
                failed += 1
                continue

            best = None  # (hands_frames, data, uid, fname)
            tried = 0
            for uid, _dur in cands:
                if tried >= MAX_VIDEOS_PER_GLOSS:
                    break
                video_path = VIDEOS_DIR / f"{uid}.mp4"
                if not video_path.exists():
                    continue
                tried += 1
                t0 = time.time()
                data, hands, ts_offset_ms = process_video(video_path, landmarker, ts_offset_ms)
                el = time.time() - t0
                if data is None:
                    continue
                total = data["processed_frames"] or 1
                print(f"  [{index}/{len(targets)}] {key}: {uid} "
                      f"hands={hands}/{total} ({hands / total:.0%}) in {el:.1f}s")
                if best is None or hands > best[0]:
                    best = (hands, data, uid, video_path.name)
                if hands / total >= 0.70:
                    break

            if best is None:
                mapping[key] = {"status": "no_landmarks", "file": ""}
                failed += 1
                continue

            hands, data, uid, _vname = best
            data, _ts, _te = clean_frames(data)
            data = interpolate_gaps(data)
            hands_after = sum(1 for f in data["frames"] if f["hands"])
            total_after = len(data["frames"]) or 1
            rate = hands_after / total_after

            if hands_after < MIN_HANDS_FRAMES or rate < MIN_DETECTION_RATE:
                mapping[key] = {
                    "status": "no_landmarks",
                    "file": "",
                    "uid": uid,
                    "hands_frames": hands_after,
                    "detection_rate": round(rate, 4),
                }
                failed += 1
                print(f"    -> rejected (too little motion)")
                continue

            fname = f"gloss_{safe_key(key)}.json"
            with open(OUTPUT_DIR / fname, "w", encoding="utf-8") as f:
                json.dump(data, f)
            mapping[key] = {
                "status": "ready",
                "file": fname,
                "uid": uid,
                "frame_count": len(data["frames"]),
                "hands_frames": hands_after,
                "detection_rate": round(rate, 4),
            }
            done += 1
            print(f"    -> saved {fname} ({hands_after}/{total_after})")

    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"\ndone={done} skipped={skipped} failed={failed} mapping={MAPPING_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
