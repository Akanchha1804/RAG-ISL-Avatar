"""Motion-data audit (M6): validate every landmark file + report quality.

Read-only: walks the landmark directory, validates structure
(landmarks.validate_dataset: frames list, 21 points/hand), and reports
empty-frame ratio, longest empty run, handedness gaps, fps coverage and
detection-rate distribution. Exit code 0 always; failures print as rows.

    .venv\\Scripts\\python.exe eval/run_motion_audit.py [landmarks_dir]
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from landmarks import validate_dataset, LandmarkValidationError  # noqa: E402


def audit_file(path: Path) -> dict:
    row = {"file": path.name, "status": "ok"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"file": path.name, "status": f"unreadable: {e}"}
    try:
        stats = validate_dataset(data)
    except LandmarkValidationError as e:
        return {"file": path.name, "status": f"invalid: {e}"}
    row.update(stats)
    frames = data["frames"]
    row["fps"] = data.get("fps")
    empty_runs, run = [], 0
    no_side = 0
    for f in frames:
        hands = (f or {}).get("hands") or []
        if not hands:
            run += 1
            continue
        if run:
            empty_runs.append(run)
            run = 0
        for h in hands:
            if not (h or {}).get("handedness"):
                no_side += 1
    if run:
        empty_runs.append(run)
    row["empty_runs"] = len(empty_runs)
    row["longest_empty_run"] = max(empty_runs) if empty_runs else 0
    row["empty_ratio"] = round(
        sum(1 for f in frames if not ((f or {}).get("hands") or [])) / len(frames), 3)
    row["hands_without_handedness"] = no_side
    # Wrist height in raw image coords (y down): mean > 0.7 means the signer
    # performs low in frame. The solvers normalize each clip into signing
    # space, so this is a DATA property readout, not a failure — but clips
    # far outside 0.3-0.7 deserve a human look (framing or genuinely low
    # signing that normalization will lift).
    wrist_ys = []
    for f in frames:
        for h in (f or {}).get("hands") or []:
            lm = (h or {}).get("landmarks") or []
            if lm and isinstance(lm[0], dict) and "y" in lm[0]:
                wrist_ys.append(lm[0]["y"])
    row["wrist_y_mean"] = round(sum(wrist_ys) / len(wrist_ys), 3) if wrist_ys else None
    return row


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        BACKEND_DIR.parent / "ISL_MediaPipe" / "output_landmarks")
    files = sorted(root.glob("*.json"))
    rows = [audit_file(p) for p in files]
    bad = [r for r in rows if r["status"] != "ok"]
    ok = [r for r in rows if r["status"] == "ok"]

    total_frames = sum(r.get("frame_count", 0) for r in ok)
    empty_total = sum(int(r.get("empty_ratio", 0) * r.get("frame_count", 0)) for r in ok)
    no_side = sum(r.get("hands_without_handedness", 0) for r in ok)
    rates = sorted(r.get("frames_with_hands", 0) / r.get("frame_count", 1) for r in ok)

    print(f"files={len(rows)} ok={len(ok)} bad={len(bad)}")
    for r in bad:
        print(f"  BAD {r['file']}: {r['status']}")
    if ok:
        print(f"frames={total_frames} empty_ratio={empty_total / total_frames:.3f} "
              f"hands_without_handedness={no_side}")
        print(f"detection_rate p10={rates[len(rates)//10]:.2f} "
              f"median={rates[len(rates)//2]:.2f} "
              f"min={rates[0]:.2f}")
        worst = sorted(ok, key=lambda r: r.get("empty_ratio", 0), reverse=True)[:5]
        print("worst empty_ratio:")
        for r in worst:
            print(f"  {r['file']}: {r['empty_ratio']} "
                  f"(longest run {r['longest_empty_run']}, fps={r['fps']})")
        no_fps = [r["file"] for r in ok if not r.get("fps")]
        if no_fps:
            print(f"missing fps: {len(no_fps)} e.g. {no_fps[:3]}")
        low = sorted(
            (r for r in ok if (r.get("wrist_y_mean") or 0) > 0.7),
            key=lambda r: r["wrist_y_mean"], reverse=True)[:8]
        if low:
            print("low signing (wrist_y_mean > 0.7 — lifted by solver normalization):")
            for r in low:
                print(f"  {r['file']}: y={r['wrist_y_mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
