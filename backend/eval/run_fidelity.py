"""Signing-space fidelity harness (M6): mirror of the solver mapping.

Reimplements frontend calibrateSigningSpace/normalizeWrist (VrmAvatar.jsx)
and Unity NormalizeWrist (HandPoseMapper.cs) in Python with the SAME
constants, then asserts every landmark file maps inside the clamped box.
Also reports framing diversity (raw wrist bbox centers) and min-span
guard reliance per axis.

    .venv\\Scripts\\python.exe eval/run_fidelity.py
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import metrics_eval  # noqa: E402

MIN_SPAN_XY = 0.12
MIN_SPAN_Z = 0.05


def calibrate(frames):
    xs, ys, zs = [], [], []
    for frame in frames:
        for hand in (frame or {}).get("hands") or []:
            lms = (hand or {}).get("landmarks") or []
            if not lms:
                continue
            w = lms[0]
            xs.append(w["x"])
            ys.append(w["y"])
            zs.append(w["z"])
    if not xs:
        return None

    # Percentile bbox, mirroring the solvers (edge mistracks excluded).
    def pct(arr, q):
        s = sorted(arr)
        return s[min(len(s) - 1, int(q * len(s)))]

    min_x, max_x = pct(xs, 0.02), pct(xs, 0.98)
    min_y, max_y = pct(ys, 0.02), pct(ys, 0.98)
    min_z, max_z = pct(zs, 0.02), pct(zs, 0.98)

    def fit(lo, hi, span_min):
        span = max(hi - lo, span_min)
        mid = (lo + hi) / 2
        return mid - span / 2, span

    x0, sx = fit(min_x, max_x, MIN_SPAN_XY)
    y0, sy = fit(min_y, max_y, MIN_SPAN_XY)
    z0, sz = fit(min_z, max_z, MIN_SPAN_Z)
    return {"min": (x0, y0, z0), "span": (sx, sy, sz),
            "raw_center": ((min_x + max_x) / 2, (min_y + max_y) / 2,
                           (min_z + max_z) / 2),
            "raw_span": (max_x - min_x, max_y - min_y, max_z - min_z)}


def normalize(calib, w):
    def clamp(v):
        return min(1.2, max(-0.2, v))

    if calib is None:
        return (w["x"], w["y"], w["z"])
    out = []
    for i, k in enumerate("xyz"):
        out.append(clamp((w[k] - calib["min"][i]) / calib["span"][i]))
    return tuple(out)


def main() -> int:
    root = BACKEND_DIR.parent / "ISL_MediaPipe" / "output_landmarks"
    files = sorted(root.glob("*.json"))
    n_files = guard_x = guard_y = guard_z = 0
    violations = []
    centers_y = []
    for path in files:
        try:
            frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
        except (OSError, ValueError, KeyError):
            violations.append({"file": path.name, "reason": "unreadable"})
            continue
        calib = calibrate(frames)
        if calib is None:
            violations.append({"file": path.name, "reason": "no wrist data"})
            continue
        n_files += 1
        rs = calib["raw_span"]
        if rs[0] < MIN_SPAN_XY:
            guard_x += 1
        if rs[1] < MIN_SPAN_XY:
            guard_y += 1
        if rs[2] < MIN_SPAN_Z:
            guard_z += 1
        centers_y.append(calib["raw_center"][1])
        for frame in frames:
            for hand in (frame or {}).get("hands") or []:
                lms = (hand or {}).get("landmarks") or []
                if not lms:
                    continue
                n = normalize(calib, lms[0])
                if not all(-0.2 - 1e-9 <= v <= 1.2 + 1e-9 for v in n):
                    violations.append({"file": path.name,
                                       "reason": f"out of box: {n}"})
                    break
            else:
                continue
            break

    centers_y.sort()
    payload = {
        "files": n_files,
        "violations": violations,
        "guard_reliance": {
            "x": round(guard_x / n_files, 3) if n_files else 0,
            "y": round(guard_y / n_files, 3) if n_files else 0,
            "z": round(guard_z / n_files, 3) if n_files else 0,
        },
        "raw_wrist_center_y": {
            "min": round(centers_y[0], 3),
            "median": round(centers_y[len(centers_y) // 2], 3),
            "max": round(centers_y[-1], 3),
        } if centers_y else {},
    }
    path = metrics_eval.save_result("fidelity", payload)
    print(f"files={n_files} violations={len(violations)} "
          f"guard_xy=({payload['guard_reliance']['x']},"
          f"{payload['guard_reliance']['y']}) "
          f"center_y range={payload['raw_wrist_center_y']} -> {path}")
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
