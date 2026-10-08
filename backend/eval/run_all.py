"""Run all three eval harnesses independently (M4 Phase 4).

    .venv\\Scripts\\python.exe eval/run_all.py

Writes backend/eval/results/{translation,retrieval,animation}_eval.json.
Each harness is independent: retrieval misses never lower translation
scores. Translation numbers are provisional until refs lock.
"""

import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def main() -> int:
    code = 0
    for script in ("run_translation_eval.py", "run_retrieval_eval.py",
                   "run_animation_eval.py"):
        print(f"=== {script} ===")
        proc = subprocess.run([sys.executable, str(BACKEND_DIR / "eval" / script)],
                              cwd=str(BACKEND_DIR))
        code = code or proc.returncode
    return code


if __name__ == "__main__":
    sys.exit(main())
