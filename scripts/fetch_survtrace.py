"""Clone the authors' SurvTRACE code (Wang & Sun, BCB '22; MIT licence) at a pinned
commit into third_party/SurvTRACE, so the official-SurvTRACE baseline runs the released
implementation, not a re-implementation.

    python scripts/fetch_survtrace.py
"""

__authors__ = ["Parsa"]

import subprocess
import sys
from pathlib import Path

URL = "https://github.com/RyanWangZf/SurvTRACE.git"
COMMIT = "e6b354fd61d9505f4026a92df7bd83a5dfd24ff4"  # main, checked 2026-10-04
DEST = Path(__file__).resolve().parents[1] / "third_party" / "SurvTRACE"


def fetch(dest: Path = DEST) -> Path:
    if (dest / "survtrace" / "model.py").is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", URL, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", COMMIT], check=True)
    return dest


if __name__ == "__main__":
    print(fetch())
    sys.exit(0)
