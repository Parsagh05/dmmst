"""Package the repository as a zip to upload as the `dmmst-code` Kaggle dataset.

    python scripts/make_kaggle_zip.py            # -> ../dmmst_code.zip

Skips everything that is generated or environment-specific (venv, git history, model
outputs, results, generated datasets, caches, logs). The bundled METABRIC / SUPPORT /
hsa_synthetic data are included.
"""

import hashlib
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".hypothesis", "tb_logs",
             "test-trainer-output", "results", "results_survtrace", "logs", "model-hub",
             "outputs", "multirun", "optuna"}
SKIP_PREFIXES = ("data/sim_", "data/ebmt", "data/ehrsim_", "notebooks/version_1/_smoke_")
SKIP_SUFFIXES = (".log", ".pyc", ".zip")


def included(rel: Path) -> bool:
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    s = rel.as_posix()
    return not s.startswith(SKIP_PREFIXES) and not s.endswith(SKIP_SUFFIXES) and s != "summery.md"


def main(out=None):
    out = Path(out or REPO.parent / "dmmst_code.zip")
    files = sorted(p for p in REPO.rglob("*") if p.is_file() and included(p.relative_to(REPO)))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path("dmmst") / f.relative_to(REPO))
    # same fingerprint the notebooks print (sat/*.py + conf/*.yaml)
    fp = hashlib.md5(b"".join(
        p.read_bytes() for p in sorted((REPO / "sat").rglob("*.py")) + sorted((REPO / "conf").rglob("*.yaml"))
    )).hexdigest()[:10]
    print(f"{out}  ({len(files)} files, {out.stat().st_size / 1e6:.1f} MB, code fingerprint {fp})")


if __name__ == "__main__":
    main(*sys.argv[1:])
