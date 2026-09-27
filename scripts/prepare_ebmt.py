"""Build the EBMT multi-event dataset for the generic `multievent` experiment group.

EBMT (European Society for Blood and Marrow Transplantation; mstate::ebmt4, n = 2279
in the full data) records, per patient, the times of five non-exclusive events after
transplant: recovery (rec), adverse event (ae), recovery + adverse event (recae),
relapse (rel) and death (srv). Death ends follow-up for the others, so this is real
semi-competing multi-event data.

Covariates are categorical (year of transplant, age class, prophylaxis, donor-recipient
gender match) and are written as ``c_*`` columns, i.e. they are encoded with the
paper's categorical token embedding (e.g. ``c_agecl_gt40``), not one-hot numbers.

    python scripts/prepare_ebmt.py                    # all 5 events -> data/ebmt/
    python scripts/prepare_ebmt.py --no-composite     # drop recae   -> data/ebmt4ev/

``recae`` is a deterministic composite of rec and ae (it happens when both have
happened), which MENSA's evaluation keeps; ``--no-composite`` gives the cleaner
4-event version.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

URL = "https://vincentarelbundock.github.io/Rdatasets/csv/mstate/ebmt4.csv"
EVENTS = [
    ("rec", "rec.s", "recovery"),
    ("ae", "ae.s", "adverse_event"),
    ("recae", "recae.s", "recovery_and_adverse_event"),
    ("rel", "rel.s", "relapse"),
    ("srv", "srv.s", "death"),
]
COVARIATES = ["year", "agecl", "proph", "match"]


def build(raw: pd.DataFrame, composite: bool = True) -> tuple:
    events = [e for e in EVENTS if composite or e[0] != "recae"]
    out = pd.DataFrame({"id": range(len(raw))})
    for c in COVARIATES:
        out[f"c_{c}"] = raw[c].astype(str).values
    for k, (dcol, scol, _) in enumerate(events, start=1):
        out[f"event{k}"] = raw[scol].astype(int).values
        # a handful of zero durations would be dropped by the label transform
        out[f"duration{k}"] = raw[dcol].astype(float).clip(lower=0.5).values
    ev = out[[f"event{k}" for k in range(1, len(events) + 1)]]
    meta = {
        "source": URL,
        "num_events": len(events),
        "num_features": len(COVARIATES),
        "event_names": [e[2] for e in events],
        "max_time": float(out[[f"duration{k}" for k in range(1, len(events) + 1)]].values.max()),
        "n": int(len(out)),
        "event_rates": ev.mean().round(4).tolist(),
        "frac_subjects_multiple_events": float((ev.sum(axis=1) > 1).mean()),
        "scenario": {"regime": "semi-competing"},
    }
    return out, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=URL, help="URL or local path of ebmt4.csv")
    ap.add_argument("--no-composite", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    raw = pd.read_csv(args.source)
    df, meta = build(raw, composite=not args.no_composite)
    out = Path(args.out or ("data/ebmt4ev" if args.no_composite else "data/ebmt"))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "data.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
