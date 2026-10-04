"""Longitudinal (coded, time-stamped) records -> SAT json (paper §2.1 "sequence" inputs, §4.2).

Input: a directory with

* ``patients.csv`` - ``id``, static ``x_*`` (numeric) / ``c_*`` (categorical) columns,
  ``event1..K``, ``duration1..K`` (time from the index date);
* ``events.csv``   - long table ``id, time, code, value``: one row per code, ``time`` in
  days relative to the index date (<= 0), ``value`` numeric for measurements (labs) and
  empty otherwise. ``VISIT`` rows mark encounters (optional).

Any real EHR extract (e.g. the CKD / COPD / diabetes cohorts of §4.2, MIMIC) can be
exported to these two files; ``sat.data.simulate_ehr`` writes them for synthetic data.

Representations (``representation``), same patients and outcomes:

* ``sequence`` - static tokens, then the history in time order. ``VISIT`` carries the
  time before index (years) as its numeric value and labs carry their standardised
  value, i.e. time and measurements enter through the paper's numeric-value embedding
  (§2.2, Fig. 2); plain codes are tokens. The most recent ``max_tokens`` are kept.
* ``bag``      - classic tabular summary: count of every code, last value of every
  measured code, number of visits. What a non-sequential model would be given.
* ``static``   - static columns only.
* ``sequence_time`` - as ``sequence``, but plain codes also carry their recency
  ``exp(t / 1 year)`` (1 at the index date, ~0.007 five years before) through the same
  numeric-value embedding, so every token - not only ``VISIT`` - knows when it happened.

The ``bag``/``static`` tables go through the generic tabular parser
(``parse_multievent``), so the three differ only in what the model is shown.
"""

__authors__ = ["Parsa"]
__status__ = "Development"

import argparse
import json
import re
from dataclasses import dataclass
from logging import DEBUG
from pathlib import Path

import numpy as np
import pandas as pd
from logdecorator import log_on_end, log_on_start
from sklearn.preprocessing import StandardScaler

from sat.data.dataset.parse_multievent import _clean, multievent
from sat.utils import logging

logger = logging.get_default_logger()

REPRESENTATIONS = ("sequence", "sequence_time", "bag", "static")


def _read(source_dir):
    src = Path(source_dir)
    patients = pd.read_csv(src / "patients.csv")
    events = pd.read_csv(src / "events.csv")
    events = events.sort_values(["id", "time"], kind="stable")
    events["code"] = events["code"].astype(str).map(lambda c: re.sub(r"\s+", "_", c))
    return patients, events


def _static_cols(patients):
    num = [c for c in patients.columns if c.startswith("x_")]
    cat = [c for c in patients.columns if c.startswith("c_")]
    return num, cat


def _bag(patients, events):
    """Counts per code, last value per measured code, number of visits."""
    ev = events[events["code"] != "VISIT"]
    counts = ev.pivot_table(index="id", columns="code", values="time", aggfunc="count", fill_value=0)
    counts.columns = [f"x_n_{c}" for c in counts.columns]
    meas = ev.dropna(subset=["value"])
    last = meas.groupby(["id", "code"])["value"].last().unstack()
    last = last.fillna(last.mean())  # never measured -> population mean
    last.columns = [f"x_last_{c}" for c in last.columns]
    visits = events[events["code"] == "VISIT"].groupby("id").size().rename("x_n_visits")
    bag = pd.concat([counts, last, visits], axis=1).reindex(patients["id"]).fillna(0.0)
    return bag.reset_index(drop=True)


def describe(source_dir, max_tokens: int = 256) -> dict:
    """Number of model input positions per representation (for the tokenizer)."""
    patients, events = _read(source_dir)
    num, cat = _static_cols(patients)
    n_static = len(num) + len(cat)
    longest = int(events.groupby("id").size().max()) if len(events) else 0
    return {
        "num_features": {
            "sequence": int(min(max_tokens, n_static + longest)),
            "sequence_time": int(min(max_tokens, n_static + longest)),
            "bag": int(_bag(patients, events).shape[1] + n_static),
            "static": int(n_static),
        },
        "n": int(len(patients)),
        "n_codes": int(events["code"].nunique()),
        "max_tokens": int(max_tokens),
    }


@dataclass
class sequence:
    source_dir: str
    processed_dir: str
    name: str
    num_events: int
    representation: str = "sequence"
    max_tokens: int = 256

    @log_on_start(DEBUG, "Create longitudinal data representation...")
    @log_on_end(DEBUG, "done!")
    def __call__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        patients, events = _read(self.source_dir)
        out_dir = Path(f"{self.processed_dir}/{self.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        outcome_cols = [f"{p}{k + 1}" for k in range(self.num_events) for p in ("event", "duration")]
        num, cat = _static_cols(patients)

        if self.representation in ("bag", "static"):
            table = patients[["id"] + cat + num + outcome_cols].copy()
            if self.representation == "bag":
                table = pd.concat([table.reset_index(drop=True), _bag(patients, events)], axis=1)
            tmp = out_dir / f"_{self.representation}_table.csv"
            table.to_csv(tmp, index=False)
            multievent(source=str(tmp), processed_dir=self.processed_dir, name=self.name,
                       num_events=self.num_events)()
            return

        # ---- sequence ---------------------------------------------------------
        stat_num = pd.DataFrame(StandardScaler().fit_transform(patients[num].astype(float)),
                                columns=num) if num else pd.DataFrame(index=patients.index)
        # per-code standardisation of measured values
        ev = events.copy()
        has = ev["value"].notna() & (ev["code"] != "VISIT")
        stats = ev[has].groupby("code")["value"].agg(["mean", "std"])
        mu = ev["code"].map(stats["mean"])
        sd = ev["code"].map(stats["std"]).replace(0, 1.0).fillna(1.0)
        ev["num"] = np.where(has, (ev["value"] - mu) / sd, 1.0)
        ev.loc[ev["code"] == "VISIT", "num"] = ev.loc[ev["code"] == "VISIT", "time"] / 365.25
        ev["mod"] = np.where(has | (ev["code"] == "VISIT"), 1, 0)
        if self.representation == "sequence_time":
            plain = ~has & (ev["code"] != "VISIT")
            ev.loc[plain, "num"] = np.exp(ev.loc[plain, "time"] / 365.25)
            ev.loc[plain, "mod"] = 1
        by_id = {i: g for i, g in ev.groupby("id", sort=False)}

        budget = self.max_tokens - len(num) - len(cat)
        if budget <= 0:
            raise ValueError("max_tokens must exceed the number of static features")
        xs, mods, nums = [], [], []
        for row, pid in enumerate(patients["id"].values):
            toks = [f"{c}_{_clean(patients.at[row, c])}" for c in cat] + num
            mod = [0] * len(cat) + [1] * len(num)
            val = [1.0] * len(cat) + [float(v) for v in stat_num.iloc[row].values]
            g = by_id.get(pid)
            if g is not None:
                g = g.iloc[-budget:]  # the most recent history
                toks += g["code"].tolist()
                mod += g["mod"].astype(int).tolist()
                val += g["num"].astype(float).tolist()
            xs.append(" ".join(toks))
            mods.append(mod)
            nums.append(val)

        data = pd.DataFrame({
            "id": patients["id"].values,
            "x": xs,
            "modality": mods,
            "numerics": nums,
            "events": patients[[f"event{k + 1}" for k in range(self.num_events)]].astype(int).values.tolist(),
            "durations": patients[[f"duration{k + 1}" for k in range(self.num_events)]].astype(float).values.tolist(),
        })
        data.to_json(out_dir / f"{self.name}.json", orient="records", lines=True)
        lens = [len(m) for m in mods]
        logger.info(f"Wrote {len(data)} sequences (tokens: median {int(np.median(lens))}, "
                    f"max {max(lens)}) -> {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Describe a longitudinal dataset and write meta.json")
    ap.add_argument("source_dir")
    ap.add_argument("--num-events", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args(argv)
    src = Path(args.source_dir)
    meta_f = src / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.is_file() else {}
    pats = pd.read_csv(src / "patients.csv")
    meta.update(describe(src, args.max_tokens))
    meta.update({
        "kind": "sequence",
        "num_events": args.num_events,
        "max_time": float(pats[[f"duration{k + 1}" for k in range(args.num_events)]].values.max()),
    })
    meta_f.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
