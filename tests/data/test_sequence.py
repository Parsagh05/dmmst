import dataclasses
import json

import numpy as np
import pandas as pd

from sat.data.dataset.parse_sequence import sequence
from sat.data.simulate import true_cif
from sat.data.simulate_ehr import SCENARIOS, save


def _cohort(tmp_path, **kw):
    sc = dataclasses.replace(SCENARIOS["base"], n=300, **kw)
    meta = save(sc, tmp_path / "src", max_tokens=48)
    return meta


def _parse(tmp_path, rep, max_tokens=48):
    sequence(source_dir=str(tmp_path / "src"), processed_dir=str(tmp_path / "hub"), name=rep,
             num_events=4, representation=rep, max_tokens=max_tokens)()
    return pd.read_json(tmp_path / "hub" / rep / f"{rep}.json", lines=True)


def test_representations_are_aligned_and_share_outcomes(tmp_path):
    meta = _cohort(tmp_path)
    frames = {rep: _parse(tmp_path, rep) for rep in ("sequence", "bag", "static")}
    for rep, d in frames.items():
        for x, v, m in zip(d.x, d.numerics, d.modality):
            assert len(x.split()) == len(v) == len(m)
        assert max(len(x.split()) for x in d.x) <= meta["num_features"][rep]
    assert frames["sequence"].events.tolist() == frames["bag"].events.tolist() == frames["static"].events.tolist()


def test_truncation_keeps_static_and_most_recent_history(tmp_path):
    _cohort(tmp_path)
    d = _parse(tmp_path, "sequence", max_tokens=12)
    ev = pd.read_csv(tmp_path / "src" / "events.csv")
    for i in range(5):
        toks = d.x[i].split()
        assert toks[0].startswith("c_sex_") and toks[1] == "x_age"
        hist = ev[ev.id == d.id[i]].code.astype(str).tolist()
        assert toks[2:] == hist[-(12 - 2):]


def test_visit_token_carries_time_and_labs_carry_values(tmp_path):
    _cohort(tmp_path)
    d = _parse(tmp_path, "sequence", max_tokens=400)
    toks, vals = d.x[0].split(), d.numerics[0]
    t_visit = [v for t, v in zip(toks, vals) if t == "VISIT"]
    assert t_visit and all(-5.01 <= v <= 0.0 for v in t_visit)  # years before index
    assert t_visit == sorted(t_visit)  # chronological


def test_true_cif_is_valid(tmp_path):
    _cohort(tmp_path)
    truth = dict(np.load(tmp_path / "src" / "truth.npz"))
    meta = json.loads((tmp_path / "src" / "meta.json").read_text())
    cif = true_cif(truth, meta["scenario"]["regime"], np.array([0, 200, 800, 2000.0]))
    assert (np.diff(cif, axis=2) >= -1e-9).all() and (cif >= 0).all() and (cif <= 1).all()
