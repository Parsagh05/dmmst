"""Sequential EHR simulator: coded visit histories -> multi-event outcomes (paper §4.2).

The paper's §4.2 evaluates on longitudinal ICD-coded records (CKD / COPD / diabetes).
Those data are not public, so this generates records with the same structure and a
KNOWN data-generating process, to test the claim that the transformer can use the
sequence (order, trends, recency) rather than a static summary.

Per patient
-----------
* static: age at index (numeric ``x_age``) and sex (categorical ``c_sex``);
* latent chronic conditions with onset before the index date: hypertension (HTN),
  type-2 diabetes (DM), chronic kidney disease (CKD), COPD, and smoking;
* latent trajectories: eGFR declining at a patient-specific slope after CKD onset,
  HbA1c drifting after DM onset, COPD exacerbations as a patient-specific Poisson
  process;
* a visit history over ``lookback_years``: at each visit diagnosis codes of active
  conditions, medications, labs with measured values (noisy), exacerbation codes and
  uninformative noise codes. Written as a long table ``events.csv`` (id, time, code,
  value) with time in days relative to the index date (negative).

Outcomes after index (``event1..4``, semi-competing: death censors the others)
-------------------------------------------------------------------------------
1. CKD progression   - current eGFR (level) and its slope (trend)
2. DM complication   - current HbA1c (level), its drift and DM duration (trend)
3. COPD hospitalisation - COPD severity (level), exacerbations in the last year (recency)
4. death             - age, number of conditions (level), eGFR / COPD trends

Each risk score mixes a *level* part (what a static "last value / counts" summary can
see) and a *temporal* part (what only the ordered, time-stamped sequence reveals):

    g_k = effect_size * standardise( sqrt(1 - w) * level_k + sqrt(w) * temporal_k )

with ``w = temporal_signal``. Event times are Weibull proportional hazards exactly as
in ``sat.data.simulate``, so ``true_cif`` / the oracle work unchanged.

    python -m sat.data.simulate_ehr --out data/ehrsim_base
    python -m sat.data.simulate_ehr --set temporal_signal=0.9 --out data/ehrsim_temporal
"""

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sat.data.simulate import _censoring, _observe, _standardise, blocking_sets

EVENT_NAMES = ["ckd_progression", "dm_complication", "copd_hospitalisation", "death"]
NOISE_CODES = [
    "R05", "M54.5", "J06.9", "K21.9", "R51", "Z00.0", "H52.4", "L30.9", "M25.5", "R10.4",
    "F41.9", "G47.0", "N39.0", "R42", "Z23", "B34.9", "M79.1", "R53.8", "K59.0", "J30.9",
]


@dataclass
class EHRScenario:
    name: str = "ehr_base"
    n: int = 5000
    lookback_years: float = 5.0
    temporal_signal: float = 0.5
    effect_size: float = 1.0
    censor_rate: float = 0.4
    admin_quantile: float = 0.9
    shapes: tuple = (0.9, 1.3)
    base_scale: float = 1500.0
    noise_codes_per_visit: float = 1.0
    # "v1": levels and trends are correlated, so a bag of counts + last values recovers
    # most of the temporal signal (sequence == bag in the first run). "v2": the current
    # lab level is drawn independently of its trend, exacerbations are either a recent
    # flare or long ago with the same count, and CKD-before-diabetes carries extra risk -
    # information only the ordered, time-stamped history contains.
    design: str = "v1"
    seed: int = 0


SCENARIOS = {
    "base": EHRScenario(),
    "static": EHRScenario(name="ehr_static", temporal_signal=0.0),
    "temporal": EHRScenario(name="ehr_temporal", temporal_signal=0.9),
    "v2_level": EHRScenario(name="ehr_v2_level", temporal_signal=0.0, design="v2"),
    "v2_mixed": EHRScenario(name="ehr_v2_mixed", temporal_signal=0.5, design="v2"),
    "v2_order": EHRScenario(name="ehr_v2_order", temporal_signal=0.9, design="v2"),
}


def _patients(sc: EHRScenario, rng):
    n = sc.n
    age = np.clip(rng.normal(62, 12, n), 30, 90)
    sex = rng.choice(["F", "M"], n)
    smoker = rng.random(n) < 0.3
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    htn = rng.random(n) < sig(-1.2 + 0.05 * (age - 60))
    dm = rng.random(n) < sig(-1.3 + 0.03 * (age - 60) + 0.6 * htn)
    ckd = rng.random(n) < sig(-2.0 + 0.04 * (age - 60) + 0.9 * dm + 0.7 * htn)
    copd = rng.random(n) < sig(-2.4 + 0.03 * (age - 60) + 1.8 * smoker)
    W = sc.lookback_years
    hi = W - 0.3 if sc.design == "v2" else W + 3.0  # v2: onsets inside the window (order visible)
    onset = {c: np.where(flag, -rng.uniform(0.3, hi, n), np.nan)  # v1: may predate the window
             for c, flag in [("htn", htn), ("dm", dm), ("ckd", ckd), ("copd", copd)]}
    p = pd.DataFrame(dict(age=age, sex=sex, smoker=smoker, htn=htn, dm=dm, ckd=ckd, copd=copd))
    for c, v in onset.items():
        p[f"onset_{c}"] = v
    # latent trajectories
    p["egfr0"] = np.clip(rng.normal(95 - 0.6 * (age - 50), 8), 30, 120)  # before CKD onset
    p["egfr_slope"] = np.where(ckd, rng.uniform(1.0, 12.0, n), rng.uniform(0.0, 1.0, n))  # ml/min per year
    p["a1c0"] = np.where(dm, rng.normal(7.4, 0.9, n), rng.normal(5.4, 0.3, n))
    p["a1c_drift"] = np.where(dm, rng.normal(0.0, 0.45, n), 0.0)  # % per year
    p["copd_sev"] = np.where(copd, rng.gamma(2.0, 0.5, n), 0.0)
    p["exac_rate"] = np.where(copd, p["copd_sev"] * rng.gamma(2.0, 0.6, n), 0.0)  # per year
    p["v2"] = sc.design == "v2"
    if sc.design == "v2":  # current levels independent of the trends
        p["egfr_now"] = np.where(ckd, np.clip(rng.normal(55, 15, n), 10, 110), np.clip(rng.normal(90, 10, n), 40, 130))
        p["a1c_now"] = np.where(dm, rng.normal(7.6, 1.0, n), rng.normal(5.4, 0.3, n))
        p["a1c0"] = p["a1c_now"]
        p["flare"] = copd & (rng.random(n) < 0.5)
    return p


def _egfr_at(r, t, W):
    """True eGFR at time t (years, 0 = index). CKD: egfr0 at onset, then declines at
    egfr_slope per year; otherwise a slow age-related decline across the window.
    v2: anchored at the independent current value egfr_now and back-computed."""
    if getattr(r, "v2", False):
        t_eff = max(t, r.onset_ckd) if r.ckd else t
        return float(np.clip(r.egfr_now - r.egfr_slope * t_eff, 5, 130))
    if r.ckd:
        return float(np.clip(r.egfr0 - r.egfr_slope * max(t - r.onset_ckd, 0.0), 5, 130))
    return float(np.clip(r.egfr0 - r.egfr_slope * (t + W), 5, 130))


def _a1c_at(r, t):
    if getattr(r, "v2", False):
        return float(np.clip(r.a1c_now + r.a1c_drift * max(t, r.onset_dm), 4.5, 14)) if r.dm else float(r.a1c_now)
    if r.dm:
        return float(np.clip(r.a1c0 + r.a1c_drift * max(t - r.onset_dm, 0.0), 4.5, 14))
    return float(r.a1c0)


def _histories(p: pd.DataFrame, sc: EHRScenario, rng):
    """Visits and codes; also returns the exacerbations in the last year (recency)."""
    rows, recent_exac = [], np.zeros(len(p))
    W = sc.lookback_years
    for i, r in p.iterrows():
        n_cond = int(r.htn + r.dm + r.ckd + r.copd)
        rate = 2.0 + 1.5 * n_cond  # visits per year
        times = np.sort(rng.uniform(-W, 0.0, rng.poisson(rate * W) + 1))
        # COPD exacerbations: Poisson process over the window
        n_exac = rng.poisson(r.exac_rate * W) if r.copd else 0
        if sc.design == "v2":  # same count; a recent flare or long ago
            lo_hi = (-0.5, 0.0) if r.flare else (-W, -0.5)
            exac = np.sort(rng.uniform(*lo_hi, n_exac))
            recent_exac[i] = np.sum(exac > -0.5)
        else:
            exac = np.sort(rng.uniform(-W, 0.0, n_exac))
            recent_exac[i] = np.sum(exac > -1.0)
        last_t = -W
        for t in times:
            active = lambda c: bool(r[c]) and r[f"onset_{c}"] <= t  # noqa: E731
            day = round(t * 365.25)

            def add(code, value=np.nan):
                rows.append((i, day, code, value))

            add("VISIT")
            for c, code in [("htn", "I10"), ("dm", "E11"), ("ckd", "N18"), ("copd", "J44")]:
                if active(c) and rng.random() < 0.75:
                    add(code)
            if r.smoker and rng.random() < 0.3:
                add("Z72.0")
            n_ex = int(np.sum((exac > last_t) & (exac <= t)))
            for _ in range(n_ex):
                add("J44.1")
            if active("dm"):
                a1c = _a1c_at(r, t)
                if rng.random() < 0.8:
                    add("metformin")
                if a1c > 8.5 and rng.random() < 0.7:
                    add("insulin")
                if rng.random() < 0.7:
                    add("LAB_HbA1c", a1c + rng.normal(0, 0.3))
            elif rng.random() < 0.15:
                add("LAB_HbA1c", r.a1c0 + rng.normal(0, 0.2))
            if (active("htn") or active("ckd")) and rng.random() < 0.7:
                add("ACEi")
            if rng.random() < (0.9 if active("ckd") else 0.35):
                add("LAB_eGFR", _egfr_at(r, t, W) + rng.normal(0, 4))
            if active("copd"):
                if rng.random() < 0.8:
                    add("inhaler")
                if rng.random() < 0.4:
                    add("LAB_FEV1", np.clip(80 - 15 * r.copd_sev + rng.normal(0, 6), 15, 110))
            for code in rng.choice(NOISE_CODES, rng.poisson(sc.noise_codes_per_visit)):
                add(str(code))
            last_t = t
    ev = pd.DataFrame(rows, columns=["id", "time", "code", "value"])
    return ev, recent_exac


def _risk_scores(p, recent_exac, sc: EHRScenario):
    egfr_now = np.array([_egfr_at(r, 0.0, sc.lookback_years) for r in p.itertuples()])
    a1c_now = np.array([_a1c_at(r, 0.0) for r in p.itertuples()])
    dm_dur = np.where(p.dm, -p.onset_dm.fillna(0), 0.0)
    copd_dur = np.where(p.copd, -p.onset_copd.fillna(0), 0.0)
    n_cond = (p.htn.astype(int) + p.dm + p.ckd + p.copd).values
    z = _standardise
    level = [
        z(-egfr_now) + 0.5 * z(p.dm.astype(float).values),
        z(a1c_now) * p.dm.values + 0.3 * z(p.htn.astype(float).values),
        z(p.copd_sev.values) + 0.5 * z(p.smoker.astype(float).values),
        z(p.age.values) + 0.6 * z(n_cond),
    ]
    temporal = [
        z(p.egfr_slope.values) * (p.ckd.values + 0.2),
        z(p.a1c_drift.values) * p.dm.values + 0.6 * z(dm_dur),
        z(recent_exac) + 0.4 * z(copd_dur),
        0.6 * z(p.egfr_slope.values) + 0.6 * z(recent_exac),
    ]
    if sc.design == "v2":
        ckd_first = (p.ckd & p.dm & (p.onset_ckd < p.onset_dm)).astype(float).values
        temporal = [
            z(p.egfr_slope.values) * (p.ckd.values + 0.2) + 0.7 * z(ckd_first),
            z(p.a1c_drift.values) * p.dm.values,
            z(recent_exac),
            0.6 * z(p.egfr_slope.values) + 0.6 * z(recent_exac),
        ]
    w = sc.temporal_signal
    g = np.stack([np.sqrt(1 - w) * z(l) + np.sqrt(w) * z(t) for l, t in zip(level, temporal)], 1)
    g = (g - g.mean(0)) / (g.std(0) + 1e-12)
    return sc.effect_size * g


def simulate(sc: EHRScenario):
    rng = np.random.default_rng(sc.seed)
    p = _patients(sc, rng)
    ev, recent_exac = _histories(p, sc, rng)
    g = _risk_scores(p, recent_exac, sc)
    K = g.shape[1]
    shapes = np.linspace(sc.shapes[0], sc.shapes[1], K)
    scales = sc.base_scale * np.ones(K)
    E = rng.exponential(size=(sc.n, K))
    T = np.maximum(scales * (E / np.exp(g)) ** (1.0 / shapes), 1.0)
    regime = "semi-competing"
    shim = dataclasses.make_dataclass("S", [("regime", str), ("censor_rate", float), ("admin_quantile", float)])
    C, admin, rate, achieved = _censoring(T, shim(regime, sc.censor_rate, sc.admin_quantile), rng)
    events, durations = _observe(T, C, regime)

    patients = pd.DataFrame({"id": np.arange(sc.n), "x_age": p.age.round(1), "c_sex": p.sex})
    for k in range(K):
        patients[f"event{k + 1}"] = events[:, k]
        patients[f"duration{k + 1}"] = durations[:, k]
    truth = {"g": g, "shapes": shapes, "scales": scales, "theta": np.array(0.0),
             "latent_times": T, "censor_times": C, "ids": np.arange(sc.n)}
    meta = {
        "scenario": dataclasses.asdict(sc) | {"regime": regime},
        "kind": "sequence",
        "num_events": K,
        "event_names": EVENT_NAMES,
        "max_time": float(durations.max()),
        "unobserved_fraction": achieved,
        "event_rates": events.mean(0).round(4).tolist(),
        "frac_subjects_multiple_events": float((events.sum(1) > 1).mean()),
        "prevalence": {c: float(p[c].mean()) for c in ["htn", "dm", "ckd", "copd", "smoker"]},
        "codes_per_patient": float(len(ev) / sc.n),
        "blocking": blocking_sets(regime, K),
    }
    return patients, ev, truth, meta


def save(sc: EHRScenario, out_dir, max_tokens: int = 256) -> dict:
    from sat.data.dataset.parse_sequence import describe

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    patients, ev, truth, meta = simulate(sc)
    patients.to_csv(out / "patients.csv", index=False)
    ev.to_csv(out / "events.csv", index=False)
    np.savez_compressed(out / "truth.npz", **truth)
    meta.update(describe(out, max_tokens=max_tokens))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="base", choices=list(SCENARIOS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--set", nargs="*", default=[], metavar="KNOB=VALUE")
    args = ap.parse_args(argv)
    sc = SCENARIOS[args.scenario]
    upd = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        upd[k] = type(getattr(sc, k))(v)
    sc = dataclasses.replace(sc, **upd)
    meta = save(sc, args.out or f"data/ehrsim_{args.scenario}", args.max_tokens)
    print(json.dumps({k: v for k, v in meta.items() if k != "scenario"}, indent=2))


if __name__ == "__main__":
    main()
