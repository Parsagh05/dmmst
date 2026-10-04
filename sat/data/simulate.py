"""Multi-event survival simulator with a known data-generating process.

Purpose: the evidence for the paper's multi-event claims (Sec. 4.3). Every knob maps
onto one claim or confound, and the true cumulative incidence functions (CIFs) are
saved so every model can be scored against an oracle.

Data-generating process
-----------------------
Covariates ``x ~ N(0, I_p)``. Event k has a Weibull proportional hazard

    H_k(t | x, z) = z * (t / lambda_k) ** p_k * exp(g_k(x))

* ``g_k`` - risk score with standard deviation ``effect_size``. A share
  ``shared_effect`` (rho) of its variance comes from a component common to all events
  ("shared disease pathway"), the rest is event specific. ``nonlinearity`` is the share
  of variance carried by non-linear terms (products, sines, squares) that a linear Cox
  model cannot represent.
* ``z`` - unobserved gamma frailty with mean 1 and variance ``frailty_var``, shared by
  all events of a subject: dependence that covariates cannot explain.
* ``lambda_k = base_scale * exp(linspace(-scale_spread, scale_spread))`` and
  ``p_k = linspace(*shapes)``. With ``scale_spread`` large the within-subject event
  order is mostly the same for everybody (a population-order predictor already scores
  well); with ``scale_spread = 0`` the order is driven by covariates, which is where
  the within-subject ranking term L_mul should matter.

Observation regimes (``regime``):

* ``non-competing`` - every event is observed if it happens before censoring.
* ``competing`` - only the first event is observed; all others are censored at it.
* ``semi-competing`` - the last event (k = K-1) is terminal (death) and censors the
  non-terminal ones; non-terminal events do not censor anything.

Censoring ``C = min(admin_time, Exp(rate))``; ``rate`` is found by bisection so that
the fraction of unobserved event slots matches ``censor_rate`` (subject-level for
``competing``). ``admin_time`` defaults to the ``admin_quantile`` of latent times.

True CIF of the *observable* event k (independent censoring), with B_k the events
that block k in the regime and A = {k} U B_k:

    CIF_k(t | x) = int_0^t (1 + theta * sum_{j in A} H_j(u|x)) ** -(1/theta + 1) dH_k(u|x)

(theta -> 0 gives exp(-sum H_j)). It is integrated numerically on a grid dense near 0.

Usage
-----
    python -m sat.data.simulate --scenario base --out data/sim_base
    python -m sat.data.simulate --list

writes ``data.csv`` (id, x_1..x_p, event1..K, duration1..K), ``truth.npz`` and
``meta.json`` into ``--out``.
"""

__authors__ = ["Parsa"]
__status__ = "Development"

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REGIMES = ("non-competing", "competing", "semi-competing")


@dataclass
class Scenario:
    name: str = "base"
    n: int = 5000
    num_events: int = 4
    num_features: int = 15
    num_informative: int = 6
    regime: str = "non-competing"
    shared_effect: float = 0.5
    frailty_var: float = 0.0
    nonlinearity: float = 0.5
    effect_size: float = 1.0
    scale_spread: float = 0.0
    shapes: Tuple[float, float] = (0.8, 1.6)
    base_scale: float = 365.0
    censor_rate: float = 0.4
    admin_quantile: float = 0.9
    seed: int = 0

    def __post_init__(self):
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.regime!r}")
        for name in ("shared_effect", "nonlinearity"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {v}")
        self.shapes = tuple(self.shapes)


# Named scenarios. Each sweep varies ONE knob around `base`, so a difference between
# two rows of a table has a single cause.
def _scenarios() -> Dict[str, Scenario]:
    s = {"base": Scenario()}

    def var(prefix, knob, values):
        for v in values:
            tag = f"{prefix}{v}".replace(".", "p")
            s[tag] = dataclasses.replace(s["base"], name=tag, **{knob: v})

    var("order_spread", "scale_spread", [0.0, 0.5, 1.0, 2.0])
    var("shared", "shared_effect", [0.0, 0.25, 0.5, 0.75, 1.0])
    var("frailty", "frailty_var", [0.0, 0.5, 1.0, 2.0])
    var("nonlin", "nonlinearity", [0.0, 0.5, 1.0])
    var("censor", "censor_rate", [0.2, 0.4, 0.6, 0.8])
    var("events", "num_events", [2, 4, 6, 8])
    var("n", "n", [500, 1000, 2500, 5000, 10000])
    for regime in REGIMES:
        tag = f"regime_{regime.replace('-', '')}"
        s[tag] = dataclasses.replace(s["base"], name=tag, regime=regime)
    return s


SCENARIOS = _scenarios()


# --------------------------------------------------------------------------- risk
def _standardise(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def _nonlinear_basis(x: np.ndarray, informative: np.ndarray, rng) -> np.ndarray:
    """A fixed bank of non-linear functions of the informative covariates."""
    cols = []
    m = len(informative)
    for _ in range(2 * m):
        a, b = rng.choice(informative, size=2, replace=False)
        kind = rng.integers(4)
        if kind == 0:
            cols.append(x[:, a] * x[:, b])
        elif kind == 1:
            cols.append(np.sin(1.5 * x[:, a]) * np.cos(x[:, b]))
        elif kind == 2:
            cols.append(x[:, a] ** 2 - 1.0)
        else:
            cols.append(np.tanh(x[:, a] * x[:, b]) + np.abs(x[:, b]))
    basis = np.stack(cols, axis=1)
    return (basis - basis.mean(0)) / (basis.std(0) + 1e-12)


def _risk_scores(x: np.ndarray, sc: Scenario, rng) -> np.ndarray:
    """g_k(x) for every event, shape (n, K), each with sd = effect_size."""
    p, K = x.shape[1], sc.num_events
    informative = rng.choice(p, size=min(sc.num_informative, p), replace=False)
    basis = _nonlinear_basis(x, informative, rng)

    def component(design):
        shared = design @ rng.normal(size=design.shape[1])
        out = []
        for _ in range(K):
            specific = design @ rng.normal(size=design.shape[1])
            out.append(
                np.sqrt(sc.shared_effect) * _standardise(shared)
                + np.sqrt(1.0 - sc.shared_effect) * _standardise(specific)
            )
        return np.stack(out, axis=1)

    lin = component(x[:, informative])
    nonlin = component(basis)
    g = np.sqrt(1.0 - sc.nonlinearity) * lin + np.sqrt(sc.nonlinearity) * nonlin
    g = (g - g.mean(0)) / (g.std(0) + 1e-12)
    return sc.effect_size * g


def _baselines(sc: Scenario) -> Tuple[np.ndarray, np.ndarray]:
    K = sc.num_events
    shapes = np.linspace(sc.shapes[0], sc.shapes[1], K)
    scales = sc.base_scale * np.exp(np.linspace(-sc.scale_spread, sc.scale_spread, K))
    return shapes, scales


def blocking_sets(regime: str, K: int) -> List[List[int]]:
    """Events whose occurrence prevents event k from being observed."""
    if regime == "non-competing":
        return [[] for _ in range(K)]
    if regime == "competing":
        return [[j for j in range(K) if j != k] for k in range(K)]
    # semi-competing: last event is terminal
    return [[K - 1] for _ in range(K - 1)] + [[]]


# ------------------------------------------------------------------- observation
def _observe(T: np.ndarray, C: np.ndarray, regime: str):
    n, K = T.shape
    if regime == "non-competing":
        events = (T <= C[:, None]).astype(int)
        durations = np.minimum(T, C[:, None])
    elif regime == "competing":
        first = T.argmin(1)
        tmin = T.min(1)
        observed = tmin <= C
        durations = np.repeat(np.minimum(tmin, C)[:, None], K, axis=1)
        events = np.zeros((n, K), dtype=int)
        events[np.arange(n)[observed], first[observed]] = 1
    else:  # semi-competing
        death = T[:, K - 1]
        stop = np.minimum(death, C)
        events = np.zeros((n, K), dtype=int)
        durations = np.zeros((n, K))
        events[:, : K - 1] = (T[:, : K - 1] <= stop[:, None]).astype(int)
        durations[:, : K - 1] = np.minimum(T[:, : K - 1], stop[:, None])
        events[:, K - 1] = (death <= C).astype(int)
        durations[:, K - 1] = np.minimum(death, C)
    return events, durations


def _unobserved_fraction(events: np.ndarray, regime: str) -> float:
    if regime == "competing":
        return float(1.0 - events.max(1).mean())
    return float(1.0 - events.mean())


def _censoring(T, sc: Scenario, rng):
    admin = float(np.quantile(T, sc.admin_quantile))
    E = rng.exponential(size=T.shape[0])  # common random numbers across the bisection

    def frac(rate):
        C = np.minimum(admin, E / rate) if rate > 0 else np.full(T.shape[0], admin)
        return _unobserved_fraction(_observe(T, C, sc.regime)[0], sc.regime), C

    floor, C = frac(0.0)
    if sc.censor_rate <= floor:
        return C, admin, 0.0, floor
    lo, hi = 0.0, 1.0
    while frac(hi)[0] < sc.censor_rate:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if frac(mid)[0] < sc.censor_rate:
            lo = mid
        else:
            hi = mid
    achieved, C = frac(hi)
    return C, admin, hi, achieved


# ---------------------------------------------------------------------- simulate
def simulate(sc: Scenario):
    """Returns (dataframe, truth dict, meta dict)."""
    rng = np.random.default_rng(sc.seed)
    n, p, K = sc.n, sc.num_features, sc.num_events

    x = rng.normal(size=(n, p))
    g = _risk_scores(x, sc, rng)
    shapes, scales = _baselines(sc)
    z = (
        rng.gamma(shape=1.0 / sc.frailty_var, scale=sc.frailty_var, size=n)
        if sc.frailty_var > 0
        else np.ones(n)
    )
    E = rng.exponential(size=(n, K))
    T = scales[None, :] * (E / (z[:, None] * np.exp(g))) ** (1.0 / shapes[None, :])
    T = np.maximum(T, 1e-3)

    C, admin, rate, achieved = _censoring(T, sc, rng)
    events, durations = _observe(T, C, sc.regime)

    df = pd.DataFrame(x, columns=[f"x_{i + 1}" for i in range(p)])
    df.insert(0, "id", np.arange(n))
    for k in range(K):
        df[f"event{k + 1}"] = events[:, k]
        df[f"duration{k + 1}"] = durations[:, k]

    truth = {
        "g": g,
        "shapes": shapes,
        "scales": scales,
        "theta": np.array(sc.frailty_var),
        "latent_times": T,
        "censor_times": C,
        "ids": np.arange(n),
    }
    multi = (events.sum(1) > 1).mean()
    meta = {
        "scenario": dataclasses.asdict(sc),
        "num_events": K,
        "num_features": p,
        "max_time": float(durations.max()),
        "admin_time": admin,
        "censor_hazard": rate,
        "unobserved_fraction": achieved,
        "event_rates": events.mean(0).round(4).tolist(),
        "frac_subjects_multiple_events": float(multi),
        "blocking": blocking_sets(sc.regime, K),
    }
    return df, truth, meta


def save(sc: Scenario, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, truth, meta = simulate(sc)
    df.to_csv(out / "data.csv", index=False)
    np.savez_compressed(out / "truth.npz", **truth)
    with (out / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ------------------------------------------------------------------- true CIFs
def true_cif(truth, regime: str, times: np.ndarray, rows=None, grid_size: int = 2000,
             chunk: int = 512) -> np.ndarray:
    """True CIF_k(t | x) of the observable events, shape (n_rows, K, len(times))."""
    g = truth["g"] if rows is None else truth["g"][rows]
    shapes, scales = np.asarray(truth["shapes"]), np.asarray(truth["scales"])
    theta = float(truth["theta"])
    K = g.shape[1]
    times = np.asarray(times, dtype=float)
    t_max = max(times.max(), 1e-6)
    # dense near zero, where H_k is steep for shape < 1
    grid = t_max * (np.linspace(0.0, 1.0, grid_size) ** 2)
    base = (grid[None, :] / scales[:, None]) ** shapes[:, None]  # (K, G)
    blocks = blocking_sets(regime, K)

    out = np.empty((g.shape[0], K, len(times)))
    for s in range(0, g.shape[0], chunk):
        H = np.exp(g[s : s + chunk])[:, :, None] * base[None]  # (c, K, G)
        for k in range(K):
            S_A = H[:, [k] + blocks[k], :].sum(1)
            w = np.exp(-S_A) if theta == 0 else (1.0 + theta * S_A) ** (-(1.0 / theta + 1.0))
            dH = np.diff(H[:, k, :], axis=1)
            cif = np.concatenate(
                [np.zeros((H.shape[0], 1)), np.cumsum(0.5 * (w[:, 1:] + w[:, :-1]) * dH, 1)],
                axis=1,
            )
            out[s : s + chunk, k, :] = np.stack(
                [np.interp(times, grid, row) for row in cif]
            )
    return np.clip(out, 0.0, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--out", default=None, help="default: data/sim_<scenario>")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KNOB=VALUE",
                    help="override scenario fields, e.g. --set n=2000 regime=competing")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for name, sc in SCENARIOS.items():
            print(name, dataclasses.asdict(sc))
        return

    sc = SCENARIOS[args.scenario]
    updates = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(sc, k)
        updates[k] = type(cur)(v) if not isinstance(cur, tuple) else tuple(map(float, v.split(",")))
    if args.seed is not None:
        updates["seed"] = args.seed
    sc = dataclasses.replace(sc, **updates)
    out = args.out or f"data/sim_{sc.name}"
    meta = save(sc, out)
    print(json.dumps({k: v for k, v in meta.items() if k != "scenario"}, indent=2))


if __name__ == "__main__":
    main()
