import json

import numpy as np

from scripts.report import build


def _write(d, tag, info, metrics):
    (d / f"{tag}.json").write_text(json.dumps({"test": {k: {"mean": v} for k, v in metrics.items()}}))
    (d / f"{tag}.meta.json").write_text(json.dumps({"tag": tag, "dataset": info["dataset"], "info": info}))


def test_tables_and_paired_ablation(tmp_path):
    rng = np.random.default_rng(0)
    for s in range(4):
        for rec, shift in [("nllpch", 0.0), ("nllpch_event_ranking", 0.02)]:
            _write(tmp_path, f"d__{rec}__s{s}", {"dataset": "d", "seed": s, "model": "ours", "recipe": rec},
                   {"ctd_weighted_avg": 0.7 + shift + rng.normal(0, 0.005), "within_ctd": 0.6 + shift})
        _write(tmp_path, f"d__coxph__s{s}", {"dataset": "d", "seed": s, "model": "coxph"},
               {"ctd_weighted_avg": 0.65})
    out = tmp_path / "rep"
    df = build([tmp_path], out)
    assert len(df) == 12
    assert (out / "table_d.csv").is_file()
    abl = (out / "ablation_paired.csv").read_text()
    assert "L_PCH + L_mul" in abl and "4/4" in abl


def test_sweep_figure_includes_base_in_every_sweep(tmp_path):
    for s in range(2):
        base = {"dataset": "sim_base", "seed": s, "sweep": "base", "value": None,
                "base_knobs": {"shared": 0.5, "regime": "non-competing"}}
        _write(tmp_path, f"b__oracle__s{s}", base | {"model": "oracle"}, {"within_ctd": 0.7})
        _write(tmp_path, f"b__l__s{s}", base | {"model": "ours", "recipe": "nllpch"}, {"within_ctd": 0.6})
        for v in (0.0, 1.0):
            _write(tmp_path, f"s{v}__l__s{s}", {"dataset": f"sim_shared{v}", "seed": s, "sweep": "shared",
                   "value": v, "model": "ours", "recipe": "nllpch"}, {"within_ctd": 0.55 + 0.1 * v})
        _write(tmp_path, f"rc__l__s{s}", {"dataset": "sim_regime_competing", "seed": s, "sweep": "regime",
               "value": "competing", "model": "ours", "recipe": "nllpch"}, {"within_ctd": 0.5})
    out = tmp_path / "rep"
    build([tmp_path], out)
    assert (out / "sweep_shared.png").is_file() and (out / "sweep_regime.png").is_file()
    tab = (out / "sweep_shared.csv").read_text()
    assert "0.5" in tab  # the base scenario appears at shared = 0.5


def test_grid_tuning_selects_on_validation(tmp_path):
    import json

    from scripts.runner import Runner, as_overrides, grid, select_best, tuning_runs

    assert len(grid({"a": [1, 2], "b": ["x", "y", "z"]})) == 6
    r = Runner(repo=".", results_dir=str(tmp_path), verbose=False)
    runs = tuning_runs("t", "finetune", "exp", [], {"learning_rate": [1e-3, 1e-4]}, [0, 1], "ds")
    for run in runs:  # config c001 is better on validation, worse on test
        good = run.info["config_id"] == "c001"
        (tmp_path / f"{run.tag}.json").write_text(json.dumps({
            "validation": {"ctd_weighted_avg": {"mean": 0.7 if good else 0.6}},
            "test": {"ctd_weighted_avg": {"mean": 0.5 if good else 0.9}}}))
        (tmp_path / f"{run.tag}.meta.json").write_text(json.dumps({"tag": run.tag, "info": run.info}))
    best = select_best(r, "t")
    assert best["config"] == {"learning_rate": 1e-4} and best["n"] == 2
    assert as_overrides(best["config"]) == ["learning_rate=0.0001"]
