# New experiment notebooks (Kaggle)

These replace every earlier notebook. **All earlier numbers are superseded** because of the
bugs below; nothing from `dmmst_full_results_old.pdf` / `_new.pdf` should go in the paper.

| notebook | question | runs (full) |
|---|---|---|
| `01_multievent_lmul` | Does the loss framework help on multi-event data? `L_rank`, `L_mul`, MMV vs Cox/DeepHit/DSM/MENSA on the simulator (with oracle), real EBMT and hsa_synthetic | ~140 (≈ 95 trained) |
| `02_synthetic_scenarios` | *When* does `L_mul` help? One simulator knob at a time: event-order spread, shared risk, frailty, censoring, regime | ~240 (≈ 145 trained) |
| `03_single_event_benchmarks` | METABRIC + SUPPORT (paper §4.1): backbone tuning, then baselines + loss/MMV ablation, SurvTRACE layout | ~150 (≈ 140 trained) |
| `05_llm_end_to_end` | §3: the end-to-end LLM (answer = one token per time unit, Eq. 8), pretrained DistilGPT2 and a GPT-2 from scratch, vs Cox / our transformer / oracle on sim_base, EBMT, METABRIC | ~45 (≈ 36 trained) |
| `06_sequential_ehr` | §4.2: coded, time-stamped histories. Same model on the full sequence vs a bag-of-codes table vs static features, in 3 synthetic EHR cohorts where trends matter 0 % / 50 % / 90 % | ~96 (≈ 81 trained) |
| `07_rerun_failed` | The 9 runs of 01/05 that crashed (both bugs fixed) | 9 |
| `08_true_order_and_tuning` | Does `L_mul` really order events better? Proper scores against the simulator's true event times; then `L_rank`/`L_mul` weight and σ tuned on validation and re-tested on 5 splits | ~150 |
| `09_sequential_ehr_v2` | §4.2 again with EHR simulator v2, where order, trends and recency are not recoverable from counts / last values | ~96 |
| `10_continuous_vs_discretised` | §2.2 / Fig. 2: numeric-value embedding vs 10 quantile bins, METABRIC + SUPPORT | 40 |
| `11_extra_sweeps` | Simulator sweeps over non-linearity, number of events (2–8), sample size (500–10k) | ~250 |
| **`12_final_support_and_encoding`** | **Final.** Corrected SUPPORT benchmark (the numeric parser had METABRIC's column layout hard-coded: five SUPPORT measurements unscaled, mean blood pressure lost) + continuous vs quantile vs binned input on METABRIC and SUPPORT | ~135 |
| **`13_final_sequential_ehr`** | **Final.** §4.2 with an improved sequence model (hidden 64, 4 layers, `[CLS]` pooling) and recency on every code (`sequence_time`), vs bag-of-codes, Cox, oracle | ~105 |
| `04_report` | All paper tables/figures from the outputs of the other notebooks (run last) | — |

Final runs: **12 and 13** (all other notebooks are already complete; SUPPORT results from 03/10 are superseded by 12). Original order: 01 → 03 → 02 → 06 → 05 → 04. 01 decides the paper's central claim.
Every notebook defaults to the real run (`SMOKE_TEST = False`).

Rough time per notebook on a Kaggle T4 (2 runs in parallel; estimated, ±2×): 01 5–9 h,
02 7–11 h, 03 4–5 h, 05 3–6 h (LLM runs are the slow ones), 06 6–10 h (sequences of up
to 256 tokens). Each run's seconds are in the log and `<tag>.meta.json`.

## What was fixed (and why every number changed)

1. **Train and validation were swapped** (`sat/data/splitter.py`). The single-split fold
   logic put ~10 % of the data in *train* and ~60 % in *validation*: every model in every
   earlier run (ours, baselines, Cox) trained on ~130 METABRIC patients instead of ~1 150.
   Regression test added.
2. **Cox PH ignored all categorical features** (`sat/coxph.py`): it used the `numerics`
   column, where categorical features are a placeholder 1.0 (4 of 9 features on METABRIC,
   6 of 14 on SUPPORT). Now one-hot encoded; METABRIC Cox = 0.627/0.622/0.620 vs published
   0.628/0.627/0.632.
3. **The "+L_mul" run never contained L_mul** (`conf/experiments/hsa_synthetic/survival-within-cindex.yaml`
   trained plain `nllpch`, only the metric changed). `L_mul` had never been tested.
4. **Model selection used the broken C-index** (`eval_ipcw_weighted_avg`, no τ, IPCW fitted
   on the evaluated split). All survival experiments now select on `eval_ctd_weighted_avg`.
5. **Error bars only varied initialisation.** New `split_seed`: the notebooks use a
   different train/val/test split per seed (as SurvTRACE does).
6. **Multi-event evaluation**: horizons were quantiles of all events pooled, so late events
   (EBMT death/relapse) were scored before they happen; now each event is scored at its
   own 25/50/75 % quantiles (`per_event_horizons`), and the hazard head gets 10 intervals
   instead of 4 on multi-event data.
7. **New within-subject metric** `within_ctd` = exactly the pairs of Eq. 5 (observed event
   vs. later-or-censored event, compared at the earlier event's time), plus
   `within_ctd_km` (population ordering) and `within_ctd_gain`. The old
   `within_subject_ipcw` dropped every observed-vs-censored pair.
8. **"MENSA" in the 4-event/EBMT notebooks was the MENSA-*inspired* head**, not the
   published model. All multi-event runs now use `mensa_paper`.
9. **Inference** (`sat/infer.py`, `sat/utils/output.py`): the pipeline was never registered,
   never received the patients' numeric values, crashed writing outputs, and wrote every
   event's interpolation to the same file. Verified: predictions for existing patients now
   equal the test predictions exactly.
10. Anomaly detection (a debugging aid costing ~40 % speed) is off by default.

New code: `sat/data/simulate.py` (simulator + true CIFs, Monte-Carlo-tested),
`sat/oracle.py` (true-CIF reference row), `sat/data/dataset/parse_multievent.py` (generic
K-event parser with categorical tokens), `sat/evaluate/multievent_metrics.py`,
`conf/experiments/multievent/*` (one config group for every simulated scenario and EBMT),
`conf/tasks/losses/nllpch_event_ranking.yaml` (L_PCH + L_mul only), Yasi's
`nllpch_mmv_rank.yaml`, `scripts/prepare_ebmt.py`, `scripts/runner.py`, `scripts/report.py`.

## How to run on Kaggle

1. **Upload the code once** as a Kaggle dataset: *Datasets → New dataset →* upload
   `dmmst_code.zip` (in `D:\Tavakoli\survival_analysis\`), name it **`dmmst-code`**.
   Re-upload it (*New version*) whenever the code changes. The notebooks print a
   *code fingerprint* so you can see which version ran. (Alternative: push the repo to
   GitHub and set `GIT_URL`.)
2. **Import a notebook** (*File → Import notebook*), then *Add Input →* `dmmst-code`,
   *Settings → Internet: on* (pip installs, EBMT download). GPU is optional — the model is
   tiny, CPU sessions work and have no weekly quota.
3. **Optional smoke test**: set `SMOKE_TEST = True`, *Run all* (2 epochs, 1 seed, reduced
   plan, ~10–30 min) to check the Kaggle environment; results go to `results/<name>_smoke/`.
   Set it back to `False` afterwards.
4. **Real run** (`SMOKE_TEST = False`, the default): *Save Version → Save & Run All (Commit)*.
   It runs in the background for up to 12 h and stops launching new runs after
   `TIME_BUDGET_MIN`. Notebook 05 needs a GPU session.
5. **Resume** if it did not finish: *Add Input →* this notebook's own output, commit again.
   Finished runs are copied back and skipped. Repeat until the log says `0 to go`.
6. **Report**: import `04_report`, add the outputs of 01–03 as inputs, run all.

Each notebook ends with `results/<name>.zip` (all `metrics.json`, run metadata, logs of every
run — `FAILED_*` if a run died — and the report). The per-run seconds are in the log; use the
first full session to estimate how many sessions a notebook needs.

## What goes where in the paper

* §3 end-to-end LLM: `05` → `table_sim_base`, `table_ebmt`, `table_metabric` (LLM rows next to Cox / ours / oracle).
* §4.2 sequential data: `06` → `table_ehrsim_*` and `representation_paired` (sequence vs bag vs static).
* §4.1 tabular: `03` → `report/table_metabric.*`, `table_support.*`, `horizons_vs_published.csv`.
* §4.3 multiple events: `01` → `table_sim_base`, `table_ebmt`, `table_hsa_synthetic`,
  `ablation_paired` (each loss term vs L_PCH, paired t-test + Wilcoxon over seeds).
* Synthetic proof-of-concept: `02` → `sweep_<knob>.pdf` + `.csv`.
* Keep claims to what the paired table supports; the oracle row is the ceiling.
* `within_ctd` is empty for the **competing** regime by construction: only the first event
  is observed and the others are censored at that same time, so no pair is orderable.
* **`within_ctd` can be gamed**: in the semi-competing settings models trained with `L_mul`
  scored above the oracle. For simulated data use the proper `true_order_*` scores
  (`scripts/true_order.py`: predicted P(event a before event b) vs the latent event times,
  added automatically to every run on a dataset with `truth.npz`); a model cannot beat
  the oracle on `true_order_brier` / `true_order_logloss` in expectation.
* **MAE-margin is unreliable on EBMT** (values > 7 000 days): with long follow-up and heavy
  censoring the Kaplan–Meier best guess for censored patients extrapolates far past the
  data. Report MAE only on METABRIC / SUPPORT / simulated data.
* Read `within_ctd` against `within_ctd_km`: on EBMT the population order alone scores
  ≈ 0.84 (Cox 0.847), so real EBMT leaves little room for `L_mul`; the simulator sweeps
  (small `order_spread`) are where covariate-driven ordering can be shown.

All four notebooks were executed end-to-end locally in smoke mode (2 epochs, 1 seed) with
no failed run before being handed over; smoke numbers are meaningless and were deleted.

## §3 and §4.2 (added)

* **§3 end-to-end LLM** — `sat/llm.py`, run with `python -m sat.llm` on any experiment group.
  Answer tokens `[N]`, `[C]`, `[Ek]` and combinations `[Ei+Ej]` (from the training data) over
  the duration cuts; loss = token cross-entropy on the answer (Eq. 8). Per-event curves come
  from the next-token probabilities with an all-`[N]` answer prefix, censoring mass removed.
  Two deliberate deviations from the sketch in the paper: censoring is treated as independent
  (its probability is removed, not modelled as an outcome), and the optional DeepHit-style
  loss / illegal-sequence penalties are not implemented. The loss is computed only at answer
  positions (identical value, ~50× cheaper than a full LM head over the prompt).
* **§4.2 coded sequences** — `sat/data/simulate_ehr.py` (synthetic EHR with known risks),
  `sat/data/dataset/parse_sequence.py` (sequence / bag / static representations),
  `conf/experiments/ehrseq/`. Visit times and lab values enter through the paper's
  numeric-value embedding. **The paper's CKD / COPD / diabetes cohorts** (or MIMIC) plug in
  directly: export `data/<name>/patients.csv` (id, `x_*`/`c_*` static columns,
  event1..K, duration1..K) and `events.csv` (id, time in days before index, code, value),
  run `python -m sat.data.dataset.parse_sequence data/<name> --num-events K`, add `<name>`
  to `REAL_SOURCES` in notebook 06. For thousands of ICD codes raise `transformer_vocab_size`.

## Not covered (needs data access)

* **MIMIC-IV / MIMIC-III**: credentialed PhysioNet access; tabular extracts plug into the
  `multievent` group (like `scripts/prepare_ebmt.py`), coded histories into `ehrseq`.
* **SEER**: parser exists; data needs a signed agreement.
* **The §4.2 cohorts themselves** (CKD / COPD / diabetes, "results provided by Zahra"):
  not in the repo — notebook 06 runs on synthetic cohorts until they are exported.

## Manuscript fixes found along the way

* Eq. 4 and Eq. 5: `η(x, y) = exp((x − y)/σ)` must be `exp(−(x − y)/σ)` (DeepHit); the code is right.
* MENSA (ML4H 2025) is uncited and does multi-event + within-subject ordering: add it to
  Table 1 and related work, and distinguish `L_mul` from its trajectory likelihood.
* §3 "End to end LLM" is now implemented (notebook 05); describe the censoring treatment and
  the all-`[N]` inference as above.
