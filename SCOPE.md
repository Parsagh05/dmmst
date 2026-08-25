# Scope

This tree is a **paper-faithful subset** of the SAT codebase, pruned to contain only the
components described in `our_papers/Dynamic_Multi_modal_Multi_event_Survival_Transformer.pdf`.

Everything inherited from earlier, abandoned work — in particular the pre-existing synthetic
datasets (`hsa-synthetic`, the DeepHit `synthetic_comprisk` copy) and the synthetic design in
the IEEE draft `AIhealth2026.pdf` — has been **removed and is not to be reused**.

Nothing here was rewritten. Files were kept, deleted, or had dead references cleaned up.

## Paper → implementation map

| Paper | Component | File |
|---|---|---|
| §2.2, Fig. 2 | Numeric value embedding (token vector scaled by standardised value) | [configuration_bert.py](sat/models/bert/configuration_bert.py), [modeling_bert.py](sat/models/bert/modeling_bert.py) |
| §2.2 | "any attentive encoder and/or decoder (e.g., GPT or BERT)" | [bert/](sat/models/bert/), [gpt2/](sat/models/gpt2/) |
| §2.2 | Layer selection; token embedding (sum/avg/cat/BERT-pool) → sentence embedding | [heads/embeddings.py](sat/models/heads/embeddings.py) |
| §2.2 | Shared MLP + task heads (`MLP_ST` / `MLP_RT` / `MLP_CT`), multi-task | [heads/mtl.py](sat/models/heads/mtl.py), [heads/base.py](sat/models/heads/base.py) |
| §2.2 | Survival head `MLP_ST` | [heads/survival.py](sat/models/heads/survival.py) |
| §2.4 | Regression head `MLP_RT` | [heads/regression.py](sat/models/heads/regression.py) |
| §2.2 | Classification head `MLP_CT` | [heads/classification.py](sat/models/heads/classification.py) |
| §2.3 Eq. 2 | Generic loss `Σ L + Σ α_ℓ P_ℓ` | [loss/meta.py](sat/loss/meta.py), [loss/base.py](sat/loss/base.py), [loss/balancing.py](sat/loss/balancing.py) |
| §2.3 | `L_PCH` — piece-wise constant hazard NLL | [loss/survival/nllpchazard.py](sat/loss/survival/nllpchazard.py) |
| §2.3 Eq. 4 | `L_rank` — ranking of samples within an event | [loss/ranking/sample.py](sat/loss/ranking/sample.py) |
| §2.3 Eq. 5 | `L_mul` — ranking of events within a sample | [loss/ranking/multievent.py](sat/loss/ranking/multievent.py) |
| §2.4 Eq. 6 | `L_MAE` — best-guess MAE, Kaplan-Meier extension (`l1_type: margin`) | [loss/regression/l1.py](sat/loss/regression/l1.py) |
| §2.4 Eq. 7 | `L_MM` — mismatch penalty | [loss/survival/mismatch.py](sat/loss/survival/mismatch.py) |

## Added: Margin-Mean-Variance loss

Not from our paper — from **UniSurv** (Machine Learning 2024, doi:10.1007/s10994-024-06686-w,
[arXiv:2409.06209](https://arxiv.org/abs/2409.06209)), which is what the DOI in `task.pdf`
resolves to.

| UniSurv | Component | File |
|---|---|---|
| Eq. 3, 4, 6 | `MMVLoss` = `L_mm + λ_v·L_v` | [loss/survival/mmv.py](sat/loss/survival/mmv.py) |
| Eq. 5 | margin time — *already existed* as `KaplanMeierArea.best_guess` | [utils/km.py](sat/utils/km.py) |
| — | recipes | [mmv.yaml](conf/tasks/losses/mmv.yaml), [nllpch_mmv.yaml](conf/tasks/losses/nllpch_mmv.yaml) |
| — | tests (12, all passing) | [test_mmv.py](tests/loss/survival/test_mmv.py) |

UniSurv's Eq. 5 is the *same* best-guess construction as our paper's Eq. 6, so `MMVLoss`
reuses `KaplanMeierArea.best_guess` rather than reimplementing it. Its full objective
(Eq. 10) decomposes across the existing framework: `L_s` → `SATNLLPCHazardLoss`,
`L_d` → `SampleRankingLoss`, and only `L_mm + λ_v·L_v` is new.

The paper's headline recipe, `Loss = L_PCH + L_rank + L_mul`, is
[nllpch_sample_event_ranking.yaml](conf/tasks/losses/nllpch_sample_event_ranking.yaml).

## Metrics

These are what the claims get measured against — note the last two, which target the
paper's actual novelty rather than generic survival performance.

| Metric | File | Evidences |
|---|---|---|
| IPCW concordance index | [concordance_index_ipcw/](sat/evaluate/concordance_index_ipcw/) | across-sample risk ranking |
| Brier score / IBS | [brier_score/](sat/evaluate/brier_score/) | calibration + discrimination |
| One-calibration | [one_calibration/](sat/evaluate/one_calibration/) | calibration |
| NLL piece-wise hazard | [nllphazard/](sat/evaluate/nllphazard/) | likelihood fit |
| MAE / MSE | [l1/](sat/evaluate/l1/), [mse/](sat/evaluate/mse/) | regression head (§2.4) |
| **Within-subject concordance** | [within_subject_concordance/](sat/evaluate/within_subject_concordance/) | **`L_mul` (Eq. 5)** |
| **Mismatch** | [mismatch/](sat/evaluate/mismatch/) | **`L_MM` (Eq. 7)** |

## Removed, and why

**Method components not in the paper.** Survival focal loss, SurvRNC, SOAP, ListMLE,
RankNet (all variants), MoCo momentum buffers, quantile regression, `IntraEventRankingLoss`.

**Restored as baselines** (removed in the first pass, brought back because `task.pdf`
requires them and §4.1 needs a comparison table): DeepHit, DSM, MENSA, plus
`sat/distributions/` and `parameter_nets.py`. These are **not** part of the paper's
method — they exist only to produce comparison numbers, and are marked as such in
`conf/experiments/defaults.yaml`.

Note `conf/experiments/metabric/mensa.yaml` **did not exist upstream** (only
`metabric_numeric/mensa.yaml`, which was malformed — no `@package` header, wrong
interpolation namespace). Both were written correctly for this tree.

**Data pipelines not in the paper.** FEMR / MOTOR-T, EHRSHOT, OMOP, MEDS.

**Inherited synthetic data.** `parse_synthetic.py` (DeepHit `synthetic_comprisk`),
`parse_synthetic_numerics.py`, `generate_synthetic_omop.py` and their configs.

**HSA-synthetic: restored.** Removed in the first pass, then restored because
`task.pdf` requires baselines on it. It is `num_events: 2`, so it is the only
multi-event dataset currently available and exercises the per-event metrics and the
within-subject C-index. **Its numbers are a pipeline check, not evidence** — the event
offsets are shared across individuals, so every subject has essentially the same event
ordering and a model emitting one constant ordering scores well without learning
anything. The purpose-built simulator is still needed for actual multi-event evidence.

**Retained infrastructure** (not described in the paper, but needed to *produce* the paper's
results): Hydra config system, the `prepare_data → train_tokenizer → train_labeltransform →
finetune` pipeline, `ci.py` (bootstrap confidence intervals), `cv.py` (k-fold), `eda.py`,
Optuna sweeps, and the SurvTrace baseline configs.

`lr_finder.py` was **removed**: it imports `sat.data.collator`, a module that does not exist
in the upstream repo either, so it is broken independently of this prune. Optuna sweeps cover
learning-rate search.

## Datasets

Bundled and verified working: **METABRIC** (1904, 9 feat, 1 event), **SUPPORT**
(8873, 14 feat, 1 event), **hsa-synthetic** (5000, 2 events). All three ship in the
repo, so the notebooks clone and run with nothing to download.

SUPPORT uses the same DeepSurv-style train/test HDF5 layout as METABRIC, so it reuses
`parse_metabric.metabric`; only the categorical/continuous column split differs and
that now comes from the config ([support.yaml](conf/data/parse/support.yaml)).

**Gaps to fill:**
- **SEER** (§4.1) — parser exists, but the data needs a signed data-request agreement
  and cannot be bundled.
- **Multi-event data** (§4.3) — a simulator must be built from scratch. This is the main
  evidence for the paper's central claim. `hsa-synthetic` runs and is useful for
  exercising the multi-event pipeline, but it is **not** a substitute as evidence: its
  event offsets are shared across individuals, so every subject has the same event
  ordering and the within-subject ranking task is degenerate.
- **Sequential ICD-code data** (§4.2, CKD/COPD/Diabetes) — not in this repo and access
  is undecided.

## Verification status

- All 10 pipeline entry points import cleanly
  (`prepare_data`, `train_tokenizer`, `train_labeltransform`, `finetune`, `eval`, `infer`,
  `ci`, `cv`, `eda`, `pipeline`).
- All 27 Hydra `_target_` references and every `defaults:` entry resolve.
- Test suite: **58 passed, 13 failed** — the 13 failures are *pre-existing*, reproducing
  identically against the original untouched repo (mock/`transformers`-version issues in
  `test_trainer.py` and `test_tokenizing.py`). The prune introduced no regressions.
- `prepare_data` on METABRIC was run once end-to-end without error; artifacts were cleaned up.
- Dependency pins that cost real debugging time are recorded in
  [requirements-kaggle.txt](requirements-kaggle.txt) with the reason for each.
- `MMVLoss`: **14** unit tests pass, and it reproduces a from-scratch NumPy
  implementation of Eqs. 3–6 to 6 decimal places.
- `MMVLoss` runs through the **real pipeline** on METABRIC via `tasks/losses=mmv` and
  `tasks/losses=nllpch_mmv`, compared against the `nllpch` baseline at matched seed.
- The restored baselines (DeepHit / DSM / MENSA) **import and their configs resolve, but
  have not been run end-to-end.** That is what the notebook is for.

## Kaggle notebooks

One per person, fully independent — they share nothing but the repo.

- [notebooks/yasi_environment_and_baselines.ipynb](notebooks/yasi_environment_and_baselines.ipynb)
  — pipeline run, the four baselines, config fluency, and an auto-generated
  `experimental_setup.md`. Ends with a checklist tracker showing which `task.pdf` boxes
  actually got ticked.
- [notebooks/parsa_mmv_loss.ipynb](notebooks/parsa_mmv_loss.ipynb) — MMV formula, toy
  NumPy verification, unit tests, and MMV vs. `nllpch` on METABRIC.

Both default to `SMOKE_TEST = True` (3 epochs, ~1 min) so a broken setup surfaces before
committing to the 500-epoch runs.

Kaggle's **"GPU T4 x2" exposes two GPUs**, which makes HF Trainer wrap the model in
`nn.DataParallel` (`if self.args.n_gpu > 1`). DataParallel scatters every forward input
across devices and dies here with `RuntimeError: chunk expects at least a 1-dimensional
tensor`, because some inputs are 0-dimensional. The notebooks therefore pin child runs
to one GPU via `CUDA_VISIBLE_DEVICES` (`USE_SINGLE_GPU = True`). That is also the right
call on merit — the model is ~3 MB, so this is overhead-bound and DataParallel's
per-step scatter/gather would make it slower even if it worked.

Each notebook ends by writing a single **zip** to `/kaggle/working` containing every
`metrics.json`, the CSV tables, `experimental_setup.md`, the **full per-run logs**
(prefixed `FAILED_` when a run died), an `environment.json` snapshot, and the
**fully-resolved Hydra config** of every run — so any number can be traced back to the
exact composed config that produced it.

Runs stream **live**: a `tqdm` bar per run, driven off the trainer's own `'epoch': N`
log lines, plus an outer bar across the run list. Errors print immediately; routine
metric lines are throttled to one per 10 s, because at `eval_steps: 1` a 500-epoch run
emits ~1500 of them. The full log is still retained for diagnosing failures. The training path needs neither `lifelines` nor a
`scipy` downgrade — `lifelines` is used only by `sat.eda`.

## Reproducing the published benchmark

The first full run scored ~0.10-0.15 below SurvTRACE's Table 2 on METABRIC, with every
model below plain Cox PH. That was mostly a **measurement** problem, not a modelling one.

**The C-index was computed wrongly.** `ComputeCIndex` looped over the duration cuts and
took the risk estimate at each, but never passed `tmax`/`tau` to the concordance
calculator, so every "horizon" scored concordance over the whole follow-up. It also fit
the IPCW censoring distribution on the *test* split instead of train. The symptom was a
C-index flat across horizons (0.571/0.577/0.583) where published values fall
(0.713/0.680/0.644).

[survtrace_metrics.py](sat/evaluate/survtrace_metrics.py) reproduces the reference
protocol exactly — `sksurv.metrics.concordance_index_ipcw` with `tau` truncation and a
train-fitted censoring distribution, evaluated at the 25/50/75% quantiles of uncensored
event times. Select it with `tasks/metrics=survtrace`.

**Feature encoding.** `conf/data/parse/metabric.yaml` discretises continuous covariates
with `KBinsDiscretizer(n_bins=10)`. SurvTRACE uses `StandardScaler`, and the paper's own
§2.2 argues explicitly *against* discretisation. The `*_numeric` parse path is the one to
use; switching alone is worth about +0.025 C-index.

**What was already correct:** the duration cuts. `train_labeltransform` computes
`linspace(0,1,cuts+1)[1:-1]` quantiles of uncensored event times, giving
`[0, q25, q50, q75, max]` — exactly SurvTRACE's `num_durations: 5`.

**Installing scikit-survival:** pick the release that matches the numpy already
present and never move numpy itself — `pandas`/`pyarrow` are compiled against it, and
downgrading gives `ValueError: numpy.dtype size changed ... Expected 96, got 88` on the
next `import pandas`. numpy>=2 → `scikit-survival`; numpy<2 → `scikit-survival==0.22.2`.
The notebooks detect this and pin numpy to its installed version so pip fails loudly
rather than breaking the image. Without sksurv everything still runs, only the `ctd_*`
metrics are absent.

`conf/experiments/survtrace_{metabric,support}/` bundles all of this with SurvTRACE's own
hyperparameters (hidden 16, intermediate 64, 3 layers, 2 heads, lr 1e-3, weight decay
1e-4, batch 64, early stopping).

### Cox PH reference

[sat/coxph.py](sat/coxph.py) fits `sksurv` Cox PH on the same split, features and metric,
so there is a sanity floor every deep model must clear:

    python -m sat.coxph experiments=survtrace_metabric/survival

**Caveat on error bars:** the splitter is hash-based on record ID, so changing `seed`
varies model initialisation but *not* the train/test split. SurvTRACE reports variation
over 10 different splits. Our spreads are therefore narrower than theirs and are not
directly comparable as uncertainty estimates.

## Upstream bugs found and fixed

All three were latent in the original repo and would have hit any real run:

| Bug | Effect | Fix |
|---|---|---|
| `log_gpu_utilization()` called `nvmlInit()` unguarded | A **logging** call aborted every GPU run when NVML was unavailable | Fail-safe with a torch-based fallback ([logging.py](sat/utils/logging.py)); requirements moved from the abandoned `nvidia-ml-py3` (2017) to `nvidia-ml-py` |
| DSM head padded `hazard` one column wider than `risk`/`survival` | Shape check threw, the exception was **swallowed**, and Brier/IPCW silently returned hard-coded `0.5` defaults — DSM looked like it worked while reporting fabricated numbers | Removed the spurious `pad_col` ([dsm.py](sat/models/heads/dsm.py)); DSM now reports genuine metrics |
| `write_output` hard-coded prediction width as `cuts + 1` | DSM/MENSA crashed at the prediction dump, *after* metrics were computed | Width derived from the tensor ([output.py](sat/utils/output.py)) |
| `parse_hsa_synthetic.py` assigned `astype("object")` back through `.loc[:, col]` | The column kept its string dtype, so the later `.at[index, col] = <list>` raised `Must have equal len keys and value when setting with an iterable`. HSA `prepare_data` could not run at all on modern pandas | Direct column assignment, matching the working `parse_metabric_numerics.py` ([parse_hsa_synthetic.py](sat/data/dataset/parse_hsa_synthetic.py)) |

Plus one in my own new code, found only by running it through the pipeline:

| `MMVLoss` prepended `0` to `duration_cuts` | The real `duration_cuts.csv` **already starts at `0.0`**, so the grid got one interval too many and the loss crashed with a 6-vs-5 size mismatch. Every unit test used a cut list omitting `t=0`, so none caught it | Grid derived from the survival tensor width, both layouts accepted, and two regression tests added ([mmv.py](sat/loss/survival/mmv.py)) |

### MMV scale mismatch

`L_PCH` is a log-likelihood (~0.8 on METABRIC); MMV is a squared error in **time units**
(~2400, since METABRIC durations run to ~355). Under `fixed` balancing the MMV term
outweighs the likelihood by three orders of magnitude and the likelihood is effectively
ignored. [nllpch_mmv.yaml](conf/tasks/losses/nllpch_mmv.yaml) therefore defaults to
`balancing: scale`. Measured effect at 3 epochs, seed 0:

| recipe | ipcw | brier | loss |
|---|---|---|---|
| `nllpch` (baseline) | 0.5188 | 0.2069 | 0.78 |
| `mmv` | 0.5292 | 0.2317 | 2434 |
| `nllpch_mmv` (fixed) | 0.5116 | 0.2344 | 2428 |
| `nllpch_mmv` (**scale**) | 0.5249 | 0.2318 | 2.18 |

Smoke-test numbers — they show the balancing behaves, not that MMV helps.

## Verified by running

- Full chain on METABRIC: `prepare_data` → `train_tokenizer` → `train_labeltransform`
  → `finetune` → `metrics.json`.
- **All four baselines** (`survival`, `deephit`, `dsm`, `mensa`) train and produce 15
  genuine test metrics each on METABRIC, with zero swallowed prediction errors.
- **Same four baselines on SUPPORT**: all pass, 15 genuine test metrics each.
- **Same four baselines on HSA-synthetic** (2 events): all pass, 23 metrics each,
  including per-event `ipcw_0th_event` / `ipcw_1th_event`. The
  `hsa_synthetic/survival-within-cindex` variant also runs and reports a
  within-subject C-index over 1516 subjects.
- Smoke runs only (3 epochs) — enough to prove the models work end-to-end, **not**
  enough for any number that goes in the paper.

`metrics.json` is **nested** — `{"validation"|"test": {metric: {mean, variance, sd}}}`.
A flat parse yields nothing; the notebooks handle this.
