# Version 2 — the complete test of the paper (current)

**Status (2026-10-05):** steps 0-1 done, notebooks written; Kaggle runs under way.
01 (tuning) and 07 (LLM, 1 seed) have run once; the fixes they needed are in
(NumPy 2 alias for SurvTRACE, ranking-loss overflow, peft/torchao on Kaggle).
**MMV removed:** it is not in the paper (it came from UniSurv via our own version-1
experiment) and our regression head (Eq. 6) already covers the same idea.

### What steps 0-1 changed
- **Protocol fixes found by the checks:** the files used for the censoring weights and
  the Kaplan-Meier "best guesses" were built from all patients, including test ones -
  now from each split's training patients only; splits are now exactly 60/10/30;
  the Brier score is now exactly scikit-survival's; checkpoint selection direction is
  now always correct.
- **Regression head (sec. 2.4) made to work:** config class, data types, a dead output
  (now softplus) and a Kaplan-Meier crash fixed; time unit = end of follow-up.
- **New:** mismatch penalty L_MM (Eq. 7, both t_q options), one loss recipe per
  combination (`tasks/losses=v2`), the full metric suite (`tasks/metrics=v2`),
  baselines Cox / RSF / DeepSurv / PC-Hazard (`sat.baselines`) and official SurvTRACE
  (`sat.survtrace_official`), DeepHit synthetic data, the LLM's DeepHit-style loss,
  illegal-sequence penalty and any backbone (GPT-2, Qwen2.5-0.5B, LoRA), grid-search
  tuning in the runner.
- Every model is scored by the same code (`sat/evaluate/shared.py`).
- Data generated at run time: `scripts/prepare_ebmt.py --no-composite`,
  `scripts/prepare_public_synthetic.py`, `scripts/fetch_survtrace.py`.
- Check of the whole data protocol: `python scripts/verify_protocol.py`.

## In one sentence
Test every idea in the paper from scratch, on public real data and published synthetic
data, using the standard protocol that the field uses, so our numbers can sit next to
SurvTRACE's.

## What is in, and what is not
| In version 2 | Later |
|---|---|
| Paper §2 (the method), §3 (the LLM), §4.1 (tabular data), §4.3 (multiple events) | §4.2 (sequential data) and our own synthetic data → version 3 |
| | SEER, EHRShot, MIMIC, Zahra's data → when we get access |

## The data (only data that already exists)
| Dataset | What it is | Patients | Events |
|---|---|---|---|
| METABRIC | breast cancer patients (real) | 1,904 | 1: death |
| SUPPORT | seriously ill hospital patients (real) | 8,873 | 1: death |
| EBMT | bone-marrow transplant patients (real) | 2,279 | 4: recovery, adverse event, relapse, death |
| hsa_synthetic | synthetic set from Tjandra et al. 2021 | 5,000 | 2, both can happen |
| DeepHit synthetic | synthetic set from the DeepHit paper (Lee et al. 2018) | 30,000 | 2, only one can happen |

## The questions we answer
1. **Does our way of feeding numbers in work?** (§2.2) Our numeric embedding vs. cutting
   numbers into bins.
2. **How should the transformer's outputs be combined?** (§2.2) Sum, mean, concatenate or
   pooled, and which layers.
3. **Do the extra losses help?** (§2.3) Every combination is tested — see the table below.
4. **Does a separate "time of event" head work?** (§2.4) The regression head with its
   losses, alone and together with the survival head.
5. **Can a language model do survival analysis directly?** (§3) Three language models
   (GPT-2, a ~0.5B model, one trained from scratch) and three ways to train them.
6. **Where do we stand against other methods?** (§4.1, §4.3) Cox, Random Survival Forest,
   DeepSurv, DeepHit, PC-Hazard, DSM, MENSA and SurvTRACE (run from its official code).
   Dynamic-DeepHit needs patient histories, so it moves to version 3 with §4.2.

## Every loss combination
Our model always uses the survival loss **L_PCH** as its base. On top of it:

**Survival head**
| # | Losses | Single event | Multiple events |
|---|---|---|---|
| S1 | L_PCH alone | yes | yes |
| S2 | L_PCH + L_rank | yes | yes |
| S3 | L_PCH + L_mul | — (needs 2+ events) | yes |
| S4 | L_PCH + L_rank + L_mul | — | yes |

**Regression ("time of event") head**
| # | Losses | Single event | Multiple events |
|---|---|---|---|
| R1 | MAE on patients with an observed event only | yes | yes |
| R2 | MAE with the "best guess" time for censored patients (Eq. 6) | yes | yes |
| R3 | R2 + mismatch penalty L_MM (Eq. 7, as the paper defines it) | — (needs 2+ events) | yes |
| R4 | R2 + L_MM using the "best guess" time for the unobserved event | — | yes |

**Model setups that use these**
- Survival head only: every S combination.
- Regression head only: every R combination.
- Both heads together: **every S × every R** combination.

So nothing is skipped: on single-event data that is 2 + 2 + 4 = 8 setups, on
multi-event data 4 + 4 + 16 = 24 setups. If run times turn out too long, we cut only
after the smoke test shows the real cost, and we will say which ones were cut.

## How we test (the standard protocol, from DSM and SurvTRACE)
- Split each dataset 60% train / 10% validation / 30% test, **10 times with different
  splits**, and report the **mean (standard deviation)**, as SurvTRACE does. Every model
  gets the same 10 splits.
- Main score: **C_td** (how well the model ranks patients by risk) at the 25%, 50% and
  75% points of the event times. Second score: **Brier score** (how accurate the
  predicted probabilities are). For multi-event data, each event is scored separately.
- Settings (learning rate, layers, ...) are chosen on the validation set with
  SurvTRACE's search grid. Every model, ours and the baselines, gets the same search.
- Extra scores, because the regression head predicts a time: calibration (integrated
  Brier score, D-calibration) and time errors (several kinds of MAE).

## Steps
| Step | What | Where it runs |
|---|---|---|
| 0 | **Check the base**: data counts, splits, metrics against reference libraries, the loss directions | locally, small tests |
| 1 | Build what is missing (mismatch loss, regression head in the runner, new metrics, new baselines, DeepHit synthetic loader, LLM losses, tuning) | locally |
| 2 | Smoke test every notebook (a few epochs) and measure run times | locally |
| 3 | Full runs | Kaggle |
| 4 | New report from version 2 results only | locally |

## Notebooks (Kaggle)
| # | Notebook | Answers | Needs |
|---|---|---|---|
| 01 | `01_tuning` | settings of every model + loss weights, on validation only | - |
| 02 | `02_tabular_benchmark` | 6 on METABRIC, SUPPORT (SurvTRACE Table 2 layout) | 01 |
| 03 | `03_multievent_benchmark` | 6 on EBMT, hsa_synthetic, DeepHit synthetic | 01 |
| 04 | `04_ablation_encoding_representation` | 1, 2 | 01 |
| 05 | `05_ablation_losses` | 3 (every S recipe) | 01 |
| 06 | `06_regression_head` | 4 (R1-R4, regression only and both heads) | 01 |
| 07 | `07_llm_end_to_end` | 5 (GPT-2, Qwen2.5-0.5B, scratch x 3 objectives) | - |
| 08 | `08_report` | every table + `REPORT.md` | outputs of 02-07 |

**Size and time of each notebook** (10 seeds, Kaggle T4, 2 parallel runs; estimated from
the per-run times of the first Kaggle runs, so treat them as rough):

| # | Runs | What the runs are | Time |
|---|---|---|---|
| 01 | 485 + 91 | A: per dataset ours 24, SurvTRACE 24, Cox 5, RSF 12, DeepSurv 16, PC-Hazard 16 (split 0). B: L_rank 9, time unit 2, and on multi-event data L_mul 9 + L_MM 3 | ~5 h |
| 02 | 180 | 9 models x 2 datasets x 10 seeds | ~1-1.5 h |
| 03 | 270 | 9 models x 3 datasets x 10 seeds | ~3-4 h |
| 04 | 320 | 3 encodings x 2 datasets x 10 + 13 representations x 2 datasets x 10 | ~3 h |
| 05 | 160 | S1-S2 on 2 single-event sets, S1-S4 on 3 multi-event sets, x 10 | ~2 h |
| 06 | 720 | 8 setups per single-event set, 24 per multi-event set, x 10 | ~8-9 h (may need one resume) |
| 07 | 270 | 3 backbones x 3 objectives x 3 datasets x 10, one at a time | ~40-50 h (4-5 sessions) |
| 08 | - | tables only | minutes |

**How to run on Kaggle**
1. Upload a notebook (File -> Import), turn on **Internet** and the **GPU (T4)**. It clones
   the code from GitHub; nothing else has to be uploaded.
2. Run **01 first**. Then 02-07 in any order (07 does not need 01): in each, *Add Input* ->
   the output of `01_tuning` (it reads `tuned.json` from there).
3. A session stops launching runs after ~11 h. To continue, run the notebook again with its
   own previous output added as an input: finished runs are copied back and skipped.
4. Finally 08 with the outputs of 02-07 as inputs.
`SMOKE_TEST = True` in the options cell runs a tiny version (1 seed, 2 epochs) to check
that a notebook works before the real run.

The notebooks are generated by `build_notebooks.py` from shared cells; the experiment plan
itself (datasets, models, grids, recipes) is in `scripts/v2.py`.

## Notes
- The paper's ranking-loss formula (Eqs. 4–5) is missing a minus sign; we use the
  correct direction (as in DeepHit), and a test checks it.
