# Multi-event datasets: what exists, what we have, what we can add

Datasets used by the multi-event survival literature (MENSA 2025, Tjandra et al. 2021,
Moon et al. 2022, Groha et al. 2020) and by our version 2. "Multi-event" here means more
than one event per patient (competing, semi-competing or fully non-exclusive).

Access: **open** = download freely; **registration** = free account + accept terms;
**credentialed** = free, but ethics course + signed data-use agreement (days to weeks);
**blocked** / **private** = not available to us.

## 1. Overview

| Dataset | Domain | Patients | Events (K) | Event type | Features | Access | Used by |
|---|---|---|---|---|---|---|---|
| **EBMT** | Bone-marrow transplant | 2,279 | 4 (5 with recovery+AE) | Semi-competing | 4 categorical (6 one-hot) | Open (R `mstate`, Python `pymsm`) | MENSA (K=5), Groha |
| **hsa synthetic** | Simulated | 5,000 | 2 | Semi-competing | Few numeric | Open | Tjandra |
| **Rotterdam** | Breast cancer | 2,982 | 2-3 (recurrence, death) | Semi-competing | ~10 clinical | Open (R `survival::rotterdam`) | MENSA |
| **PRO-ACT** | ALS clinical trials | ~3,000 (after MENSA's filtering) | 4 (speech, swallowing, handwriting, walking) | **Non-exclusive (true multi-event)** | ~8 baseline | Registration (ncri1.partners.org/ProACT) | MENSA |
| **MIMIC-IV** | ICU / hospital (Boston) | ~19,000 (MENSA cohort) | 3 (e.g. sepsis, respiratory failure, death) | Semi-competing | ~100 | Credentialed (PhysioNet) | MENSA |
| **MIMIC-III** | ICU (Boston) | 13,801 stays | 3 (respiratory failure, shock, death) | Semi-competing | Many, incl. time series | Credentialed (same PhysioNet access as MIMIC-IV) | Tjandra, Moon |
| **ADNI** | Alzheimer's | 1,604 | 2 (Alzheimer's onset, death) | Semi-competing | Clinical, cognitive, imaging | Application (adni.loni.usc.edu) | Tjandra |
| **SEER** | US cancer registry | 11k-500k+ | 1-2 (cause of death) | Competing | ~17 | Blocked (SEER+ policy) | MENSA, Tjandra |
| Dana-Farber VTE | Cancer, blood clots | - | 2 (VTE, death) | Competing | Longitudinal EHR | Private | Moon |
| Simulated illness-death (`coxed`) | Simulated | 5,000 | 2-3 states | Multi-state | Few | Open (R package) | Groha |

Patient counts and feature numbers for MENSA's datasets are from its appendix (Table 7),
whose text extraction was partly scrambled; check them when the prepare scripts are written.

## 2. Recommended order for version 3

| Priority | Dataset | Why | Effort |
|---|---|---|---|
| 1 | **Rotterdam** | Free today; MENSA reports results on it -> direct comparison | Small: one prepare script |
| 2 | **EBMT with 5 events** | Same events as MENSA -> direct comparison | Tiny: `prepare_ebmt.py` without `--no-composite` |
| 3 | **PRO-ACT** | The best true multi-event dataset (events not exclusive) - DMMST's main claim | Medium: register, build events from ALSFRS-R |
| 4 | **MIMIC-IV** (+ MIMIC-III) | Large, rich, allows data over time (the "dynamic" claim) | Large: credentialing + cohort building (reuse MENSA's or healthylaife's pipeline) |
| 5 | **ADNI** | Real multi-event with strong features; Tjandra reports results | Medium: application + preprocessing |

## 3. How a dataset is added (same for all)

1. `scripts/prepare_<name>.py` writes `data/<name>/data.csv` + `meta.json`
   (columns: `id`, numeric `x_*`, categorical `c_*`, `event1..K`, `duration1..K`),
   following `scripts/prepare_ebmt.py`.
2. One entry in `DATASETS` in `scripts/v2.py` (`exp="multievent"`, `hub=<name>`, `events=K`).
3. The notebooks pick it up (tuning, benchmark, loss and regression experiments).

## 4. Data rules

- PRO-ACT, MIMIC, ADNI: **never** commit to GitHub or upload to a public Kaggle dataset
  (same rule as SEER). Keep them in git-ignored `data/` folders; on Kaggle use a private
  dataset only if the data-use agreement allows it.
- MIMIC: the agreement also forbids sending the data to online services (including AI tools).
- Unofficial copies (e.g. user uploads of full MIMIC on Kaggle) must not be used.

## Sources

- MENSA: Lillelund et al., ML4H 2025 (`related_new_papers/MENSA_Lillelund2025.pdf`, Appendix E.1)
- Tjandra, He, Wiens, "A Hierarchical Approach to Multi-Event Survival Analysis", AAAI 2021
- Moon et al., "SurvLatent ODE", MLHC 2022 (arXiv 2204.09633)
- Groha et al., "A General Framework for Survival Analysis and Multi-State Modelling" (arXiv 2006.04893)
