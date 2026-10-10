# Version 2 explained from zero

This document explains, without assuming any background, what survival analysis is, what our model
does, what every loss and dataset is, what we ran in version 2, and what came out. Numbers are from
notebooks 01–06 (the language-model notebook 07 is not included).

---

## Part 1 — The problem

### 1.1 What is survival analysis?
We want to predict **when** something will happen to a patient — death, relapse of a cancer, recovery
after a transplant. That "something" is called an **event**, and the time until it happens is the
**event time**.

It is not a normal prediction problem because of **censoring**: for many patients we never see the
event. A study runs for, say, 10 years; a patient who is still alive at the end, or who moved away
after 3 years, has no event time. We only know *"the event had not happened by time T"*. Such a patient
is **censored** at time T. Throwing censored patients away would bias everything (they are often the
healthiest ones), so survival models are built to use the partial information they carry.

### 1.2 What does a survival model output?
Instead of one number, a survival model usually outputs a **survival curve** for each patient:

> S(t) = the probability that the event has **not** happened by time t.

S starts at 1 (at time 0 nothing has happened) and goes down. A high-risk patient's curve drops fast,
a low-risk patient's curve drops slowly. From the curve you can read a risk at any time, or an
expected event time.

### 1.3 One event or several events?
- **Single event**: one kind of event, e.g. death.
- **Multiple events**: several kinds per patient, e.g. after a bone-marrow transplant: recovery, an
  adverse event, relapse, death. They can be:
  - **competing**: only the first one is seen (if you die of A you cannot later die of B);
  - **semi-competing / non-exclusive**: several can happen to the same patient, in some order
    (recovery, then relapse, then death).

Our paper (DMMST) is about a model that handles **multiple events**.

---

## Part 2 — Our model

### 2.1 The idea in one paragraph
Each patient is a list of features (age, lab values, treatment, ...). We turn every feature into a
small vector (a **token**), feed all tokens of a patient into a **transformer** (the same kind of
network used in language models), and get one vector that summarises the patient. On top of that
vector sit **heads** — small networks that produce the outputs:

- the **survival head** outputs one survival curve per event;
- the **regression head** (optional) outputs one predicted event time per event.

### 2.2 The numeric embedding (paper §2.2)
How do you give a number like "age = 63" to a transformer? Two options:
- **Discretise** it: cut age into bins (e.g. 60–65) and treat the bin as a word. Simple, but loses
  precision.
- **Numeric embedding** (what the paper proposes): every feature has its own learned vector, and that
  vector is **multiplied by the (standardised) value**. Age 63 and age 64 then give almost the same
  input, as they should.

We also tried **quantile** scaling (the value is replaced by its rank among all patients before the
multiplication).

### 2.3 Representation: how to get one patient vector
The transformer outputs a vector per token per layer. We tried different ways to combine them:
- over **layers**: sum, mean, concatenate, or a special **[CLS]** token;
- over **tokens**: max or mean;
- **all layers** or only the **last** one.

### 2.4 How the survival head works (L_PCH)
Time is cut into a few intervals (e.g. 0–14 days, 14–18 days, ...). In each interval the head predicts
a **hazard**: how likely the event is in that interval if it has not happened yet. Multiplying the
"no event" chances interval by interval gives the survival curve. This is called a
**piecewise-constant hazard (PCH)** model.

The model is trained by **L_PCH**, the likelihood of what we actually observed:
- for a patient with an event at time T: "survive until T, then the event happens at T";
- for a censored patient at time T: "survive until T" (nothing more is claimed).
This is how censored patients are used correctly.

---

## Part 3 — The losses
A **loss** is the number the model tries to make small during training. L_PCH alone already gives a
working model. The paper adds more losses; testing whether they help is a big part of version 2.

| Loss | Paper | What it asks of the model, in plain words |
|---|---|---|
| **L_PCH** | Eq. 3 | Make the observed event times (and censoring times) likely. The base loss. |
| **L_rank** | Eq. 4 | **Between patients**: if patient A had the event before patient B, A should have been predicted as riskier than B (for the same event). |
| **L_mul** | Eq. 5 | **Within one patient**: if a patient had event 1 before event 2, event 1 should have been predicted as riskier at that time. Needs ≥ 2 events. |
| **L_MAE** | Eq. 6 | For the regression head: the predicted time should be close to the real time. |
| **L_MM** | Eq. 7 | For the regression head: the predicted times of the different events should agree with each other (the "mismatch" penalty). Needs ≥ 2 events. |

Two details:
- **Best guess (Eq. 6)**: for a censored patient we do not know the true time, only that it is later
  than T. The "best guess" is an estimate of the real time made from all training patients
  (Kaplan–Meier curve): "patients who were still event-free at T had their event on average at time X".
  L_MAE can either ignore censored patients (**R1**) or use their best guess (**R2**).
- **σ and weight**: L_rank and L_mul have a strength (weight) and a sharpness (σ). These were tuned.
- The paper writes Eqs. 4–5 with the wrong sign; we use the correct direction (as in DeepHit) and a
  test checks it.

### The recipes we tested
Survival head:

| Recipe | Losses |
|---|---|
| **S1** | L_PCH only |
| **S2** | L_PCH + L_rank |
| **S3** | L_PCH + L_mul (multi-event data only) |
| **S4** | L_PCH + L_rank + L_mul (multi-event data only) |

Regression head:

| Recipe | Losses |
|---|---|
| **R1** | L_MAE on patients with an observed event only |
| **R2** | L_MAE with the best guess for censored patients |
| **R3** | R2 + L_MM, with the observed time as reference |
| **R4** | R2 + L_MM, with the best guess as reference |

"—" in a table means the recipe needs ≥ 2 events and does not exist on that dataset.

We also removed one loss that was in earlier versions, **MMV** (from another paper, UniSurv): it is
not in our paper and does the same job as our regression head.

---

## Part 4 — The data

| Dataset | Real? | Patients | Events | What it is |
|---|---|---|---|---|
| **METABRIC** | real | 1,904 | 1: death | breast-cancer patients; 9 features (genes + clinical) |
| **SUPPORT** | real | 8,873 | 1: death | seriously ill hospital patients; 14 features |
| **EBMT** | real | 2,279 | 4: recovery, adverse event, relapse, death | bone-marrow transplant patients; only 4 categorical features |
| **hsa synthetic** | simulated | 5,000 | 2, both can happen | from Tjandra et al. (2021); ~70% censored at the end of the study |
| **DeepHit synthetic** | simulated | 30,000 | 2, competing | from the DeepHit paper (Lee et al. 2018) |

METABRIC and SUPPORT are the standard datasets of SurvTRACE, our closest competitor, so we can compare
with its published numbers. **SEER** was planned as a large real dataset, but because of the updated
SEER+ data-access policies we could not get access to it.

---

## Part 5 — How we measured (the protocol)

### 5.1 Splits
Each dataset is split at random into **60% training** (the model learns from it), **10% validation**
(used to choose settings and when to stop training) and **30% test** (used only for the final numbers).
This is done **10 times** with different random splits, and every model gets the same 10 splits.
Results are written as **mean (standard deviation)** over the 10 splits — e.g. 0.667 (0.013). The
standard deviation shows how much the result moves from split to split.

### 5.2 Tuning (notebook 01)
Every model has settings (learning rate, network size, ...). We tried many combinations on the
**validation** split and kept the best — for every model, not only ours, so the comparison is fair.
Then, with the chosen network, we tuned the loss weights (L_rank, L_mul, L_MM).

### 5.3 Early stopping and "patience"
During training the model is checked on validation regularly. If it has not improved for a number of
checks (the **patience**), training stops. We use 30 for our loss and regression experiments (Part 9).

### 5.4 The scores

| Score | Meaning | Good value |
|---|---|---|
| **C_td** | Of all pairs of patients, how often is the one who had the event first predicted as riskier? Measured at the 25%, 50% and 75% points of the event times, then averaged. | higher; 0.5 = coin flip, 1 = perfect |
| **Brier score** | Squared error of the predicted probability "event by time t" vs what happened. | lower |
| **IBS** | Brier score averaged over the whole follow-up. | lower |
| **IBLL** | Like IBS, but with log-loss. | lower |
| **AUC** | Like C_td, another standard way to measure ranking at a time point. | higher |
| **Antolini C** | C_td version that uses the whole curve. | higher |
| **D-calibration p** | A test of whether predicted probabilities match observed frequencies. Large test sets make it very strict. | higher (> 0.05 = no evidence of a problem) |
| **MAE** | Average error of a predicted event time, in the dataset's time unit (days/months). Censored patients are handled with the best guess ("margin") or "pseudo-observations". | lower |
| **Harrell C** | Like C_td but for a predicted *time*: are earlier events predicted earlier? | higher |

In short: **C_td = does the model put patients in the right order; IBS = are its probabilities right.**

---

## Part 6 — The models we compared against

| Model | What it is |
|---|---|
| **SurvTRACE** | A transformer survival model (2022), run with the authors' own code. Our closest competitor. |
| **DeepHit** | Neural network that predicts the event time distribution directly (2018). |
| **DSM** | Deep Survival Machines: mixture of simple distributions (2021). |
| **MENSA** | Recent multi-event model (2024). |
| **RSF** | Random Survival Forest: many decision trees. Strong classic. |
| **PC-Hazard** | Simple neural network with the same piecewise hazard as our head. |
| **DeepSurv** | Neural version of the Cox model. |
| **Cox** | The classic statistical survival model (linear). |

DeepHit, DSM and MENSA use our transformer with their own head; the others are separate models. On
multi-event data, single-event models are trained once per event.

---

## Part 7 — What each notebook did

| Notebook | What it did | Runs |
|---|---|---|
| **01 Tuning** | chose the settings of every model and the loss weights, on validation only | 576 |
| **02 Tabular benchmark** | all 9 models on METABRIC and SUPPORT (single event) | 180 |
| **03 Multi-event benchmark** | all 9 models on EBMT, hsa synthetic, DeepHit synthetic | 270 |
| **04 Encoding / representation** | numeric embedding vs quantile vs bins; 13 ways to pool the transformer | 320 |
| **05 Survival losses** | S1–S4 on all datasets | 160 |
| **06 Regression head** | R1–R4 alone, and every S × R together with the survival head ("joint") | 720 |

---

## Part 8 — The results, in plain words

### 8.1 Are we better than the other models? (02, 03)
C_td, mean over 10 splits:

| | METABRIC | SUPPORT | EBMT | hsa synth. | DeepHit synth. |
|---|---|---|---|---|---|
| Best other model | SurvTRACE 0.678 | SurvTRACE 0.638 | RSF 0.549 | SurvTRACE 0.868 | DeepSurv 0.762 |
| Our best recipe | 0.672 | 0.620 | 0.537 | **0.868** | 0.744 |
| Our rank (of 9, S1) | 5 | 4 | 6 | 3 | 5 |

**Plainly: our model is good but not the best.** It is always in the better half and ties SurvTRACE
on hsa synthetic, but on the other four datasets something else is slightly better (by 0.006–0.018).
Our **calibration** (IBS) is good — close to the best on four datasets, and better than SurvTRACE on
METABRIC. A sanity check also passed: our run of SurvTRACE gets almost its published numbers
(0.678 vs 0.691, 0.638 vs 0.640), so the setup is right.

**EBMT is hard for every model** (all between 0.50 and 0.55): its 4 features say little about the
outcome.

### 8.2 Do the extra losses help? (05)

| C_td | METABRIC | SUPPORT | EBMT | hsa synth. | DeepHit synth. |
|---|---|---|---|---|---|
| S1: L_PCH | 0.667 | 0.615 | 0.529 | 0.849 | 0.743 |
| S2: + L_rank | 0.670 | 0.620 | 0.535 | 0.868 | 0.743 |
| S3: + L_mul | — | — | 0.530 | 0.859 | 0.742 |
| S4: + both | — | — | 0.537 | 0.867 | 0.742 |

- **L_rank helps a little, on every dataset.** The gains are small; statistically clear only on hsa.
- **Their real benefit is reliability.** With L_PCH alone, training sometimes goes wrong on one split:
  on hsa one split reached only 0.71 (the others ~0.86), and on EBMT several splits had very bad
  probabilities (Brier up to 0.6). With L_rank + L_mul (S4) this did not happen: EBMT's IBS went from
  0.338 to 0.233 and became much more stable.
- On DeepHit synthetic nothing changes — L_PCH alone is enough there.

### 8.3 Does the numeric embedding help? (04)
- METABRIC: yes — 0.666 vs 0.642 with bins.
- SUPPORT: no — bins are slightly better (0.627 vs 0.617).
- How the transformer outputs are combined hardly matters (all within ±0.01), except that the [CLS]
  token is clearly worst.

### 8.4 Does the regression head work? (06)
- **R2 (best guess) is better than R1 (observed only)**: it orders the predicted times better on every
  dataset (Harrell C, e.g. 0.57 → 0.66 on hsa) and has a smaller error on 4 of 5 datasets.
- **L_MM (R3/R4)** changes little.
- **Training both heads together** ("joint"):
  - does not hurt the survival predictions, as long as L_rank is used (on hsa: 0.84–0.85 with L_rank,
    0.72–0.82 without);
  - on hsa it even makes the predicted times better (Harrell C 0.69 on average, up to 0.79, vs 0.66
    alone);
  - on DeepHit synthetic the predicted times get worse (0.55 vs 0.70), because joint models pick their
    best checkpoint by the survival score, not by the time prediction.

### 8.5 What this means for the paper

| Claim | Supported? |
|---|---|
| "Better accuracy than other models" | **No** — competitive, upper half |
| "Well calibrated" | Mostly (not on EBMT) |
| "L_rank and L_mul improve the model" | Yes, but as **stability and calibration**; accuracy gains are small |
| "The numeric embedding is better than bins" | On METABRIC yes, on SUPPORT no |
| "The regression head works; the best-guess loss helps" | Yes |
| "Training both heads together helps" | On hsa yes; neutral elsewhere; worse time prediction on DeepHit synthetic |

---

## Part 9 — Problems we found and fixed along the way
Version 2 was run on Kaggle; several issues were found from the results and fixed before the final
numbers. They are listed so you know the final results are trustworthy.

| Problem | What happened | Fix |
|---|---|---|
| SurvTRACE crashed | its 2021 code used a NumPy name removed in NumPy 2 | restored the name |
| Ranking loss gave NaN | on whole-number times a formula ran outside its interval and overflowed | kept it inside the interval |
| Qwen (LLM) crashed | a library version clash on Kaggle | bypassed the check |
| IBS/IBLL/AUC missing on METABRIC | some test patients were followed longer than any training patient | handle them correctly (cap the time) |
| Brier wrong on hsa for neural models | 70% of hsa patients (censored at the end) were silently dropped from the Brier score | keep them |
| Joint models collapsed | regression loss measured in days (hundreds) drowned the survival loss | measure time as a fraction of follow-up |
| Some joint runs never learned | they start slowly and were stopped too early (patience 10) | patience 30 (and the same for notebook 05, to be fair) |
| "Last layer" runs crashed (04) | wrong layer index | use the real last-layer number |
| Kaggle killed long sessions | runs went past the 12-hour limit and nothing was saved | notebooks stop at 11 h 20 and save |

---

## Part 10 — Where everything is
- Results of each notebook: `results/version_2/<notebook>/`
- Figures and reports: `results/version_2/analysis/`
  - `REPORT.md` — the short comparison of our losses vs other models
  - `report_latex/report.pdf` — 3-page report
  - `report_latex/full_report.pdf` — 16-page report with every table
  - `EXPLAINED.md` — this document
- The plan of version 2: `notebooks/version_2/VERSION.md`
- Not yet included: notebook 07 (the language-model part of the paper, §3) and sequential data (§4.2,
  planned for version 3).
