# Version 3 — our own synthetic data (future)

**Status:** not started. Comes after version 2.

## In one sentence
Use data that we generate ourselves, where we know the true answer, to test the paper's
claims under controlled conditions — including sequential patient histories (§4.2),
which version 2 has no data for.

## What will be in it
| Part | What it is |
|---|---|
| Multi-event simulator (`sat/data/simulate.py`) | Generates patients with 2–8 events. We can change one thing at a time — whether events compete, how much is censored, how many patients, how non-linear the risk is — and we know each patient's true risk, so we can measure the real error. |
| EHR sequence simulator (`sat/data/simulate_ehr.py`) | Generates 5-year visit histories (diagnosis codes, medicines, lab values) with outcomes for kidney disease, diabetes, COPD and death. We control how much of the risk is only visible in the *order and trend* of the history. This tests §4.2. |
| Dynamic-DeepHit | The standard baseline for patient histories over time; it needs sequences, so it belongs here. |

## Later, when we get access
SEER (cancer registry), EHRShot and MIMIC (hospital records), and Zahra's
CKD/COPD/diabetes data can replace or complement the simulated data.
