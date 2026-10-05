# Version 3 — datasets to consider later

Not implemented. A list of standard survival datasets beyond the five of version 2
(METABRIC, SUPPORT, EBMT, hsa_synthetic, DeepHit synthetic), to decide on later.

The descriptions are from the literature and still have to be checked when a dataset is
picked: exact size, events, censoring rate and the licence.

## If one is added
- Add it in `scripts/v2.py` (`DATASETS`) with a small prepare script, like
  `scripts/prepare_ebmt.py`.
- **Notebook 01 has to include it.** Otherwise `tuned.json` has no entry for it: the
  model settings silently fall back to defaults, and 05/06 crash looking for its loss
  weights. Attach 01's previous output so that only the new dataset's runs execute.
- Then 02 (single event) or 03 (multiple events), and 05 and 06. 04 and 07 are optional.
- Cost: a small dataset adds about 40-60 min to 01, and about as much as METABRIC or EBMT
  to each later notebook.

## A. Free, no account

### Single event
| Dataset | Patients | Event | Get it from | Used by | Why |
|---|---|---|---|---|---|
| **GBSG** (Rotterdam–GBSG) | ~2,232 | recurrence-free survival | `pycox.datasets.gbsg` | DeepSurv, pycox | The third standard tabular set after METABRIC and SUPPORT |
| **FLCHAIN** | ~7,874 | death | `pycox.datasets.flchain` | pycox, deep survival papers | Large, real |
| **NWTCO** | ~4,028 | relapse | `pycox.datasets.nwtco` | pycox, DSM-style papers | Standard |
| **WHAS500** | 500 | death after heart attack | `sksurv.datasets.load_whas500` | textbooks, scikit-survival | Small classic |
| **AIDS (ACTG 320)** | ~1,151 | AIDS or death | `sksurv.datasets.load_aids` | scikit-survival | Clinical trial |

### Multiple events (more useful for §4.3)
| Dataset | Patients | Events | Get it from | Used by | Why |
|---|---|---|---|---|---|
| **Rotterdam (full)** | 2,982 | recurrence and death (semi-competing, like EBMT) | R `survival::rotterdam` (Rdatasets CSV) | MENSA (2024) | A second real semi-competing set |
| **MGUS2** | 1,384 | progression to plasma-cell cancer vs death (competing) | R `survival::mgus2` | competing-risks papers | Real competing risks; our only competing data now is synthetic |
| **Colon** (chemotherapy trial) | 929 | recurrence and death | R `survival::colon` | classic two-event set | Trial data |
| **PBC** (Mayo liver trial) | 418 | transplant vs death (competing) | R `survival::pbc` | classic | Small |
| **Prostate (Byar)** | ~500 | prostate-cancer death vs other causes | Vanderbilt biostat datasets | classic competing risks | Small |

## B. Free with registration
| Dataset | What | Used by | Note |
|---|---|---|---|
| **PRO-ACT** (ALS) | ~10,000 ALS patients, several events (speech, swallowing, walking, death, ...) | MENSA (2024) | The strongest real multi-event benchmark; register on the PRO-ACT website |
| **Framingham** (BioLINCC teaching set) | heart-disease cohort, competing events | competing-risks papers | BioLINCC request |

## C. Hard to get (not now)
| Dataset | Problem |
|---|---|
| **SEER** | We could not get SEER*Stat access |
| **MIMIC-III / IV** | PhysioNet credentialing and a training course (weeks) |
| **UNOS, CRASH-2** | Institutional data requests |
| **EHRShot** | Stanford data use agreement |

## D. Patient histories (for §4.2, this version)
| Dataset | What | Note |
|---|---|---|
| **PBC2** (`JM::pbc2`) | repeated lab visits over time, death / transplant | Used by Dynamic-DeepHit; free |
| **Framingham** (repeated visits) | repeated examinations | BioLINCC request |

## E. Many datasets at once
**SurvSet** (Drysdale 2022): a Python package with 76 ready survival datasets in one
format. Useful if broad coverage is asked for; most of its datasets are small.

## Suggested first picks
1. **GBSG** — completes the METABRIC / SUPPORT / GBSG trio.
2. **Rotterdam** (recurrence + death) — real semi-competing, used by MENSA.
3. **MGUS2** — real competing risks.
4. Maybe **PRO-ACT** — the best real multi-event set; needs registration.

Items 1-3 are free and go through the same pipeline as METABRIC / EBMT.
