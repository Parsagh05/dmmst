# Version 0 — the first experiments (archived)

**Status:** finished, superseded. Do not use these numbers.

## What this was
The very first notebooks we ran, before the code was audited:

| Notebook | What it did |
|---|---|
| `yasi_environment_and_baselines.ipynb` | Set up the pipeline and ran 5 baselines on 3 datasets |
| `parsa_mmv_loss.ipynb` | Tested the MMV loss (from UniSurv) against the plain survival loss |
| `new_exp_SA_kaggle_v2.ipynb` | A sweep of MMV loss weights |
| `multievent4_kaggle.ipynb` | A 4-event synthetic dataset test |
| `ebmt4_kaggle.ipynb` | A first try on the real EBMT dataset |

Results: [results/version_0/](../../results/version_0/)

## Why it is archived
Later we found bugs in the code these notebooks used (for example the training and
validation sets were swapped, and every seed used the same split). So all of these
results are wrong and must not go in the paper.
