# Loss functions

Scope: the loss stack described in the paper (§2.3, §2.4). See [SCOPE.md](../SCOPE.md) for
the full paper→code map.

## Generic form (Eq. 2)

```
Loss = Σ_i Σ_j L(y_ij, t_ij, e_ij) + Σ_ℓ α_ℓ P_ℓ
```

`L` is any single-event survival loss; `P_ℓ` are penalty terms with weights `α_ℓ`.
This is implemented by [`MetaLoss`](../sat/loss/meta.py), which takes a list of losses and a
list of `coeffs`. The `balance_strategy` field lets the `α_ℓ` be learned rather than fixed
(`fixed` reproduces the paper as written; the others are available for ablation).

## Components

| Symbol | Class | Config | Notes |
|---|---|---|---|
| `L_PCH` | `SATNLLPCHazardLoss` | [nllpch.yaml](../conf/tasks/losses/nllpch.yaml) | Piece-wise constant hazard NLL |
| `L_rank` (Eq. 4) | `SampleRankingLoss` | [sample_ranking.yaml](../conf/tasks/losses/sample_ranking.yaml) | Ranks **samples** within one event type |
| `L_mul` (Eq. 5) | `MultiEventRankingLoss` | [event_ranking.yaml](../conf/tasks/losses/event_ranking.yaml) | Ranks **event types** within one sample |
| `L_MAE` (Eq. 6) | `L1Loss(l1_type="margin")` | [l1.yaml](../conf/tasks/losses/l1.yaml) | Best-guess MAE; Kaplan-Meier extension for censored subjects |
| `L_MM` (Eq. 7) | `MismatchLoss` | [mismatch.yaml](../conf/tasks/losses/mismatch.yaml) | Penalises getting the *first* event wrong |

Both ranking losses share the vectorised `η(x,y) = exp((x−y)/σ)` implementation in
[`RankingLoss`](../sat/loss/base.py); they differ only in which tensor axis they compare over.
`sigma` and `margin` are exposed per-loss.

## The paper's recipe

`Loss = L_PCH + L_rank + L_mul` →
[nllpch_sample_event_ranking.yaml](../conf/tasks/losses/nllpch_sample_event_ranking.yaml)

```bash
python -m sat.finetune experiments=<data>/survival tasks/losses=nllpch_sample_event_ranking
```

Ablation ladder for the two penalty terms:

| Recipe | Loss |
|---|---|
| `nllpch` | `L_PCH` |
| `nllpch_sample_ranking` | `L_PCH + L_rank` |
| `nllpch_sample_event_ranking` | `L_PCH + L_rank + L_mul` |

Coefficients come from `likelihood_loss_coeff`, `ranking_loss_coeff`,
`event_ranking_loss_coeff` in [conf/experiments/defaults.yaml](../conf/experiments/defaults.yaml)
and are overridable from the CLI.

## Why `L_mul` needs its own metric

`L_rank` and `L_mul` optimise different things, so a single C-index cannot separate them.
`L_rank` targets across-sample ranking (standard IPCW C-index); `L_mul` targets the ordering
of events *within* one subject, which is measured by
[within-subject concordance](../sat/evaluate/within_subject_concordance/). Any ablation of
`L_mul` must report both.
