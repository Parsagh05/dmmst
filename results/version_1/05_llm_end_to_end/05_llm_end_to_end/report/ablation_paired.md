### Paired differences vs L_PCH (same seed = same split and init)

|                                                                           | vs    |   n |   delta |   delta_sd | wins   |   p_ttest |   p_wilcoxon |
|:--------------------------------------------------------------------------|:------|----:|--------:|-----------:|:-------|----------:|-------------:|
| ('ebmt', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   3 |  0.0003 |     0.0024 | 2/3    |   0.858   |         1    |
| ('metabric', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                  | L_PCH |   3 |  0.0099 |     0.0075 | 3/3    |   0.149   |         0.25 |
| ('sim_base', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                  | L_PCH |   3 | -0.004  |     0.006  | 1/3    |   0.365   |         0.5  |
| ('ebmt', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')          | L_PCH |   3 |  0.036  |     0.0043 | 3/3    |   0.00478 |         0.25 |
| ('sim_base', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 | -0.0117 |     0.0042 | 0/3    |   0.0407  |         0.25 |
| ('ebmt', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')     | L_PCH |   3 |  0.036  |     0.0043 | 3/3    |   0.00478 |         0.25 |
| ('sim_base', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)') | L_PCH |   3 | -0.0117 |     0.0042 | 0/3    |   0.0407  |         0.25 |
| ('ebmt', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                     | L_PCH |   3 |  0.0051 |     0.002  | 0/3    |   0.05    |         0.25 |
| ('metabric', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |  0.0071 |     0.0009 | 0/3    |   0.00533 |         0.25 |
| ('sim_base', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |  0.0151 |     0.003  | 0/3    |   0.0131  |         0.25 |
| ('ebmt', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                | L_PCH |   3 | 32.8835 |    59.0708 | 1/3    |   0.437   |         0.5  |
| ('metabric', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 |  3.7693 |     1.3082 | 0/3    |   0.0379  |         0.25 |
| ('sim_base', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 | 20.4121 |     9.1424 | 0/3    |   0.0608  |         0.25 |
