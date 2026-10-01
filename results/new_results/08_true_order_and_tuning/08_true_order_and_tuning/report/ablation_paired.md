### Paired differences vs L_PCH (same seed = same split and init)

|                                                                                           | vs    |   n |   delta |   delta_sd | wins   |   p_ttest |   p_wilcoxon |
|:------------------------------------------------------------------------------------------|:------|----:|--------:|-----------:|:-------|----------:|-------------:|
| ('metabric', 'C_td ↑', 'L_PCH + L_rank')                                                  | L_PCH |   5 |  0.0013 |     0.0113 | 3/5    |  0.809    |       0.812  |
| ('sim_base', 'C_td ↑', 'L_PCH + L_mul')                                                   | L_PCH |   5 | -0.005  |     0.0025 | 0/5    |  0.0108   |       0.0625 |
| ('sim_base', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                                  | L_PCH |   5 |  0.0007 |     0.0035 | 2/5    |  0.675    |       0.812  |
| ('sim_regime_semicompeting', 'C_td ↑', 'L_PCH + L_mul')                                   | L_PCH |   5 | -0.0113 |     0.003  | 0/5    |  0.00113  |       0.0625 |
| ('sim_regime_semicompeting', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                  | L_PCH |   5 | -0.0012 |     0.0048 | 2/5    |  0.607    |       0.625  |
| ('sim_base', 'Within-subject C ↑', 'L_PCH + L_mul')                                       | L_PCH |   5 | -0.0022 |     0.008  | 1/5    |  0.567    |       0.625  |
| ('sim_base', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   5 | -0.0138 |     0.0061 | 0/5    |  0.00717  |       0.0625 |
| ('sim_regime_semicompeting', 'Within-subject C ↑', 'L_PCH + L_mul')                       | L_PCH |   5 |  0.0491 |     0.0192 | 5/5    |  0.00467  |       0.0625 |
| ('sim_regime_semicompeting', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   5 |  0.0438 |     0.0161 | 5/5    |  0.00365  |       0.0625 |
| ('sim_base', 'Within C gain over KM ↑', 'L_PCH + L_mul')                                  | L_PCH |   5 | -0.0022 |     0.008  | 1/5    |  0.567    |       0.625  |
| ('sim_base', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   5 | -0.0138 |     0.0061 | 0/5    |  0.00717  |       0.0625 |
| ('sim_regime_semicompeting', 'Within C gain over KM ↑', 'L_PCH + L_mul')                  | L_PCH |   5 |  0.0491 |     0.0192 | 5/5    |  0.00467  |       0.0625 |
| ('sim_regime_semicompeting', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)') | L_PCH |   5 |  0.0438 |     0.0161 | 5/5    |  0.00365  |       0.0625 |
| ('sim_base', 'True-order C ↑', 'L_PCH + L_mul')                                           | L_PCH |   5 | -0.0025 |     0.0034 | 1/5    |  0.181    |       0.188  |
| ('sim_base', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')                          | L_PCH |   5 | -0.0057 |     0.0051 | 0/5    |  0.066    |       0.0625 |
| ('sim_regime_semicompeting', 'True-order C ↑', 'L_PCH + L_mul')                           | L_PCH |   5 | -0.0246 |     0.0105 | 0/5    |  0.00637  |       0.0625 |
| ('sim_regime_semicompeting', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')          | L_PCH |   5 | -0.04   |     0.0103 | 0/5    |  0.00095  |       0.0625 |
| ('sim_base', 'True-order Brier ↓', 'L_PCH + L_mul')                                       | L_PCH |   5 |  0.0054 |     0.0011 | 0/5    |  0.000444 |       0.0625 |
| ('sim_base', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   5 |  0.0158 |     0.0008 | 0/5    |  1.94e-06 |       0.0625 |
| ('sim_regime_semicompeting', 'True-order Brier ↓', 'L_PCH + L_mul')                       | L_PCH |   5 |  0.0134 |     0.0012 | 0/5    |  1.35e-05 |       0.0625 |
| ('sim_regime_semicompeting', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   5 |  0.0175 |     0.0025 | 0/5    |  9.46e-05 |       0.0625 |
| ('sim_base', 'True-order log-loss ↓', 'L_PCH + L_mul')                                    | L_PCH |   5 |  0.0125 |     0.0026 | 0/5    |  0.000438 |       0.0625 |
| ('sim_base', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')                   | L_PCH |   5 |  0.0354 |     0.0019 | 0/5    |  2.14e-06 |       0.0625 |
| ('sim_regime_semicompeting', 'True-order log-loss ↓', 'L_PCH + L_mul')                    | L_PCH |   5 |  0.0298 |     0.0021 | 0/5    |  6.14e-06 |       0.0625 |
| ('sim_regime_semicompeting', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')   | L_PCH |   5 |  0.0387 |     0.0053 | 0/5    |  8.29e-05 |       0.0625 |
| ('metabric', 'Brier ↓', 'L_PCH + L_rank')                                                 | L_PCH |   5 | -0.0033 |     0.0067 | 3/5    |  0.324    |       0.312  |
| ('sim_base', 'Brier ↓', 'L_PCH + L_mul')                                                  | L_PCH |   5 |  0.0017 |     0.0021 | 2/5    |  0.15     |       0.312  |
| ('sim_base', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                                 | L_PCH |   5 |  0.0148 |     0.0025 | 0/5    |  0.000192 |       0.0625 |
| ('sim_regime_semicompeting', 'Brier ↓', 'L_PCH + L_mul')                                  | L_PCH |   5 |  0.0039 |     0.0014 | 0/5    |  0.00391  |       0.0625 |
| ('sim_regime_semicompeting', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   5 |  0.0152 |     0.0038 | 0/5    |  0.000838 |       0.0625 |
| ('metabric', 'MAE-margin ↓', 'L_PCH + L_rank')                                            | L_PCH |   5 | -0.8546 |     2.1366 | 4/5    |  0.422    |       0.438  |
| ('sim_base', 'MAE-margin ↓', 'L_PCH + L_mul')                                             | L_PCH |   5 |  0.4029 |    10.3248 | 3/5    |  0.935    |       0.812  |
| ('sim_base', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                            | L_PCH |   5 | 20.0185 |     8.1099 | 0/5    |  0.00526  |       0.0625 |
| ('sim_regime_semicompeting', 'MAE-margin ↓', 'L_PCH + L_mul')                             | L_PCH |   5 |  7.7974 |     7.5635 | 1/5    |  0.0825   |       0.125  |
| ('sim_regime_semicompeting', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   5 | 25.3502 |     9.9757 | 0/5    |  0.00473  |       0.0625 |
