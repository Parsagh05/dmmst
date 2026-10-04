### Paired differences vs L_PCH (same seed = same split and init)

|                                                                                     | vs    |   n |     delta |   delta_sd | wins   |   p_ttest |   p_wilcoxon |
|:------------------------------------------------------------------------------------|:------|----:|----------:|-----------:|:-------|----------:|-------------:|
| ('ebmt', 'C_td ↑', 'L_PCH + L_mul')                                                 | L_PCH |   4 |   -0.0119 |     0.0111 | 0/4    |  0.122    |       0.125  |
| ('ebmt', 'C_td ↑', 'L_PCH + L_rank + L_mul + MMV [λv=1]')                           | L_PCH |   3 |   -0.0024 |     0.023  | 2/3    |  0.874    |       1      |
| ('ebmt', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                                | L_PCH |   5 |   -0.0078 |     0.009  | 0/5    |  0.125    |       0.0625 |
| ('ebmt', 'C_td ↑', 'L_PCH + L_rank')                                                | L_PCH |   5 |   -0.0023 |     0.0063 | 1/5    |  0.459    |       0.438  |
| ('hsa_synthetic', 'C_td ↑', 'L_PCH + L_mul')                                        | L_PCH |   5 |   -0.0214 |     0.0054 | 0/5    |  0.000905 |       0.0625 |
| ('hsa_synthetic', 'C_td ↑', 'L_PCH + L_rank + L_mul + MMV [λv=1]')                  | L_PCH |   5 |   -0.0094 |     0.0064 | 0/5    |  0.0298   |       0.0625 |
| ('hsa_synthetic', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                       | L_PCH |   5 |   -0.003  |     0.005  | 2/5    |  0.256    |       0.438  |
| ('hsa_synthetic', 'C_td ↑', 'L_PCH + L_rank')                                       | L_PCH |   5 |    0.0007 |     0.0046 | 3/5    |  0.75     |       1      |
| ('sim_base', 'C_td ↑', 'L_PCH + L_mul')                                             | L_PCH |   5 |   -0.0129 |     0.0037 | 0/5    |  0.00152  |       0.0625 |
| ('sim_base', 'C_td ↑', 'L_PCH + L_rank + L_mul + MMV [λv=1]')                       | L_PCH |   5 |   -0.0013 |     0.0093 | 3/5    |  0.776    |       0.812  |
| ('sim_base', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                            | L_PCH |   5 |    0.0004 |     0.0068 | 2/5    |  0.896    |       1      |
| ('sim_base', 'C_td ↑', 'L_PCH + L_rank')                                            | L_PCH |   5 |    0.0004 |     0.0064 | 3/5    |  0.896    |       0.812  |
| ('ebmt', 'Within-subject C ↑', 'L_PCH + L_mul')                                     | L_PCH |   4 |    0.0414 |     0.0058 | 4/4    |  0.000743 |       0.125  |
| ('ebmt', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul + MMV [λv=1]')               | L_PCH |   3 |    0.0518 |     0.0128 | 3/3    |  0.0198   |       0.25   |
| ('ebmt', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')                    | L_PCH |   5 |    0.0395 |     0.0079 | 5/5    |  0.000369 |       0.0625 |
| ('ebmt', 'Within-subject C ↑', 'L_PCH + L_rank')                                    | L_PCH |   5 |   -0.0014 |     0.003  | 2/5    |  0.344    |       0.438  |
| ('hsa_synthetic', 'Within-subject C ↑', 'L_PCH + L_mul')                            | L_PCH |   5 |   -0.009  |     0.0526 | 2/5    |  0.721    |       0.812  |
| ('hsa_synthetic', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul + MMV [λv=1]')      | L_PCH |   5 |   -0.0018 |     0.0378 | 1/5    |  0.92     |       0.625  |
| ('hsa_synthetic', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')           | L_PCH |   5 |    0.0066 |     0.0392 | 1/5    |  0.726    |       0.625  |
| ('hsa_synthetic', 'Within-subject C ↑', 'L_PCH + L_rank')                           | L_PCH |   5 |   -0.0053 |     0.0474 | 2/5    |  0.815    |       0.812  |
| ('sim_base', 'Within-subject C ↑', 'L_PCH + L_mul')                                 | L_PCH |   5 |   -0.0079 |     0.0076 | 1/5    |  0.0805   |       0.125  |
| ('sim_base', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul + MMV [λv=1]')           | L_PCH |   5 |   -0.0004 |     0.0098 | 2/5    |  0.932    |       1      |
| ('sim_base', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')                | L_PCH |   5 |   -0.0158 |     0.0111 | 0/5    |  0.034    |       0.0625 |
| ('sim_base', 'Within-subject C ↑', 'L_PCH + L_rank')                                | L_PCH |   5 |   -0.0349 |     0.0102 | 0/5    |  0.0016   |       0.0625 |
| ('ebmt', 'Within C gain over KM ↑', 'L_PCH + L_mul')                                | L_PCH |   4 |    0.0414 |     0.0058 | 4/4    |  0.000743 |       0.125  |
| ('ebmt', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul + MMV [λv=1]')          | L_PCH |   3 |    0.0518 |     0.0128 | 3/3    |  0.0198   |       0.25   |
| ('ebmt', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')               | L_PCH |   5 |    0.0395 |     0.0079 | 5/5    |  0.000369 |       0.0625 |
| ('ebmt', 'Within C gain over KM ↑', 'L_PCH + L_rank')                               | L_PCH |   5 |   -0.0014 |     0.003  | 2/5    |  0.344    |       0.438  |
| ('hsa_synthetic', 'Within C gain over KM ↑', 'L_PCH + L_mul')                       | L_PCH |   5 |   -0.009  |     0.0526 | 2/5    |  0.721    |       0.812  |
| ('hsa_synthetic', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul + MMV [λv=1]') | L_PCH |   5 |   -0.0018 |     0.0378 | 1/5    |  0.92     |       0.625  |
| ('hsa_synthetic', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   5 |    0.0066 |     0.0392 | 1/5    |  0.726    |       0.625  |
| ('hsa_synthetic', 'Within C gain over KM ↑', 'L_PCH + L_rank')                      | L_PCH |   5 |   -0.0053 |     0.0474 | 2/5    |  0.815    |       0.812  |
| ('sim_base', 'Within C gain over KM ↑', 'L_PCH + L_mul')                            | L_PCH |   5 |   -0.0079 |     0.0076 | 1/5    |  0.0805   |       0.125  |
| ('sim_base', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul + MMV [λv=1]')      | L_PCH |   5 |   -0.0004 |     0.0098 | 2/5    |  0.932    |       1      |
| ('sim_base', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')           | L_PCH |   5 |   -0.0158 |     0.0111 | 0/5    |  0.034    |       0.0625 |
| ('sim_base', 'Within C gain over KM ↑', 'L_PCH + L_rank')                           | L_PCH |   5 |   -0.0349 |     0.0102 | 0/5    |  0.0016   |       0.0625 |
| ('ebmt', 'Brier ↓', 'L_PCH + L_mul')                                                | L_PCH |   4 |    0.005  |     0.0026 | 0/4    |  0.0308   |       0.125  |
| ('ebmt', 'Brier ↓', 'L_PCH + L_rank + L_mul + MMV [λv=1]')                          | L_PCH |   3 |    0.0286 |     0.0053 | 0/3    |  0.0114   |       0.25   |
| ('ebmt', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                               | L_PCH |   5 |    0.0052 |     0.0016 | 0/5    |  0.00196  |       0.0625 |
| ('ebmt', 'Brier ↓', 'L_PCH + L_rank')                                               | L_PCH |   5 |   -0.0002 |     0.0011 | 3/5    |  0.734    |       0.812  |
| ('hsa_synthetic', 'Brier ↓', 'L_PCH + L_mul')                                       | L_PCH |   5 |    0.0333 |     0.0297 | 1/5    |  0.066    |       0.125  |
| ('hsa_synthetic', 'Brier ↓', 'L_PCH + L_rank + L_mul + MMV [λv=1]')                 | L_PCH |   5 |    0.1083 |     0.0337 | 0/5    |  0.002    |       0.0625 |
| ('hsa_synthetic', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   5 |    0.0759 |     0.0324 | 0/5    |  0.00638  |       0.0625 |
| ('hsa_synthetic', 'Brier ↓', 'L_PCH + L_rank')                                      | L_PCH |   5 |    0.0819 |     0.0284 | 0/5    |  0.00299  |       0.0625 |
| ('sim_base', 'Brier ↓', 'L_PCH + L_mul')                                            | L_PCH |   5 |    0.0055 |     0.0015 | 0/5    |  0.00112  |       0.0625 |
| ('sim_base', 'Brier ↓', 'L_PCH + L_rank + L_mul + MMV [λv=1]')                      | L_PCH |   5 |    0.0466 |     0.0045 | 0/5    |  2.12e-05 |       0.0625 |
| ('sim_base', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                           | L_PCH |   5 |    0.0154 |     0.001  | 0/5    |  4.46e-06 |       0.0625 |
| ('sim_base', 'Brier ↓', 'L_PCH + L_rank')                                           | L_PCH |   5 |    0.0138 |     0.002  | 0/5    |  9.38e-05 |       0.0625 |
| ('ebmt', 'MAE-margin ↓', 'L_PCH + L_mul')                                           | L_PCH |   4 |   50.32   |    60.8157 | 1/4    |  0.197    |       0.25   |
| ('ebmt', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul + MMV [λv=1]')                     | L_PCH |   3 | -128.724  |    48.3621 | 3/3    |  0.044    |       0.25   |
| ('ebmt', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                          | L_PCH |   5 |   46.3819 |    37.6242 | 0/5    |  0.051    |       0.0625 |
| ('ebmt', 'MAE-margin ↓', 'L_PCH + L_rank')                                          | L_PCH |   5 |   12.7271 |    38.2975 | 2/5    |  0.499    |       0.438  |
| ('hsa_synthetic', 'MAE-margin ↓', 'L_PCH + L_mul')                                  | L_PCH |   5 |  -12.6755 |    20.97   | 4/5    |  0.248    |       0.312  |
| ('hsa_synthetic', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul + MMV [λv=1]')            | L_PCH |   5 |  -30.6157 |    15.0864 | 5/5    |  0.0105   |       0.0625 |
| ('hsa_synthetic', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   5 |  -23.1953 |    13.3111 | 5/5    |  0.0176   |       0.0625 |
| ('hsa_synthetic', 'MAE-margin ↓', 'L_PCH + L_rank')                                 | L_PCH |   5 |  -24.8878 |    18.004  | 5/5    |  0.0365   |       0.0625 |
| ('sim_base', 'MAE-margin ↓', 'L_PCH + L_mul')                                       | L_PCH |   5 |    3.5644 |     8.0034 | 2/5    |  0.376    |       0.438  |
| ('sim_base', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul + MMV [λv=1]')                 | L_PCH |   5 |   18.3037 |     6.9443 | 0/5    |  0.00414  |       0.0625 |
| ('sim_base', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   5 |   21.5646 |     5.5466 | 0/5    |  0.000964 |       0.0625 |
| ('sim_base', 'MAE-margin ↓', 'L_PCH + L_rank')                                      | L_PCH |   5 |   20.9608 |     6.1033 | 0/5    |  0.00155  |       0.0625 |
