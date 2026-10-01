### Paired differences vs L_PCH (same seed = same split and init)

|                                                                                              | vs    |   n |    delta |   delta_sd | wins   |   p_ttest |   p_wilcoxon |
|:---------------------------------------------------------------------------------------------|:------|----:|---------:|-----------:|:-------|----------:|-------------:|
| ('ehrsim_base', 'bag', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                           | L_PCH |   3 |  -0.0041 |     0.0027 | 0/3    |  0.116    |         0.25 |
| ('ehrsim_base', 'sequence', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   3 |  -0.0026 |     0.0041 | 1/3    |  0.388    |         0.5  |
| ('ehrsim_base', 'static', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                        | L_PCH |   3 |  -0.0013 |     0.0033 | 1/3    |  0.556    |         1    |
| ('ehrsim_static', 'bag', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                         | L_PCH |   3 |  -0.0003 |     0.0018 | 1/3    |  0.809    |         0.75 |
| ('ehrsim_static', 'sequence', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                    | L_PCH |   3 |   0.0037 |     0.0061 | 2/3    |  0.4      |         0.5  |
| ('ehrsim_static', 'static', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   3 |   0.0024 |     0.0026 | 2/3    |  0.258    |         0.5  |
| ('ehrsim_temporal', 'bag', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                       | L_PCH |   3 |  -0.0018 |     0.0016 | 0/3    |  0.188    |         0.25 |
| ('ehrsim_temporal', 'sequence', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                  | L_PCH |   3 |  -0.0073 |     0.0056 | 0/3    |  0.153    |         0.25 |
| ('ehrsim_temporal', 'static', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                    | L_PCH |   3 |  -0.0021 |     0.0043 | 1/3    |  0.489    |         0.75 |
| ('ehrsim_base', 'bag', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')               | L_PCH |   3 |   0.0541 |     0.0192 | 3/3    |  0.0396   |         0.25 |
| ('ehrsim_base', 'sequence', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')          | L_PCH |   3 |   0.0685 |     0.0448 | 3/3    |  0.118    |         0.25 |
| ('ehrsim_base', 'static', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 |   0.1854 |     0.0507 | 3/3    |  0.024    |         0.25 |
| ('ehrsim_static', 'bag', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')             | L_PCH |   3 |   0.088  |     0.0266 | 3/3    |  0.0293   |         0.25 |
| ('ehrsim_static', 'sequence', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.0815 |     0.0331 | 3/3    |  0.0508   |         0.25 |
| ('ehrsim_static', 'static', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')          | L_PCH |   3 |   0.1602 |     0.0067 | 3/3    |  0.000588 |         0.25 |
| ('ehrsim_temporal', 'bag', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')           | L_PCH |   3 |   0.0717 |     0.0404 | 3/3    |  0.0913   |         0.25 |
| ('ehrsim_temporal', 'sequence', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.0758 |     0.0352 | 3/3    |  0.065    |         0.25 |
| ('ehrsim_temporal', 'static', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.1061 |     0.0583 | 3/3    |  0.0877   |         0.25 |
| ('ehrsim_base', 'bag', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')          | L_PCH |   3 |   0.0541 |     0.0192 | 3/3    |  0.0396   |         0.25 |
| ('ehrsim_base', 'sequence', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')     | L_PCH |   3 |   0.0685 |     0.0448 | 3/3    |  0.118    |         0.25 |
| ('ehrsim_base', 'static', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')       | L_PCH |   3 |   0.1854 |     0.0507 | 3/3    |  0.024    |         0.25 |
| ('ehrsim_static', 'bag', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.088  |     0.0266 | 3/3    |  0.0293   |         0.25 |
| ('ehrsim_static', 'sequence', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')   | L_PCH |   3 |   0.0815 |     0.0331 | 3/3    |  0.0508   |         0.25 |
| ('ehrsim_static', 'static', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')     | L_PCH |   3 |   0.1602 |     0.0067 | 3/3    |  0.000588 |         0.25 |
| ('ehrsim_temporal', 'bag', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.0717 |     0.0404 | 3/3    |  0.0913   |         0.25 |
| ('ehrsim_temporal', 'sequence', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)') | L_PCH |   3 |   0.0758 |     0.0352 | 3/3    |  0.065    |         0.25 |
| ('ehrsim_temporal', 'static', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')   | L_PCH |   3 |   0.1061 |     0.0583 | 3/3    |  0.0877   |         0.25 |
| ('ehrsim_base', 'bag', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                          | L_PCH |   3 |   0.0182 |     0.0037 | 0/3    |  0.0136   |         0.25 |
| ('ehrsim_base', 'sequence', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                     | L_PCH |   3 |   0.0127 |     0.0064 | 0/3    |  0.0753   |         0.25 |
| ('ehrsim_base', 'static', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                       | L_PCH |   3 |   0.0019 |     0.0008 | 0/3    |  0.0531   |         0.25 |
| ('ehrsim_static', 'bag', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                        | L_PCH |   3 |   0.0146 |     0.0029 | 0/3    |  0.013    |         0.25 |
| ('ehrsim_static', 'sequence', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                   | L_PCH |   3 |   0.0137 |     0.0008 | 0/3    |  0.00109  |         0.25 |
| ('ehrsim_static', 'static', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                     | L_PCH |   3 |   0.0012 |     0.0006 | 0/3    |  0.0792   |         0.25 |
| ('ehrsim_temporal', 'bag', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   3 |   0.0159 |     0.0023 | 0/3    |  0.00714  |         0.25 |
| ('ehrsim_temporal', 'sequence', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |   0.0135 |     0.0079 | 0/3    |  0.0981   |         0.25 |
| ('ehrsim_temporal', 'static', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                   | L_PCH |   3 |   0.0026 |     0.0007 | 0/3    |  0.0209   |         0.25 |
| ('ehrsim_base', 'bag', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                     | L_PCH |   3 |  89.1412 |    28.9548 | 0/3    |  0.0334   |         0.25 |
| ('ehrsim_base', 'sequence', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                | L_PCH |   3 |  52.0511 |    51.7388 | 1/3    |  0.224    |         0.5  |
| ('ehrsim_base', 'static', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                  | L_PCH |   3 |  -0.4026 |    25.4803 | 2/3    |  0.981    |         0.75 |
| ('ehrsim_static', 'bag', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                   | L_PCH |   3 |  57.896  |    53.977  | 0/3    |  0.204    |         0.25 |
| ('ehrsim_static', 'sequence', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')              | L_PCH |   3 |  63.7002 |    16.4451 | 0/3    |  0.0215   |         0.25 |
| ('ehrsim_static', 'static', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                | L_PCH |   3 | -15.7953 |    22.3926 | 2/3    |  0.346    |         0.5  |
| ('ehrsim_temporal', 'bag', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |  79.6083 |    18.389  | 0/3    |  0.0173   |         0.25 |
| ('ehrsim_temporal', 'sequence', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 |  57.2368 |    54.6872 | 0/3    |  0.212    |         0.25 |
| ('ehrsim_temporal', 'static', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')              | L_PCH |   3 |  -7.0404 |     4.9018 | 3/3    |  0.131    |         0.25 |
