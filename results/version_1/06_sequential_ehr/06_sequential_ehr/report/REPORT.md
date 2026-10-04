### ehrsim_base (mean ± sd over seeds)

| label                                     | C_td ↑        | Brier ↓       | Within-subject C ↑   | Within C, population KM   | Within C gain over KM ↑   | MAE-margin ↓      | MAE-uncens. ↓     |   n |
|:------------------------------------------|:--------------|:--------------|:---------------------|:--------------------------|:--------------------------|:------------------|:------------------|----:|
| Oracle (true CIF)                         | 0.723 ± 0.005 | 0.155 ± 0.001 | 0.681 ± 0.004        | 0.581 ± 0.004             | 0.100 ± 0.001             | 1299.147 ± 3.106  | 1398.968 ± 9.029  |   3 |
| Cox PH (cause-specific) [bag]             | 0.708 ± 0.008 | 0.156 ± 0.002 | 0.691 ± 0.010        | 0.581 ± 0.004             | 0.110 ± 0.007             | 1029.157 ± 8.367  | 889.446 ± 18.331  |   3 |
| L_PCH [bag]                               | 0.705 ± 0.011 | 0.155 ± 0.002 | 0.707 ± 0.012        | 0.581 ± 0.004             | 0.126 ± 0.008             | 1034.506 ± 13.059 | 887.732 ± 34.886  |   3 |
| L_PCH + L_rank + L_mul (paper) [bag]      | 0.701 ± 0.009 | 0.174 ± 0.002 | 0.761 ± 0.022        | 0.581 ± 0.004             | 0.180 ± 0.020             | 1123.647 ± 17.023 | 922.900 ± 33.423  |   3 |
| DeepHit [sequence]                        | 0.697 ± 0.012 | 0.223 ± 0.006 | 0.421 ± 0.015        | 0.581 ± 0.004             | -0.160 ± 0.016            | 2050.453 ± 50.903 | 2437.054 ± 69.504 |   3 |
| MENSA [sequence]                          | 0.690 ± 0.012 | 0.173 ± 0.014 | 0.596 ± 0.035        | 0.581 ± 0.004             | 0.015 ± 0.038             | 1093.548 ± 57.430 | 935.929 ± 141.916 |   3 |
| L_PCH [sequence]                          | 0.700 ± 0.011 | 0.161 ± 0.003 | 0.702 ± 0.040        | 0.581 ± 0.004             | 0.121 ± 0.038             | 1084.532 ± 22.110 | 984.336 ± 49.508  |   3 |
| L_PCH + L_rank + L_mul (paper) [sequence] | 0.698 ± 0.007 | 0.173 ± 0.004 | 0.771 ± 0.014        | 0.581 ± 0.004             | 0.190 ± 0.012             | 1136.583 ± 29.656 | 958.754 ± 38.002  |   3 |
| L_PCH [static]                            | 0.500 ± 0.002 | 0.194 ± 0.001 | 0.536 ± 0.050        | 0.581 ± 0.004             | -0.045 ± 0.048            | 1237.132 ± 19.505 | 1113.635 ± 41.222 |   3 |
| L_PCH + L_rank + L_mul (paper) [static]   | 0.499 ± 0.002 | 0.196 ± 0.002 | 0.721 ± 0.006        | 0.581 ± 0.004             | 0.141 ± 0.008             | 1236.729 ± 20.039 | 1073.664 ± 44.447 |   3 |

### ehrsim_static (mean ± sd over seeds)

| label                                     | C_td ↑        | Brier ↓       | Within-subject C ↑   | Within C, population KM   | Within C gain over KM ↑   | MAE-margin ↓      | MAE-uncens. ↓     |   n |
|:------------------------------------------|:--------------|:--------------|:---------------------|:--------------------------|:--------------------------|:------------------|:------------------|----:|
| Oracle (true CIF)                         | 0.735 ± 0.003 | 0.150 ± 0.000 | 0.688 ± 0.003        | 0.580 ± 0.005             | 0.108 ± 0.003             | 1348.632 ± 12.603 | 1445.988 ± 18.176 |   3 |
| Cox PH (cause-specific) [bag]             | 0.725 ± 0.004 | 0.151 ± 0.001 | 0.703 ± 0.008        | 0.580 ± 0.005             | 0.123 ± 0.009             | 1040.460 ± 7.244  | 887.338 ± 13.146  |   3 |
| L_PCH [bag]                               | 0.710 ± 0.003 | 0.157 ± 0.001 | 0.680 ± 0.041        | 0.580 ± 0.005             | 0.100 ± 0.037             | 1102.598 ± 31.243 | 977.179 ± 57.310  |   3 |
| L_PCH + L_rank + L_mul (paper) [bag]      | 0.710 ± 0.004 | 0.172 ± 0.002 | 0.768 ± 0.020        | 0.580 ± 0.005             | 0.188 ± 0.015             | 1160.494 ± 23.491 | 934.106 ± 64.482  |   3 |
| DeepHit [sequence]                        | 0.712 ± 0.007 | 0.216 ± 0.003 | 0.454 ± 0.015        | 0.580 ± 0.005             | -0.126 ± 0.012            | 2118.000 ± 15.222 | 2509.363 ± 16.415 |   3 |
| MENSA [sequence]                          | 0.712 ± 0.008 | 0.158 ± 0.005 | 0.662 ± 0.042        | 0.580 ± 0.005             | 0.082 ± 0.046             | 1067.742 ± 28.869 | 879.447 ± 61.403  |   3 |
| L_PCH [sequence]                          | 0.708 ± 0.005 | 0.156 ± 0.001 | 0.693 ± 0.023        | 0.580 ± 0.005             | 0.113 ± 0.024             | 1081.013 ± 10.571 | 935.579 ± 14.023  |   3 |
| L_PCH + L_rank + L_mul (paper) [sequence] | 0.712 ± 0.008 | 0.170 ± 0.001 | 0.775 ± 0.011        | 0.580 ± 0.005             | 0.195 ± 0.009             | 1144.713 ± 20.192 | 910.873 ± 56.853  |   3 |
| L_PCH [static]                            | 0.502 ± 0.004 | 0.193 ± 0.002 | 0.563 ± 0.008        | 0.580 ± 0.005             | -0.017 ± 0.005            | 1290.426 ± 17.810 | 1124.666 ± 35.667 |   3 |
| L_PCH + L_rank + L_mul (paper) [static]   | 0.504 ± 0.002 | 0.194 ± 0.001 | 0.723 ± 0.002        | 0.580 ± 0.005             | 0.143 ± 0.003             | 1274.631 ± 6.847  | 1053.022 ± 25.130 |   3 |

### ehrsim_temporal (mean ± sd over seeds)

| label                                     | C_td ↑        | Brier ↓       | Within-subject C ↑   | Within C, population KM   | Within C gain over KM ↑   | MAE-margin ↓      | MAE-uncens. ↓     |   n |
|:------------------------------------------|:--------------|:--------------|:---------------------|:--------------------------|:--------------------------|:------------------|:------------------|----:|
| Oracle (true CIF)                         | 0.708 ± 0.003 | 0.155 ± 0.001 | 0.677 ± 0.003        | 0.595 ± 0.004             | 0.082 ± 0.007             | 1256.318 ± 1.342  | 1357.960 ± 7.851  |   3 |
| Cox PH (cause-specific) [bag]             | 0.688 ± 0.009 | 0.162 ± 0.002 | 0.680 ± 0.009        | 0.595 ± 0.004             | 0.086 ± 0.012             | 1036.622 ± 7.462  | 907.816 ± 16.121  |   3 |
| L_PCH [bag]                               | 0.686 ± 0.007 | 0.162 ± 0.002 | 0.695 ± 0.040        | 0.595 ± 0.004             | 0.101 ± 0.037             | 1045.387 ± 14.635 | 918.684 ± 55.590  |   3 |
| L_PCH + L_rank + L_mul (paper) [bag]      | 0.684 ± 0.007 | 0.178 ± 0.004 | 0.767 ± 0.003        | 0.595 ± 0.004             | 0.173 ± 0.006             | 1124.996 ± 23.413 | 979.352 ± 52.164  |   3 |
| DeepHit [sequence]                        | 0.681 ± 0.006 | 0.223 ± 0.006 | 0.439 ± 0.009        | 0.595 ± 0.004             | -0.155 ± 0.008            | 1957.399 ± 26.075 | 2345.690 ± 33.890 |   3 |
| MENSA [sequence]                          | 0.675 ± 0.009 | 0.169 ± 0.007 | 0.702 ± 0.041        | 0.595 ± 0.004             | 0.107 ± 0.044             | 1063.609 ± 56.703 | 931.532 ± 108.226 |   3 |
| L_PCH [sequence]                          | 0.684 ± 0.009 | 0.167 ± 0.005 | 0.692 ± 0.032        | 0.595 ± 0.004             | 0.098 ± 0.030             | 1081.203 ± 31.431 | 990.749 ± 67.074  |   3 |
| L_PCH + L_rank + L_mul (paper) [sequence] | 0.677 ± 0.004 | 0.180 ± 0.005 | 0.768 ± 0.004        | 0.595 ± 0.004             | 0.173 ± 0.005             | 1138.440 ± 26.938 | 995.466 ± 46.342  |   3 |
| L_PCH [static]                            | 0.500 ± 0.001 | 0.193 ± 0.001 | 0.622 ± 0.059        | 0.595 ± 0.004             | 0.027 ± 0.060             | 1190.659 ± 12.586 | 1064.975 ± 18.290 |   3 |
| L_PCH + L_rank + L_mul (paper) [static]   | 0.498 ± 0.004 | 0.196 ± 0.002 | 0.728 ± 0.012        | 0.595 ± 0.004             | 0.133 ± 0.008             | 1183.618 ± 16.969 | 1018.140 ± 33.281 |   3 |


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


### Sequence / static vs bag-of-codes input (paired on seed)

|                                                                                         | vs   |   n |   delta |   delta_sd | wins   |   p_ttest |   p_wilcoxon |
|:----------------------------------------------------------------------------------------|:-----|----:|--------:|-----------:|:-------|----------:|-------------:|
| ('ehrsim_base', 'L_PCH', 'C_td ↑', 'sequence')                                          | bag  |   3 | -0.0051 |     0.0001 | 0/3    |  9.13e-05 |         0.25 |
| ('ehrsim_base', 'L_PCH', 'C_td ↑', 'static')                                            | bag  |   3 | -0.2053 |     0.0126 | 0/3    |  0.00126  |         0.25 |
| ('ehrsim_base', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'sequence')                 | bag  |   3 | -0.0036 |     0.0029 | 0/3    |  0.164    |         0.25 |
| ('ehrsim_base', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'static')                   | bag  |   3 | -0.2025 |     0.0074 | 0/3    |  0.00045  |         0.25 |
| ('ehrsim_static', 'L_PCH', 'C_td ↑', 'sequence')                                        | bag  |   3 | -0.0019 |     0.0034 | 1/3    |  0.43     |         0.5  |
| ('ehrsim_static', 'L_PCH', 'C_td ↑', 'static')                                          | bag  |   3 | -0.2086 |     0.0065 | 0/3    |  0.000328 |         0.25 |
| ('ehrsim_static', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'sequence')               | bag  |   3 |  0.0021 |     0.005  | 2/3    |  0.544    |         0.75 |
| ('ehrsim_static', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'static')                 | bag  |   3 | -0.2059 |     0.0055 | 0/3    |  0.000239 |         0.25 |
| ('ehrsim_temporal', 'L_PCH', 'C_td ↑', 'sequence')                                      | bag  |   3 | -0.002  |     0.0017 | 0/3    |  0.176    |         0.25 |
| ('ehrsim_temporal', 'L_PCH', 'C_td ↑', 'static')                                        | bag  |   3 | -0.1861 |     0.0069 | 0/3    |  0.000453 |         0.25 |
| ('ehrsim_temporal', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'sequence')             | bag  |   3 | -0.0075 |     0.0036 | 0/3    |  0.068    |         0.25 |
| ('ehrsim_temporal', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'static')               | bag  |   3 | -0.1864 |     0.0107 | 0/3    |  0.00111  |         0.25 |
| ('ehrsim_base', 'L_PCH', 'Within-subject C ↑', 'sequence')                              | bag  |   3 | -0.0048 |     0.0353 | 2/3    |  0.836    |         1    |
| ('ehrsim_base', 'L_PCH', 'Within-subject C ↑', 'static')                                | bag  |   3 | -0.171  |     0.0426 | 0/3    |  0.0201   |         0.25 |
| ('ehrsim_base', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'sequence')     | bag  |   3 |  0.0096 |     0.0286 | 2/3    |  0.618    |         0.75 |
| ('ehrsim_base', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'static')       | bag  |   3 | -0.0397 |     0.0212 | 0/3    |  0.0836   |         0.25 |
| ('ehrsim_static', 'L_PCH', 'Within-subject C ↑', 'sequence')                            | bag  |   3 |  0.0135 |     0.0457 | 2/3    |  0.659    |         0.75 |
| ('ehrsim_static', 'L_PCH', 'Within-subject C ↑', 'static')                              | bag  |   3 | -0.1173 |     0.033  | 0/3    |  0.0254   |         0.25 |
| ('ehrsim_static', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'sequence')   | bag  |   3 |  0.0071 |     0.0119 | 2/3    |  0.411    |         0.5  |
| ('ehrsim_static', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'static')     | bag  |   3 | -0.045  |     0.0178 | 0/3    |  0.0484   |         0.25 |
| ('ehrsim_temporal', 'L_PCH', 'Within-subject C ↑', 'sequence')                          | bag  |   3 | -0.0033 |     0.0156 | 1/3    |  0.749    |         0.75 |
| ('ehrsim_temporal', 'L_PCH', 'Within-subject C ↑', 'static')                            | bag  |   3 | -0.0737 |     0.091  | 0/3    |  0.296    |         0.25 |
| ('ehrsim_temporal', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'sequence') | bag  |   3 |  0.0007 |     0.0074 | 1/3    |  0.879    |         1    |
| ('ehrsim_temporal', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'static')   | bag  |   3 | -0.0394 |     0.0142 | 0/3    |  0.0405   |         0.25 |
| ('ehrsim_base', 'L_PCH', 'Brier ↓', 'sequence')                                         | bag  |   3 |  0.0053 |     0.0013 | 0/3    |  0.0191   |         0.25 |
| ('ehrsim_base', 'L_PCH', 'Brier ↓', 'static')                                           | bag  |   3 |  0.0382 |     0.003  | 0/3    |  0.00203  |         0.25 |
| ('ehrsim_base', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'sequence')                | bag  |   3 | -0.0002 |     0.0032 | 1/3    |  0.909    |         1    |
| ('ehrsim_base', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'static')                  | bag  |   3 |  0.0219 |     0.0001 | 0/3    |  9.33e-06 |         0.25 |
| ('ehrsim_static', 'L_PCH', 'Brier ↓', 'sequence')                                       | bag  |   3 | -0.001  |     0.0008 | 3/3    |  0.162    |         0.25 |
| ('ehrsim_static', 'L_PCH', 'Brier ↓', 'static')                                         | bag  |   3 |  0.0357 |     0.0023 | 0/3    |  0.00135  |         0.25 |
| ('ehrsim_static', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'sequence')              | bag  |   3 | -0.0019 |     0.003  | 2/3    |  0.395    |         0.5  |
| ('ehrsim_static', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'static')                | bag  |   3 |  0.0223 |     0.0012 | 0/3    |  0.000966 |         0.25 |
| ('ehrsim_temporal', 'L_PCH', 'Brier ↓', 'sequence')                                     | bag  |   3 |  0.0051 |     0.0033 | 0/3    |  0.114    |         0.25 |
| ('ehrsim_temporal', 'L_PCH', 'Brier ↓', 'static')                                       | bag  |   3 |  0.0314 |     0.0034 | 0/3    |  0.00385  |         0.25 |
| ('ehrsim_temporal', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'sequence')            | bag  |   3 |  0.0028 |     0.0036 | 1/3    |  0.316    |         0.5  |
| ('ehrsim_temporal', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'static')              | bag  |   3 |  0.0181 |     0.005  | 0/3    |  0.0244   |         0.25 |
