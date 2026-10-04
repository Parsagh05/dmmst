### ehrsim2_level (mean ± sd over seeds)

| label                                     | C_td ↑        | Brier ↓       | Within-subject C ↑   | Within C, population KM   | Within C gain over KM ↑   | MAE-margin ↓      | MAE-uncens. ↓     | True-order C ↑   | True-order Brier ↓   | True-order log-loss ↓   |   n |
|:------------------------------------------|:--------------|:--------------|:---------------------|:--------------------------|:--------------------------|:------------------|:------------------|:-----------------|:---------------------|:------------------------|----:|
| Oracle (true CIF)                         | 0.745 ± 0.002 | 0.150 ± 0.002 | 0.688 ± 0.001        | 0.576 ± 0.006             | 0.112 ± 0.007             | 1329.158 ± 9.562  | 1429.560 ± 11.536 | 0.669 ± 0.002    | 0.209 ± 0.000        | 0.603 ± 0.001           |   3 |
| Cox PH (cause-specific) [bag]             | 0.735 ± 0.003 | 0.150 ± 0.002 | 0.712 ± 0.004        | 0.576 ± 0.006             | 0.136 ± 0.004             | 1027.552 ± 2.191  | 902.750 ± 2.082   | 0.663 ± 0.005    | 0.210 ± 0.001        | 0.606 ± 0.001           |   3 |
| L_PCH [bag]                               | 0.716 ± 0.005 | 0.157 ± 0.005 | 0.661 ± 0.042        | 0.576 ± 0.006             | 0.084 ± 0.037             | 1082.830 ± 36.097 | 973.816 ± 73.161  | 0.636 ± 0.006    | 0.217 ± 0.001        | 0.622 ± 0.002           |   3 |
| L_PCH + L_rank + L_mul (paper) [bag]      | 0.717 ± 0.005 | 0.173 ± 0.002 | 0.797 ± 0.009        | 0.576 ± 0.006             | 0.220 ± 0.003             | 1173.920 ± 2.639  | 980.556 ± 11.053  | 0.595 ± 0.008    | 0.232 ± 0.001        | 0.655 ± 0.002           |   3 |
| DeepHit [sequence]                        | 0.721 ± 0.002 | 0.220 ± 0.005 | 0.470 ± 0.009        | 0.576 ± 0.006             | -0.106 ± 0.012            | 2204.750 ± 62.977 | 2609.158 ± 68.951 | 0.568 ± 0.004    | 0.277 ± 0.003        | 0.757 ± 0.008           |   3 |
| MENSA [sequence]                          | 0.721 ± 0.004 | 0.158 ± 0.005 | 0.609 ± 0.022        | 0.576 ± 0.006             | 0.033 ± 0.016             | 1109.353 ± 35.170 | 997.832 ± 43.911  | 0.633 ± 0.020    | 0.222 ± 0.008        | 0.633 ± 0.018           |   3 |
| L_PCH [sequence]                          | 0.726 ± 0.005 | 0.155 ± 0.002 | 0.686 ± 0.018        | 0.576 ± 0.006             | 0.110 ± 0.024             | 1069.613 ± 12.250 | 962.009 ± 24.285  | 0.646 ± 0.004    | 0.216 ± 0.001        | 0.621 ± 0.003           |   3 |
| L_PCH + L_rank + L_mul (paper) [sequence] | 0.724 ± 0.004 | 0.173 ± 0.001 | 0.755 ± 0.018        | 0.576 ± 0.006             | 0.179 ± 0.019             | 1165.794 ± 5.433  | 978.730 ± 11.670  | 0.619 ± 0.006    | 0.231 ± 0.000        | 0.654 ± 0.001           |   3 |
| L_PCH [static]                            | 0.502 ± 0.007 | 0.193 ± 0.001 | 0.588 ± 0.025        | 0.576 ± 0.006             | 0.011 ± 0.020             | 1296.820 ± 20.415 | 1147.965 ± 40.488 | 0.493 ± 0.012    | 0.250 ± 0.000        | 0.694 ± 0.000           |   3 |
| L_PCH + L_rank + L_mul (paper) [static]   | 0.500 ± 0.002 | 0.196 ± 0.001 | 0.727 ± 0.011        | 0.576 ± 0.006             | 0.151 ± 0.006             | 1283.863 ± 18.702 | 1079.339 ± 26.801 | 0.499 ± 0.007    | 0.252 ± 0.001        | 0.698 ± 0.001           |   3 |

### ehrsim2_mixed (mean ± sd over seeds)

| label                                     | C_td ↑        | Brier ↓       | Within-subject C ↑   | Within C, population KM   | Within C gain over KM ↑   | MAE-margin ↓      | MAE-uncens. ↓     | True-order C ↑   | True-order Brier ↓   | True-order log-loss ↓   |   n |
|:------------------------------------------|:--------------|:--------------|:---------------------|:--------------------------|:--------------------------|:------------------|:------------------|:-----------------|:---------------------|:------------------------|----:|
| Oracle (true CIF)                         | 0.725 ± 0.004 | 0.156 ± 0.001 | 0.680 ± 0.003        | 0.576 ± 0.012             | 0.104 ± 0.014             | 1266.857 ± 13.423 | 1361.667 ± 16.489 | 0.634 ± 0.001    | 0.218 ± 0.002        | 0.621 ± 0.004           |   3 |
| Cox PH (cause-specific) [bag]             | 0.697 ± 0.002 | 0.164 ± 0.001 | 0.688 ± 0.009        | 0.576 ± 0.012             | 0.112 ± 0.005             | 1028.732 ± 3.941  | 904.632 ± 2.926   | 0.622 ± 0.008    | 0.226 ± 0.001        | 0.642 ± 0.002           |   3 |
| L_PCH [bag]                               | 0.691 ± 0.009 | 0.165 ± 0.002 | 0.667 ± 0.052        | 0.576 ± 0.012             | 0.091 ± 0.061             | 1067.202 ± 20.658 | 966.591 ± 46.058  | 0.601 ± 0.012    | 0.229 ± 0.002        | 0.648 ± 0.004           |   3 |
| L_PCH + L_rank + L_mul (paper) [bag]      | 0.689 ± 0.009 | 0.182 ± 0.001 | 0.775 ± 0.013        | 0.576 ± 0.012             | 0.199 ± 0.003             | 1135.816 ± 22.268 | 968.915 ± 45.249  | 0.574 ± 0.003    | 0.240 ± 0.001        | 0.673 ± 0.002           |   3 |
| DeepHit [sequence]                        | 0.682 ± 0.000 | 0.224 ± 0.005 | 0.437 ± 0.028        | 0.576 ± 0.012             | -0.139 ± 0.036            | 2009.014 ± 40.584 | 2378.578 ± 46.197 | 0.542 ± 0.011    | 0.280 ± 0.002        | 0.763 ± 0.004           |   3 |
| MENSA [sequence]                          | 0.681 ± 0.016 | 0.178 ± 0.012 | 0.609 ± 0.051        | 0.576 ± 0.012             | 0.033 ± 0.056             | 1116.249 ± 63.097 | 990.365 ± 132.125 | 0.580 ± 0.033    | 0.238 ± 0.010        | 0.669 ± 0.024           |   3 |
| L_PCH [sequence]                          | 0.687 ± 0.009 | 0.170 ± 0.003 | 0.705 ± 0.019        | 0.576 ± 0.012             | 0.129 ± 0.029             | 1071.702 ± 22.522 | 975.124 ± 32.792  | 0.613 ± 0.005    | 0.229 ± 0.002        | 0.651 ± 0.004           |   3 |
| L_PCH + L_rank + L_mul (paper) [sequence] | 0.688 ± 0.005 | 0.180 ± 0.001 | 0.776 ± 0.009        | 0.576 ± 0.012             | 0.200 ± 0.010             | 1121.795 ± 14.604 | 945.703 ± 28.776  | 0.572 ± 0.004    | 0.241 ± 0.002        | 0.675 ± 0.004           |   3 |
| L_PCH [static]                            | 0.498 ± 0.006 | 0.194 ± 0.001 | 0.566 ± 0.062        | 0.576 ± 0.012             | -0.010 ± 0.052            | 1216.372 ± 16.212 | 1107.030 ± 32.045 | 0.500 ± 0.005    | 0.251 ± 0.000        | 0.694 ± 0.000           |   3 |
| L_PCH + L_rank + L_mul (paper) [static]   | 0.496 ± 0.001 | 0.196 ± 0.001 | 0.728 ± 0.009        | 0.576 ± 0.012             | 0.152 ± 0.005             | 1194.146 ± 19.682 | 1020.516 ± 46.065 | 0.508 ± 0.007    | 0.251 ± 0.000        | 0.696 ± 0.001           |   3 |

### ehrsim2_order (mean ± sd over seeds)

| label                                     | C_td ↑        | Brier ↓       | Within-subject C ↑   | Within C, population KM   | Within C gain over KM ↑   | MAE-margin ↓      | MAE-uncens. ↓     | True-order C ↑   | True-order Brier ↓   | True-order log-loss ↓   |   n |
|:------------------------------------------|:--------------|:--------------|:---------------------|:--------------------------|:--------------------------|:------------------|:------------------|:-----------------|:---------------------|:------------------------|----:|
| Oracle (true CIF)                         | 0.696 ± 0.001 | 0.161 ± 0.001 | 0.681 ± 0.004        | 0.631 ± 0.010             | 0.049 ± 0.006             | 1217.319 ± 12.896 | 1301.425 ± 18.070 | 0.603 ± 0.003    | 0.223 ± 0.001        | 0.631 ± 0.003           |   3 |
| Cox PH (cause-specific) [bag]             | 0.653 ± 0.003 | 0.175 ± 0.002 | 0.679 ± 0.014        | 0.631 ± 0.010             | 0.048 ± 0.005             | 1030.558 ± 5.047  | 913.879 ± 4.290   | 0.593 ± 0.009    | 0.235 ± 0.002        | 0.662 ± 0.003           |   3 |
| L_PCH [bag]                               | 0.655 ± 0.008 | 0.174 ± 0.001 | 0.690 ± 0.034        | 0.631 ± 0.010             | 0.059 ± 0.026             | 1047.092 ± 6.869  | 951.677 ± 22.648  | 0.600 ± 0.003    | 0.233 ± 0.001        | 0.659 ± 0.003           |   3 |
| L_PCH + L_rank + L_mul (paper) [bag]      | 0.649 ± 0.001 | 0.185 ± 0.001 | 0.771 ± 0.007        | 0.631 ± 0.010             | 0.140 ± 0.009             | 1098.233 ± 9.736  | 954.898 ± 9.697   | 0.563 ± 0.015    | 0.244 ± 0.002        | 0.682 ± 0.004           |   3 |
| DeepHit [sequence]                        | 0.640 ± 0.001 | 0.229 ± 0.007 | 0.411 ± 0.028        | 0.631 ± 0.010             | -0.220 ± 0.018            | 1929.737 ± 71.226 | 2279.106 ± 82.926 | 0.527 ± 0.005    | 0.282 ± 0.006        | 0.767 ± 0.016           |   3 |
| MENSA [sequence]                          | 0.645 ± 0.003 | 0.182 ± 0.006 | 0.704 ± 0.024        | 0.631 ± 0.010             | 0.073 ± 0.028             | 1067.218 ± 16.984 | 952.599 ± 21.811  | 0.584 ± 0.015    | 0.237 ± 0.003        | 0.667 ± 0.006           |   3 |
| L_PCH [sequence]                          | 0.647 ± 0.005 | 0.176 ± 0.001 | 0.695 ± 0.033        | 0.631 ± 0.010             | 0.063 ± 0.031             | 1052.996 ± 26.419 | 948.504 ± 57.577  | 0.586 ± 0.010    | 0.237 ± 0.003        | 0.667 ± 0.006           |   3 |
| L_PCH + L_rank + L_mul (paper) [sequence] | 0.644 ± 0.003 | 0.184 ± 0.001 | 0.772 ± 0.008        | 0.631 ± 0.010             | 0.141 ± 0.003             | 1084.587 ± 13.985 | 936.564 ± 18.668  | 0.569 ± 0.003    | 0.243 ± 0.001        | 0.680 ± 0.003           |   3 |
| L_PCH [static]                            | 0.502 ± 0.006 | 0.195 ± 0.001 | 0.631 ± 0.002        | 0.631 ± 0.010             | 0.000 ± 0.012             | 1159.697 ± 6.526  | 1079.144 ± 14.095 | 0.503 ± 0.014    | 0.250 ± 0.001        | 0.693 ± 0.002           |   3 |
| L_PCH + L_rank + L_mul (paper) [static]   | 0.503 ± 0.005 | 0.196 ± 0.001 | 0.726 ± 0.006        | 0.631 ± 0.010             | 0.095 ± 0.011             | 1134.429 ± 10.056 | 991.949 ± 22.298  | 0.515 ± 0.004    | 0.251 ± 0.000        | 0.696 ± 0.001           |   3 |


### Paired differences vs L_PCH (same seed = same split and init)

|                                                                                            | vs    |   n |    delta |   delta_sd | wins   |   p_ttest |   p_wilcoxon |
|:-------------------------------------------------------------------------------------------|:------|----:|---------:|-----------:|:-------|----------:|-------------:|
| ('ehrsim2_level', 'bag', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                       | L_PCH |   3 |   0.0009 |     0.0004 | 3/3    |  0.0603   |         0.25 |
| ('ehrsim2_level', 'sequence', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                  | L_PCH |   3 |  -0.0022 |     0.0047 | 1/3    |  0.512    |         0.75 |
| ('ehrsim2_level', 'static', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                    | L_PCH |   3 |  -0.0028 |     0.0048 | 0/3    |  0.423    |         1    |
| ('ehrsim2_mixed', 'bag', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                       | L_PCH |   3 |  -0.0021 |     0.0004 | 0/3    |  0.0123   |         0.25 |
| ('ehrsim2_mixed', 'sequence', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                  | L_PCH |   3 |   0.0009 |     0.0043 | 2/3    |  0.754    |         0.75 |
| ('ehrsim2_mixed', 'static', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                    | L_PCH |   3 |  -0.0021 |     0.0061 | 1/3    |  0.615    |         1    |
| ('ehrsim2_order', 'bag', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                       | L_PCH |   3 |  -0.0063 |     0.0078 | 1/3    |  0.295    |         0.5  |
| ('ehrsim2_order', 'sequence', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                  | L_PCH |   3 |  -0.0028 |     0.0078 | 2/3    |  0.598    |         1    |
| ('ehrsim2_order', 'static', 'C_td ↑', 'L_PCH + L_rank + L_mul (paper)')                    | L_PCH |   3 |   0.0002 |     0.002  | 1/3    |  0.899    |         1    |
| ('ehrsim2_level', 'bag', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')           | L_PCH |   3 |   0.136  |     0.036  | 3/3    |  0.0225   |         0.25 |
| ('ehrsim2_level', 'sequence', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.069  |     0.0218 | 3/3    |  0.0318   |         0.25 |
| ('ehrsim2_level', 'static', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.1398 |     0.0143 | 3/3    |  0.00349  |         0.25 |
| ('ehrsim2_mixed', 'bag', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')           | L_PCH |   3 |   0.1083 |     0.0602 | 3/3    |  0.0894   |         0.25 |
| ('ehrsim2_mixed', 'sequence', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.071  |     0.0198 | 3/3    |  0.025    |         0.25 |
| ('ehrsim2_mixed', 'static', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.1615 |     0.0568 | 3/3    |  0.0388   |         0.25 |
| ('ehrsim2_order', 'bag', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')           | L_PCH |   3 |   0.0808 |     0.028  | 3/3    |  0.0377   |         0.25 |
| ('ehrsim2_order', 'sequence', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.0777 |     0.0335 | 3/3    |  0.0567   |         0.25 |
| ('ehrsim2_order', 'static', 'Within-subject C ↑', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.0946 |     0.0065 | 3/3    |  0.00157  |         0.25 |
| ('ehrsim2_level', 'bag', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.136  |     0.036  | 3/3    |  0.0225   |         0.25 |
| ('ehrsim2_level', 'sequence', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)') | L_PCH |   3 |   0.069  |     0.0218 | 3/3    |  0.0318   |         0.25 |
| ('ehrsim2_level', 'static', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')   | L_PCH |   3 |   0.1398 |     0.0143 | 3/3    |  0.00349  |         0.25 |
| ('ehrsim2_mixed', 'bag', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.1083 |     0.0602 | 3/3    |  0.0894   |         0.25 |
| ('ehrsim2_mixed', 'sequence', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)') | L_PCH |   3 |   0.071  |     0.0198 | 3/3    |  0.025    |         0.25 |
| ('ehrsim2_mixed', 'static', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')   | L_PCH |   3 |   0.1615 |     0.0568 | 3/3    |  0.0388   |         0.25 |
| ('ehrsim2_order', 'bag', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.0808 |     0.028  | 3/3    |  0.0377   |         0.25 |
| ('ehrsim2_order', 'sequence', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)') | L_PCH |   3 |   0.0777 |     0.0335 | 3/3    |  0.0567   |         0.25 |
| ('ehrsim2_order', 'static', 'Within C gain over KM ↑', 'L_PCH + L_rank + L_mul (paper)')   | L_PCH |   3 |   0.0946 |     0.0065 | 3/3    |  0.00157  |         0.25 |
| ('ehrsim2_level', 'bag', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')               | L_PCH |   3 |  -0.041  |     0.0126 | 0/3    |  0.0301   |         0.25 |
| ('ehrsim2_level', 'sequence', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')          | L_PCH |   3 |  -0.0268 |     0.0085 | 0/3    |  0.0318   |         0.25 |
| ('ehrsim2_level', 'static', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 |   0.0059 |     0.0111 | 2/3    |  0.453    |         0.5  |
| ('ehrsim2_mixed', 'bag', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')               | L_PCH |   3 |  -0.0276 |     0.0092 | 0/3    |  0.0353   |         0.25 |
| ('ehrsim2_mixed', 'sequence', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')          | L_PCH |   3 |  -0.041  |     0.0016 | 0/3    |  0.000527 |         0.25 |
| ('ehrsim2_mixed', 'static', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 |   0.0081 |     0.0103 | 2/3    |  0.305    |         0.5  |
| ('ehrsim2_order', 'bag', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')               | L_PCH |   3 |  -0.0361 |     0.0175 | 0/3    |  0.0702   |         0.25 |
| ('ehrsim2_order', 'sequence', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')          | L_PCH |   3 |  -0.0175 |     0.0091 | 0/3    |  0.08     |         0.25 |
| ('ehrsim2_order', 'static', 'True-order C ↑', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 |   0.0115 |     0.0127 | 3/3    |  0.259    |         0.25 |
| ('ehrsim2_level', 'bag', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')           | L_PCH |   3 |   0.0146 |     0.0018 | 0/3    |  0.00486  |         0.25 |
| ('ehrsim2_level', 'sequence', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.0149 |     0.0006 | 0/3    |  0.000517 |         0.25 |
| ('ehrsim2_level', 'static', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.0021 |     0.0006 | 0/3    |  0.0284   |         0.25 |
| ('ehrsim2_mixed', 'bag', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')           | L_PCH |   3 |   0.0115 |     0.0012 | 0/3    |  0.00368  |         0.25 |
| ('ehrsim2_mixed', 'sequence', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.0115 |     0.0006 | 0/3    |  0.000849 |         0.25 |
| ('ehrsim2_mixed', 'static', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.0008 |     0.0006 | 0/3    |  0.15     |         0.25 |
| ('ehrsim2_order', 'bag', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')           | L_PCH |   3 |   0.011  |     0.0025 | 0/3    |  0.0169   |         0.25 |
| ('ehrsim2_order', 'sequence', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')      | L_PCH |   3 |   0.0064 |     0.0017 | 0/3    |  0.0236   |         0.25 |
| ('ehrsim2_order', 'static', 'True-order Brier ↓', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.0011 |     0.0005 | 0/3    |  0.056    |         0.25 |
| ('ehrsim2_level', 'bag', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.0326 |     0.0044 | 0/3    |  0.0061   |         0.25 |
| ('ehrsim2_level', 'sequence', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')   | L_PCH |   3 |   0.0329 |     0.0022 | 0/3    |  0.00149  |         0.25 |
| ('ehrsim2_level', 'static', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')     | L_PCH |   3 |   0.0043 |     0.0013 | 0/3    |  0.0285   |         0.25 |
| ('ehrsim2_mixed', 'bag', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.0253 |     0.0022 | 0/3    |  0.00248  |         0.25 |
| ('ehrsim2_mixed', 'sequence', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')   | L_PCH |   3 |   0.0239 |     0.001  | 0/3    |  0.000562 |         0.25 |
| ('ehrsim2_mixed', 'static', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')     | L_PCH |   3 |   0.0016 |     0.0012 | 0/3    |  0.147    |         0.25 |
| ('ehrsim2_order', 'bag', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')        | L_PCH |   3 |   0.0232 |     0.0052 | 0/3    |  0.0166   |         0.25 |
| ('ehrsim2_order', 'sequence', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')   | L_PCH |   3 |   0.0132 |     0.0039 | 0/3    |  0.0278   |         0.25 |
| ('ehrsim2_order', 'static', 'True-order log-loss ↓', 'L_PCH + L_rank + L_mul (paper)')     | L_PCH |   3 |   0.0023 |     0.001  | 0/3    |  0.0539   |         0.25 |
| ('ehrsim2_level', 'bag', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   3 |   0.0163 |     0.0043 | 0/3    |  0.0229   |         0.25 |
| ('ehrsim2_level', 'sequence', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |   0.0175 |     0.0023 | 0/3    |  0.00581  |         0.25 |
| ('ehrsim2_level', 'static', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                   | L_PCH |   3 |   0.0032 |     0.0004 | 0/3    |  0.00468  |         0.25 |
| ('ehrsim2_mixed', 'bag', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   3 |   0.0167 |     0.0011 | 0/3    |  0.00145  |         0.25 |
| ('ehrsim2_mixed', 'sequence', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |   0.0101 |     0.0019 | 0/3    |  0.0113   |         0.25 |
| ('ehrsim2_mixed', 'static', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                   | L_PCH |   3 |   0.002  |     0.0011 | 0/3    |  0.0911   |         0.25 |
| ('ehrsim2_order', 'bag', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                      | L_PCH |   3 |   0.0111 |     0.0006 | 0/3    |  0.00105  |         0.25 |
| ('ehrsim2_order', 'sequence', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |   0.0082 |     0.0005 | 0/3    |  0.00139  |         0.25 |
| ('ehrsim2_order', 'static', 'Brier ↓', 'L_PCH + L_rank + L_mul (paper)')                   | L_PCH |   3 |   0.0013 |     0.0004 | 0/3    |  0.0229   |         0.25 |
| ('ehrsim2_level', 'bag', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |  91.0894 |    33.4579 | 0/3    |  0.0421   |         0.25 |
| ('ehrsim2_level', 'sequence', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 |  96.181  |    10.3476 | 0/3    |  0.00384  |         0.25 |
| ('ehrsim2_level', 'static', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')              | L_PCH |   3 | -12.957  |    16.3508 | 2/3    |  0.304    |         0.5  |
| ('ehrsim2_mixed', 'bag', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |  68.6134 |     6.5009 | 0/3    |  0.00298  |         0.25 |
| ('ehrsim2_mixed', 'sequence', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 |  50.0931 |     8.8291 | 0/3    |  0.0102   |         0.25 |
| ('ehrsim2_mixed', 'static', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')              | L_PCH |   3 | -22.2262 |    23.0998 | 2/3    |  0.238    |         0.5  |
| ('ehrsim2_order', 'bag', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')                 | L_PCH |   3 |  51.1414 |     9.0086 | 0/3    |  0.0102   |         0.25 |
| ('ehrsim2_order', 'sequence', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')            | L_PCH |   3 |  31.5914 |    14.82   | 0/3    |  0.0662   |         0.25 |
| ('ehrsim2_order', 'static', 'MAE-margin ↓', 'L_PCH + L_rank + L_mul (paper)')              | L_PCH |   3 | -25.2685 |     3.7542 | 3/3    |  0.00728  |         0.25 |


### Sequence / static vs bag-of-codes input (paired on seed)

|                                                                                       | vs   |   n |   delta |   delta_sd | wins   |   p_ttest |   p_wilcoxon |
|:--------------------------------------------------------------------------------------|:-----|----:|--------:|-----------:|:-------|----------:|-------------:|
| ('ehrsim2_level', 'L_PCH', 'C_td ↑', 'sequence')                                      | bag  |   3 |  0.0095 |     0.0019 | 3/3    |  0.0133   |         0.25 |
| ('ehrsim2_level', 'L_PCH', 'C_td ↑', 'static')                                        | bag  |   3 | -0.2137 |     0.0035 | 0/3    |  8.88e-05 |         0.25 |
| ('ehrsim2_level', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'sequence')             | bag  |   3 |  0.0064 |     0.0039 | 3/3    |  0.103    |         0.25 |
| ('ehrsim2_level', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'static')               | bag  |   3 | -0.2174 |     0.0037 | 0/3    |  9.77e-05 |         0.25 |
| ('ehrsim2_mixed', 'L_PCH', 'C_td ↑', 'sequence')                                      | bag  |   3 | -0.0041 |     0.0013 | 0/3    |  0.0337   |         0.25 |
| ('ehrsim2_mixed', 'L_PCH', 'C_td ↑', 'static')                                        | bag  |   3 | -0.1928 |     0.003  | 0/3    |  7.82e-05 |         0.25 |
| ('ehrsim2_mixed', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'sequence')             | bag  |   3 | -0.0011 |     0.0052 | 1/3    |  0.757    |         1    |
| ('ehrsim2_mixed', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'static')               | bag  |   3 | -0.1928 |     0.0092 | 0/3    |  0.000751 |         0.25 |
| ('ehrsim2_order', 'L_PCH', 'C_td ↑', 'sequence')                                      | bag  |   3 | -0.0083 |     0.0074 | 0/3    |  0.192    |         0.25 |
| ('ehrsim2_order', 'L_PCH', 'C_td ↑', 'static')                                        | bag  |   3 | -0.1525 |     0.0139 | 0/3    |  0.00276  |         0.25 |
| ('ehrsim2_order', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'sequence')             | bag  |   3 | -0.0047 |     0.0034 | 0/3    |  0.134    |         0.25 |
| ('ehrsim2_order', 'L_PCH + L_rank + L_mul (paper)', 'C_td ↑', 'static')               | bag  |   3 | -0.1461 |     0.005  | 0/3    |  0.000389 |         0.25 |
| ('ehrsim2_level', 'L_PCH', 'Within-subject C ↑', 'sequence')                          | bag  |   3 |  0.0257 |     0.0599 | 2/3    |  0.534    |         0.5  |
| ('ehrsim2_level', 'L_PCH', 'Within-subject C ↑', 'static')                            | bag  |   3 | -0.073  |     0.04   | 0/3    |  0.0871   |         0.25 |
| ('ehrsim2_level', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'sequence') | bag  |   3 | -0.0413 |     0.0184 | 0/3    |  0.0601   |         0.25 |
| ('ehrsim2_level', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'static')   | bag  |   3 | -0.0692 |     0.0028 | 0/3    |  0.000539 |         0.25 |
| ('ehrsim2_mixed', 'L_PCH', 'Within-subject C ↑', 'sequence')                          | bag  |   3 |  0.0382 |     0.0551 | 2/3    |  0.353    |         0.5  |
| ('ehrsim2_mixed', 'L_PCH', 'Within-subject C ↑', 'static')                            | bag  |   3 | -0.1009 |     0.09   | 0/3    |  0.192    |         0.25 |
| ('ehrsim2_mixed', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'sequence') | bag  |   3 |  0.0009 |     0.0127 | 2/3    |  0.914    |         1    |
| ('ehrsim2_mixed', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'static')   | bag  |   3 | -0.0478 |     0.0075 | 0/3    |  0.00805  |         0.25 |
| ('ehrsim2_order', 'L_PCH', 'Within-subject C ↑', 'sequence')                          | bag  |   3 |  0.0043 |     0.0526 | 2/3    |  0.899    |         1    |
| ('ehrsim2_order', 'L_PCH', 'Within-subject C ↑', 'static')                            | bag  |   3 | -0.0589 |     0.035  | 0/3    |  0.1      |         0.25 |
| ('ehrsim2_order', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'sequence') | bag  |   3 |  0.0012 |     0.0057 | 1/3    |  0.741    |         1    |
| ('ehrsim2_order', 'L_PCH + L_rank + L_mul (paper)', 'Within-subject C ↑', 'static')   | bag  |   3 | -0.0451 |     0.0116 | 0/3    |  0.0212   |         0.25 |
| ('ehrsim2_level', 'L_PCH', 'Brier ↓', 'sequence')                                     | bag  |   3 | -0.0017 |     0.0036 | 2/3    |  0.492    |         0.75 |
| ('ehrsim2_level', 'L_PCH', 'Brier ↓', 'static')                                       | bag  |   3 |  0.036  |     0.0046 | 0/3    |  0.00537  |         0.25 |
| ('ehrsim2_level', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'sequence')            | bag  |   3 | -0.0005 |     0.0019 | 2/3    |  0.703    |         0.75 |
| ('ehrsim2_level', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'static')              | bag  |   3 |  0.023  |     0.0008 | 0/3    |  0.000398 |         0.25 |
| ('ehrsim2_mixed', 'L_PCH', 'Brier ↓', 'sequence')                                     | bag  |   3 |  0.0049 |     0.0031 | 0/3    |  0.11     |         0.25 |
| ('ehrsim2_mixed', 'L_PCH', 'Brier ↓', 'static')                                       | bag  |   3 |  0.0296 |     0.003  | 0/3    |  0.00351  |         0.25 |
| ('ehrsim2_mixed', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'sequence')            | bag  |   3 | -0.0017 |     0.0016 | 3/3    |  0.222    |         0.25 |
| ('ehrsim2_mixed', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'static')              | bag  |   3 |  0.0149 |     0.0019 | 0/3    |  0.00531  |         0.25 |
| ('ehrsim2_order', 'L_PCH', 'Brier ↓', 'sequence')                                     | bag  |   3 |  0.0018 |     0.0012 | 0/3    |  0.126    |         0.25 |
| ('ehrsim2_order', 'L_PCH', 'Brier ↓', 'static')                                       | bag  |   3 |  0.0211 |     0.0005 | 0/3    |  0.000181 |         0.25 |
| ('ehrsim2_order', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'sequence')            | bag  |   3 | -0.0011 |     0.0013 | 3/3    |  0.286    |         0.25 |
| ('ehrsim2_order', 'L_PCH + L_rank + L_mul (paper)', 'Brier ↓', 'static')              | bag  |   3 |  0.0114 |     0.0008 | 0/3    |  0.00146  |         0.25 |
