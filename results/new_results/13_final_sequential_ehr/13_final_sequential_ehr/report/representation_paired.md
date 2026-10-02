### Sequence / static vs bag-of-codes input (paired on seed)

|                                                                   | vs   |   n |   delta |   delta_sd | wins   |   p_ttest |   p_wilcoxon |
|:------------------------------------------------------------------|:-----|----:|--------:|-----------:|:-------|----------:|-------------:|
| ('ehrsim2_level', 'L_PCH', 'C_td ↑', 'sequence')                  | bag  |   5 |  0.0029 |     0.0084 | 3/5    |   0.478   |       0.625  |
| ('ehrsim2_level', 'L_PCH', 'C_td ↑', 'sequence_time')             | bag  |   5 | -0.0157 |     0.0087 | 0/5    |   0.0157  |       0.0625 |
| ('ehrsim2_mixed', 'L_PCH', 'C_td ↑', 'sequence')                  | bag  |   5 | -0.003  |     0.0027 | 0/5    |   0.0703  |       0.0625 |
| ('ehrsim2_mixed', 'L_PCH', 'C_td ↑', 'sequence_time')             | bag  |   5 | -0.0199 |     0.0056 | 0/5    |   0.00137 |       0.0625 |
| ('ehrsim2_order', 'L_PCH', 'C_td ↑', 'sequence')                  | bag  |   5 | -0.0016 |     0.0028 | 1/5    |   0.264   |       0.312  |
| ('ehrsim2_order', 'L_PCH', 'C_td ↑', 'sequence_time')             | bag  |   5 | -0.0121 |     0.0051 | 0/5    |   0.00605 |       0.0625 |
| ('ehrsim2_level', 'L_PCH', 'Within-subject C ↑', 'sequence')      | bag  |   5 |  0.005  |     0.0131 | 3/5    |   0.443   |       0.438  |
| ('ehrsim2_level', 'L_PCH', 'Within-subject C ↑', 'sequence_time') | bag  |   5 | -0.0173 |     0.0209 | 2/5    |   0.137   |       0.312  |
| ('ehrsim2_mixed', 'L_PCH', 'Within-subject C ↑', 'sequence')      | bag  |   5 | -0.018  |     0.0169 | 1/5    |   0.0754  |       0.125  |
| ('ehrsim2_mixed', 'L_PCH', 'Within-subject C ↑', 'sequence_time') | bag  |   5 | -0.0211 |     0.0424 | 2/5    |   0.328   |       0.438  |
| ('ehrsim2_order', 'L_PCH', 'Within-subject C ↑', 'sequence')      | bag  |   5 |  0.0001 |     0.0208 | 2/5    |   0.995   |       1      |
| ('ehrsim2_order', 'L_PCH', 'Within-subject C ↑', 'sequence_time') | bag  |   5 |  0.0084 |     0.0319 | 3/5    |   0.589   |       0.438  |
| ('ehrsim2_level', 'L_PCH', 'Brier ↓', 'sequence')                 | bag  |   5 |  0.0015 |     0.0033 | 2/5    |   0.374   |       0.438  |
| ('ehrsim2_level', 'L_PCH', 'Brier ↓', 'sequence_time')            | bag  |   5 |  0.0051 |     0.002  | 0/5    |   0.00451 |       0.0625 |
| ('ehrsim2_mixed', 'L_PCH', 'Brier ↓', 'sequence')                 | bag  |   5 |  0.0033 |     0.0026 | 0/5    |   0.0432  |       0.0625 |
| ('ehrsim2_mixed', 'L_PCH', 'Brier ↓', 'sequence_time')            | bag  |   5 |  0.0054 |     0.0029 | 0/5    |   0.0139  |       0.0625 |
| ('ehrsim2_order', 'L_PCH', 'Brier ↓', 'sequence')                 | bag  |   5 |  0.0028 |     0.0041 | 0/5    |   0.2     |       0.0625 |
| ('ehrsim2_order', 'L_PCH', 'Brier ↓', 'sequence_time')            | bag  |   5 |  0.0041 |     0.0041 | 0/5    |   0.0911  |       0.0625 |
