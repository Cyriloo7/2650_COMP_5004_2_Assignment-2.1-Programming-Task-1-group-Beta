| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| running_mean_and_variance | 0.00147 | 0.07044 | 47.90x |
| best_split_of_a_feature | 0.0005086 | 0.1671 | 328.52x |
| tree_prediction | 0.003202 | 0.04588 | 14.33x |
| confusion_matrix_counts | 0.007117 | 0.05466 | 7.68x |

Inputs: running moments 20,000 x 4 values; best split 2,000 values; tree prediction 20,000 samples (tree of depth 8, 65 leaves); confusion matrix 100,000 labels.
Mean of 3 timed runs after 1 warm-up run; random seed 0. 2026-11-16, Python 3.14.6, NumPy 2.5.3.
