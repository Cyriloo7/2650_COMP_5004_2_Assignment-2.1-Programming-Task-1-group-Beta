| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| running_mean_and_variance | 0.002135 | 0.12 | 56.18x |
| best_split_of_a_feature | 0.0007704 | 0.1585 | 205.72x |
| tree_prediction | 0.003387 | 0.03354 | 9.90x |
| confusion_matrix_counts | 0.007102 | 0.0538 | 7.58x |

Inputs: running moments 20,000 x 4 values; best split 2,000 values; tree prediction 20,000 samples (tree of depth 8, 65 leaves); confusion matrix 100,000 labels.
Mean of 3 timed runs after 1 warm-up run; random seed 0. 2026-11-16, Python 3.14.6, NumPy 2.5.3.
