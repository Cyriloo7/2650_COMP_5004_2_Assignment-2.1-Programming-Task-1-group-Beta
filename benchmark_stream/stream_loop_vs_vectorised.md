| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| running_mean_and_variance | 0.001427 | 0.03144 | 22.04x |
| best_split_of_a_feature | 0.000536 | 0.1003 | 187.17x |
| tree_prediction | 0.0026 | 0.01458 | 5.61x |
| confusion_matrix_counts | 0.008174 | 0.03186 | 3.90x |

Inputs: running moments 20,000 x 4 values; best split 2,000 values; tree prediction 20,000 samples (tree of depth 8, 65 leaves); confusion matrix 100,000 labels.
Mean of 3 timed runs after 1 warm-up run; random seed 0. 2026-11-12, Python 3.12.3, NumPy 2.4.4.
