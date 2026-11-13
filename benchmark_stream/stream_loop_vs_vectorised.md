| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| running_mean_and_variance | 0.00127 | 0.02385 | 18.79x |
| best_split_of_a_feature | 0.0003867 | 0.07165 | 185.28x |
| tree_prediction | 0.002173 | 0.01127 | 5.19x |
| confusion_matrix_counts | 0.006972 | 0.025 | 3.59x |

Inputs: running moments 20,000 x 4 values; best split 2,000 values; tree prediction 20,000 samples (tree of depth 8, 65 leaves); confusion matrix 100,000 labels.
Mean of 3 timed runs after 1 warm-up run; random seed 0. 2026-11-12, Python 3.12.3, NumPy 2.4.4.
