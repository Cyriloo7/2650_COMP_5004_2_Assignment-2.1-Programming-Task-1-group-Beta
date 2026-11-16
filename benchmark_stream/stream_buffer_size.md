Stream: Dataset/Housing.csv, 544 houses, label `airconditioning`, chunks of 25 rows (22 chunks). A model is scored only once its first buffer has been trained, so the accuracy column for chunks 12-21 compares every buffer size on the same samples.

**File order** (sorted by price, high to low, so the data changes over the stream):

| buffer_size | First scored chunk | Retrains | Mean retrain time (ms) | Accuracy (chunks 12-21) | Accuracy (all scored chunks) | Training time (s) | Peak memory (KB) |
|---|---|---|---|---|---|---|---|
| 50 | 2 | 10 | 12.4 | 0.828 | 0.715 | 0.154 | 14 |
| 100 | 4 | 5 | 16.7 | 0.848 | 0.730 | 0.112 | 21 |
| 200 | 8 | 2 | 20.6 | 0.717 | 0.718 | 0.068 | 41 |
| 300 | 12 | 1 | 27.1 | 0.873 | 0.873 | 0.056 | 60 |
| always predict the most common label | - | - | - | 0.873 | 0.686 | - | - |

**Shuffled order** (seed 0, no change over time):

| buffer_size | First scored chunk | Retrains | Mean retrain time (ms) | Accuracy (chunks 12-21) | Accuracy (all scored chunks) | Training time (s) | Peak memory (KB) |
|---|---|---|---|---|---|---|---|
| 50 | 2 | 10 | 13.2 | 0.693 | 0.686 | 0.158 | 14 |
| 100 | 4 | 5 | 17.8 | 0.676 | 0.685 | 0.113 | 21 |
| 200 | 8 | 2 | 21.2 | 0.697 | 0.695 | 0.066 | 41 |
| 300 | 12 | 1 | 26.0 | 0.779 | 0.779 | 0.047 | 60 |
| always predict the most common label | - | - | - | 0.672 | 0.686 | - | - |
