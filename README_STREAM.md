# numcompute_stream: Assignment 2.2 (a streaming, buffered decision tree)

`numcompute_stream` extends the Assignment 2.1 package `numcompute` into a streaming framework built around a single decision tree:

- Every part learns from data that arrives in **chunks**:
  - the model, the transformers and the pipeline use `partial_fit`;
  - metrics use `update`, `result` and `reset`;
  - statistics use `update_stats`.
- The tree uses **buffered retraining**. Each `partial_fit` adds the chunk to a buffer. Once the buffer holds `buffer_size` samples, the tree is retrained on the buffered data and the buffer is cleared. Between retrains, `predict` uses the latest tree.
- Only NumPy and matplotlib are used. The Assignment 2.1 code is **reused, not changed or copied** (see [Reuse of Assignment 2.1](#reuse-of-assignment-21)).

> This file is the README for Assignment 2.2. The repository's `README.md` belongs to Assignment 2.1 and is left as it was.

## What Assignment 2.2 adds to the repository

```text
numcompute_stream/                   the package (spec modules + io.py for the demo)
├── io.py                            load_csv, iter_chunks
├── preprocessing.py                 StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
├── stats.py                         StreamingStats (update_stats), update_moments, update_min_max
├── metrics.py                       Accuracy, Precision, Recall, F1, ConfusionMatrix, AUC, RollingMetric
├── tree.py                          DecisionTreeClassifier (buffered retraining)
├── pipeline.py                      Pipeline (partial_fit)
├── _checks.py                       check_int_at_least: one rule for every integer setting
├── stream.py                        StreamTrainer
└── visualise.py                     plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth
tests_stream/test_stream_*.py        9 test files, 160 tests
demo_stream/stream_demo.ipynb        the demo (already run); saved plots in demo_stream/figures/
benchmark_stream/stream_loop_vs_vectorised.py   benchmark (a): loops vs vectorised code
benchmark_stream/stream_buffer_size.py          benchmark (b): accuracy and retraining time vs buffer_size
(Dataset/Housing.csv                 the 2.1 dataset, already in the repo: the stream for the demo, benchmark and tests)
README_STREAM.md                     this file
```

None of these paths existed in Assignment 2.1, so adding them changes no 2.1 file. Every 2.2 folder ends in `_stream`, so none of them can clash with the 2.1 folders `benchmark/`, `demo/` and `tests/`. The data is the 2.1 file `Dataset/Housing.csv`, which is not copied or changed.

## How to run

From the repository root:

```bash
python -m pip install -e .                       # the 2.1 package (numpy)
python -m pip install matplotlib pytest notebook # extra tools for 2.2
python -m pytest                                 # 2.1 and 2.2 tests together
python -m pytest tests_stream                    # only the 2.2 tests
python -m benchmark_stream.stream_loop_vs_vectorised   # benchmark (a), about 5 s
python -m benchmark_stream.stream_buffer_size          # benchmark (b), about 5 s
python -m notebook demo_stream/stream_demo.ipynb # the demo
```

## Example

```python
from numcompute.io import DEFAULT_MISSING_VALUES
from numcompute_stream import (DecisionTreeClassifier, Imputer, OneHotEncoder, Pipeline, StandardScaler,
                               StreamTrainer, iter_chunks, load_csv)
from numcompute_stream import visualise as vis

# Housing data: predict "airconditioning"; the file also writes missing cells as NAN
X, y, names = load_csv("Dataset/Housing.csv", target_column="airconditioning",
                       missing_values=(*DEFAULT_MISSING_VALUES, "NAN"))
text = [names.index(n) for n in ("mainroad", "guestroom", "basement", "hotwaterheating", "prefarea", "furnishingstatus")]


def make_pipeline():
    """The specification's example, with the encoder and imputer that this data needs."""
    return Pipeline([
        ("encode", OneHotEncoder(columns=text, max_categories=3)),
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=4, buffer_size=100)),
    ])


pipe = make_pipeline()
for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=25):
    pipe.partial_fit(X_chunk, y_chunk)  # retrains every time 100 samples are buffered

# Or let StreamTrainer test, then train on every chunk and log the results
trainer = StreamTrainer(make_pipeline(), window=100).run(iter_chunks(X, y, chunk_size=25))
vis.plot_metric_over_time(trainer.get("rolling_accuracy"), title="Accuracy", ylabel="accuracy")
```

## How each part works

- **`tree.py`: `DecisionTreeClassifier(max_depth, min_samples_split, max_features, criterion, buffer_size)`**
  - **Building** follows the course's classification tree (`classification-tree.html`). `_build(X, y, depth)` tries every question, keeps the one with the biggest information gain, splits, repeats on each side, and stops at `max_depth`, at fewer than `min_samples_split` samples, at a pure node, or when no gain is above 0.
  - A question is `x[feature] <= threshold`, with thresholds halfway between neighbouring values. A leaf predicts its majority class (a tie goes to the first class) and keeps the class proportions for `predict_proba`.
  - A test checks that on yes/no data it makes exactly the same predictions as the course code.
  - **Buffered retraining**:
    - `partial_fit` appends the chunk to a buffer. When at least `buffer_size` samples are waiting, a new tree is trained from scratch on them and the buffer is cleared.
    - `predict` uses the latest tree in between. Before the first retrain it raises the 2.1 `NotFittedError`, saying how full the buffer is.
    - `fit` trains from scratch on the given data.
  - **Vectorised**:
    - the gain of every threshold of a feature is computed at once from cumulative class counts;
    - `predict` moves all samples down the tree together.
  - **Edge cases**:
    - NaN features raise an error asking for an `Imputer`;
    - NaN labels are rejected;
    - constant features cannot be split;
    - ties are broken deterministically;
    - neighbouring floats with no midpoint still split correctly;
    - new classes may appear between retrains;
    - empty chunks change nothing.
- **`preprocessing.py`**: every transformer extends its 2.1 class and adds `partial_fit`. The 2.1 `transform`, input checks and error messages are kept.
  - `StandardScaler`:
    - keeps a running mean and variance (Welford's method), so the result equals the 2.1 batch `fit`;
    - like 2.1, a constant column gets its exact value as mean and `scale_ = 0`, and an all-missing column gets NaN.
  - `MinMaxScaler`: running min and max.
  - `Imputer`: running means (`"mean"`) or a constant. The 2.1 strategies `"median"` and `"most_frequent"` work with `fit` only.
  - `OneHotEncoder`:
    - adds categories in order of arrival, so older categories keep their columns, and the 2.1 `transform` does the encoding;
    - `handle_unknown` defaults to `"ignore"` for streams, and missing values encode as all zeros;
    - `columns` picks which columns to encode, and `max_categories` keeps the output width fixed.
- **`stats.py`**:
  - `update_moments` (mean and variance) and `update_min_max` are shared helpers, also used by the scalers and imputer. `update_moments` is the vectorised, chunk form of the 2.1 `StreamingStats.merge`; a test checks they agree.
  - `StreamingStats.update_stats(X_chunk)` keeps:
    - mean, variance (exactly 0 for constant columns), min and max per feature;
    - quantiles from a random sample (using the 2.1 `quantile`);
    - histograms (using the 2.1 `histogram`), optionally over the last `window` chunks only.
- **`metrics.py`**:
  - Every metric has `update`, `result` and `reset`.
  - Accuracy, precision, recall and F1 share one accumulated confusion matrix, counted chunk by chunk with the 2.1 `confusion_matrix`, so streaming results equal batch results.
  - Precision, recall and F1 average with the 2.1 helper, so `average="binary" | "macro" | "micro"` and `pos_label` behave exactly like the 2.1 functions. The default is `"macro"` because streams are often multi-class.
  - `AUC` accumulates score histograms and measures the area with the 2.1 `auc`.
  - `RollingMetric` uses only the last `window` samples.
- **`pipeline.py`**: extends the 2.1 `Pipeline`.
  - `partial_fit` updates each transformer with the chunk, then transforms it for the next step.
  - It also adds `predict_proba`, `classes_` and `is_fitted`.
- **`stream.py`: `StreamTrainer`**
  - `fit_chunk` and `score_chunk` together run test-then-train evaluation.
  - Per chunk it logs the chunk, cumulative and rolling metrics, the training time, the memory footprint (pickled size, including the buffer), and for the buffered tree whether it retrained, how long that took, and how many samples are buffered.
  - Chunks that arrive before the first retrain are only used for training.
- **`visualise.py`**:
  - the three required plots (`compare_models` is used to compare buffer sizes);
  - each returns the figure and accepts `save_path=` and `show=`;
  - the confusion matrix comes from the 2.1 `confusion_matrix`.
- **`io.py`**:
  - `load_csv` reads the file with the 2.1 `read_csv`, and numeric columns are parsed by the 2.1 reader itself;
  - `iter_chunks` uses the 2.1 `batch_iterator`.

## Reuse of Assignment 2.1

Every piece below is imported from the 2.1 package `numcompute`; no 2.1 file was edited and no 2.1 function was copied.

| 2.2 file | 2.1 code it reuses |
|---|---|
| `io.py` | `read_csv` (text and numbers), `DEFAULT_MISSING_VALUES`, `batch_iterator` |
| `preprocessing.py` | subclasses `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder` (its `transform` and feature names); input checks, `TransformerMixin`, `NotFittedError`, `safe_divide` |
| `stats.py` | `histogram`, `quantile`, `safe_divide`, `NotFittedError` |
| `metrics.py` | `confusion_matrix`, `auc`, the averaging helper behind `precision` / `recall` / `f1`, `_has_nan`, `_kind_group` |
| `tree.py` | input checks, `NotFittedError`, `safe_divide`, `_has_nan`, `_kind_group` |
| `_checks.py` | `_is_integer` (the one integer rule shared by `tree`, `stats`, `metrics` and `preprocessing`) |
| `pipeline.py` | subclasses `Pipeline` |
| `stream.py`, `visualise.py` | `NotFittedError`; `confusion_matrix` |
| `benchmark_stream/stream_loop_vs_vectorised.py` | `compare`, `format_results_table`; the 2.1 `StreamingStats` as a loop baseline |

**No duplicated code.** An automatic check compared all 98 functions of `numcompute_stream` with the 118 functions of `numcompute`:
- No 2.2 function is identical or near-identical to a 2.1 function, and none duplicates another 2.2 function.
- The one rule that was repeated in 8 places (a setting must be an integer >= 1) now lives in `_checks.py`, with one error message form everywhere.
- There is no hand-written safe division, integer check, NaN check, CSV parsing or confusion counting; these all come from 2.1.

## Results

**(a) Loops vs vectorised code.** 2.1's `compare()` checks that both versions give the same result before reporting a speed-up. Speed-ups vary a little from run to run.

| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| running_mean_and_variance | 0.00127 | 0.02385 | 18.79x |
| best_split_of_a_feature | 0.0003867 | 0.07165 | 185.28x |
| tree_prediction | 0.002173 | 0.01127 | 5.19x |
| confusion_matrix_counts | 0.006972 | 0.025 | 3.59x |

Inputs: running moments 20,000 x 4 values; best split 2,000 values; tree prediction 20,000 samples (tree of depth 8, 65 leaves); confusion matrix 100,000 labels.
Mean of 3 timed runs after 1 warm-up run; random seed 0. 2026-11-12, Python 3.12.3, NumPy 2.4.4.

**(b) Buffer size.** The stream is `Dataset/Housing.csv` (the 2.1 dataset: 544 houses with a known `airconditioning` label), in chunks of 25 rows. The pipeline is the same for every row: encode the text columns, fill missing values, scale, then `DecisionTreeClassifier(max_depth=4, buffer_size=...)`, tested then trained on every chunk. The data has missing cells (also written as `NAN`) and text columns, so the encoder and imputer do real work.

Stream: Dataset/Housing.csv, 544 houses, label `airconditioning`, chunks of 25 rows (22 chunks). A model is scored only once its first buffer has been trained, so the accuracy column for chunks 12-21 compares every buffer size on the same samples.

**File order** (sorted by price, high to low, so the data changes over the stream):

| buffer_size | First scored chunk | Retrains | Mean retrain time (ms) | Accuracy (chunks 12-21) | Accuracy (all scored chunks) | Training time (s) | Peak memory (KB) |
|---|---|---|---|---|---|---|---|
| 50 | 2 | 10 | 4.1 | 0.828 | 0.715 | 0.051 | 14 |
| 100 | 4 | 5 | 6.5 | 0.848 | 0.730 | 0.042 | 21 |
| 200 | 8 | 2 | 8.9 | 0.717 | 0.718 | 0.027 | 41 |
| 300 | 12 | 1 | 12.5 | 0.873 | 0.873 | 0.023 | 60 |
| always predict the most common label | - | - | - | 0.873 | 0.686 | - | - |

**Shuffled order** (seed 0, no change over time):

| buffer_size | First scored chunk | Retrains | Mean retrain time (ms) | Accuracy (chunks 12-21) | Accuracy (all scored chunks) | Training time (s) | Peak memory (KB) |
|---|---|---|---|---|---|---|---|
| 50 | 2 | 10 | 5.1 | 0.693 | 0.686 | 0.061 | 14 |
| 100 | 4 | 5 | 6.8 | 0.676 | 0.685 | 0.042 | 21 |
| 200 | 8 | 2 | 8.7 | 0.697 | 0.695 | 0.025 | 41 |
| 300 | 12 | 1 | 12.0 | 0.779 | 0.779 | 0.021 | 60 |
| always predict the most common label | - | - | - | 0.672 | 0.686 | - | - |

- **Retraining cost:**
  - small buffers retrain more often (10 times for 50 samples, 1 time for 300);
  - each retrain of a small buffer is quicker (4-6 ms for 50-100 samples against 12 ms for 300; timings vary a little from run to run);
  - the memory footprint grows with the buffer (14 KB for 50, 60 KB for 300).
- **Delay:** a model cannot predict before its first buffer is trained, so a larger buffer starts later (chunk 12 for 300, chunk 2 for 50).
- **Accuracy needs care.**
  - About 69% of the houses have no air conditioning, and in the last chunks (the cheaper houses of the price-sorted file) about 87% have none.
  - In file order, the largest buffer only ties "always answer no" on those chunks (0.873), and the smaller buffers are below it.
  - In shuffled order, only the buffer of 300 clearly beats the baseline (0.779 against 0.672).
  - So on this small data set, how many samples each tree sees matters more than how often it retrains. The signal is weak, and the comparison covers only about 245 samples, so differences of a few points are within noise.

## Tests

- **Counts**: 160 tests in `tests_stream/` cover **100% of the lines and 100% of the branches** of `numcompute_stream`. Together with the 627 Assignment 2.1 tests, all 787 pass, and the 2.1 coverage gate (100% of `numcompute`) still passes.
- **What they check**:
  - streaming results against the 2.1 batch results: every metric and every `average`, the scalers, imputer and encoder, and the running mean and variance against the 2.1 `StreamingStats`;
  - the vectorised split search and prediction against simple loops, and the tree against the course's own tree code;
  - buffered retraining: the buffer fills, retrains at `buffer_size`, is cleared, and `predict` uses the latest tree in between;
  - an end-to-end run of the demo's pipeline on `Housing.csv` (text columns, missing values, retraining every 100 samples);
  - **robustness** (`test_stream_robustness.py`):
    - results do not depend on the chunk size (1, 7, 64 or all rows at once);
    - the same buffered tree comes out for every chunk size that divides `buffer_size`;
    - the buffer always stays below `buffer_size`, so memory stays bounded;
    - a variance and a mean that are exact for values like 1e9 + 0, 1, 2, ... fed one at a time;
    - 1e9 and 1e9 + 1 are told apart inside a column that also holds -1e18 and 1e18;
    - values near 1e308 cannot overflow a threshold;
    - 100,000 identical rows give one leaf quickly;
    - ties choose the lowest feature and the lowest threshold;
  - edge cases: NaN features, labels and scores; constant and all-missing columns; empty chunks and chunks bigger than the buffer; new classes or categories mid-stream; wrong shapes, text input, unfitted models and invalid settings.
- **Without matplotlib**: the plotting tests are skipped if matplotlib is not installed. The 2.1 `requirements.txt` lists only NumPy.

## Debugging log

Each problem was found by a test or a measurement, diagnosed, fixed, and then pinned by a test that fails without the fix.

| # | What went wrong | Diagnosis | Fix | Test that now guards it |
|---|---|---|---|---|
| 1 | The tree grew leaf by leaf, not by buffered retraining as the specification asks | Found when checking the code against the specification | Rewrote `tree.py` with the course's recursive build and a buffer | `test_partial_fit_buffers_until_buffer_size_then_retrains_and_clears` |
| 2 | `StreamTrainer` tried to score a chunk before any model existed | A model without `is_fitted` was treated as ready | `model_ready` also needs one `fit_chunk` first | `test_models_without_a_buffer_are_logged_too` |
| 3 | `StandardScaler` turned a constant column of 0.1 into -1 instead of 0 | Round-off left a tiny variance, and the scaler divided by almost zero | Detect constant columns exactly (min == max), as 2.1 does | `test_constant_and_all_missing_columns_match_the_2_1_scaler` |
| 4 | `ConfusionMatrix` changed its labels before rejecting NaN input | The validation came after the state update | Validate first | `test_a_rejected_update_leaves_the_confusion_matrix_unchanged` |
| 5 | Precision, recall and F1 repeated the 2.1 averaging logic | Found by comparing 2.2 functions with 2.1 functions | Reuse the 2.1 helper (`binary`, `macro`, `micro` now match 2.1) | `test_streaming_scores_equal_the_2_1_batch_scores` |
| 6 | A threshold between values near +1.7e308 and -1.7e308 raised an overflow warning | `high - low` overflows, and only a later guard hid the result | Compute `low / 2 + high / 2`, and keep the guard for neighbouring floats | `test_extreme_values_cannot_overflow_the_threshold` |
| 7 | All-equal features still paid for the cumulative class counts | Measured: 33 ms for 200,000 x 5 constant values | Skip a constant feature right after its sort (10 ms) | `test_all_equal_features_are_cheap_and_give_one_leaf` (checks the result and a generous time bound) |
| 8 | The integer check for settings was repeated in 8 places with different messages | The duplicate audit flagged two near-identical constructors | One shared `check_int_at_least` | `test_integer_settings_are_checked_the_same_way_everywhere` |

## Design decisions and known limits

Every limit below is either fixed above or a deliberate choice, with the reason.

- **The buffered tree forgets older data.** The specification says to retrain on the buffer and clear it. The benchmark shows the cost: a small buffer sees few samples per tree. The benefit is bounded memory and a tree that follows a changing stream.
- **Quantiles are approximate once the stream is longer than `sample_size`.** They come from a random sample (reservoir sampling), so memory does not grow. They are exact until that many rows have been seen. Raise `sample_size` for more accuracy.
- **`AUC` is approximate.** Scores are counted in 1,000 bins, so memory stays small. Scores in the same bin count as ties, so the error is at most about 1/1000 of the score range (a test checks it against the pairwise definition).
- **Loops over features, never over samples.** `best_split`, the histograms and the encoder loop over the columns (a handful), and everything inside is NumPy. The recursion in `_build` is as deep as `max_depth`.
- **The `Imputer` supports `mean` and `constant` in a stream.** `median` and `most_frequent` need all the data at once, so they raise a clear error in `partial_fit` and work with `fit`.
- **Text columns must be encoded before the tree.** The tree rejects text and NaN with a message that names the fix (`OneHotEncoder`, `Imputer`).
- **Floating-point limits.** Scaling a column that holds both 1e18 and values near 1e9 can hide a difference of 1, because `x - mean` loses it in float64. This is the nature of 64-bit floats, not a bug. The tree itself, used on the raw column, keeps such values apart (tested).
