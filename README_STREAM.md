# numcompute_stream: Assignment 2.2 (a streaming, buffered decision tree)

`numcompute_stream` extends the Assignment 2.1 package `numcompute` into a streaming framework built around a single decision tree:

- Every part learns from data that arrives in **chunks**:
  - the model, the transformers and the pipeline use `partial_fit`;
  - metrics use `update`, `result` and `reset`;
  - statistics use `update_stats`.
- The tree uses **buffered retraining**. Each `partial_fit` adds the chunk to a buffer. Once the buffer holds `buffer_size` samples, the tree is retrained on the buffered data and the buffer is cleared. Between retrains, `predict` uses the latest tree.
- Only NumPy and matplotlib are used. The Assignment 2.1 code is **reused, not changed or copied** (see [Reuse of Assignment 2.1](#reuse-of-assignment-21)).

> This file is the README for Assignment 2.2. The repository's `README.md` belongs to Assignment 2.1 and is left as it was.

**Where each required item is.** Every 2.2 folder ends in `_stream`, so it cannot clash with the 2.1 folders of the same name:

| The instructions ask for | In this repository |
|---|---|
| `numcompute_stream/` package with all modules | `numcompute_stream/` |
| `tests/` folder (at least 30 unit tests) | `tests_stream/` (185 tests) |
| `demo/stream_demo.ipynb` | `demo_stream/stream_demo.ipynb` |
| `benchmark/` (loop vs vectorised) | `benchmark_stream/` (loop vs vectorised, and buffer sizes) |
| `README.md` with usage instructions | `README_STREAM.md` (this file) |

## What Assignment 2.2 adds to the repository

```text
numcompute_stream/                   the package (spec modules + io.py for the demo)
├── io.py                            load_csv, iter_chunks
├── preprocessing.py                 StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
├── stats.py                         StreamingStats (update_stats), update_moments, update_min_max
├── metrics.py                       Accuracy, Precision, Recall, F1, ConfusionMatrix, AUC, RollingMetric
├── tree.py                          DecisionTreeClassifier (buffered retraining, feature_importances_, export_text)
├── pipeline.py                      Pipeline (partial_fit)
├── _checks.py                       check_int_at_least: one rule for every integer setting
├── stream.py                        StreamTrainer
└── visualise.py                     plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth,
                                     plot_parameter_sweep
tests_stream/test_stream_*.py        9 test files, 185 tests
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
python -m pip install -e ".[dev]"                # numpy, plus matplotlib, pytest and pytest-cov
python -m pip install notebook                   # to open the demo
python -m pytest                                 # 2.1 and 2.2 tests together (812)
python -m pytest tests_stream                    # only the 2.2 tests (185)
python -m pytest tests_stream --cov=numcompute_stream --cov-report=term-missing   # with line and branch coverage
python -m benchmark_stream.stream_loop_vs_vectorised   # benchmark (a), a few seconds
python -m benchmark_stream.stream_buffer_size          # benchmark (b), a few seconds
python -m notebook demo_stream/stream_demo.ipynb # the demo
```

## Example

```python
from numcompute.io import DEFAULT_MISSING_VALUES
from numcompute_stream import (AUC, F1, Accuracy, DecisionTreeClassifier, Imputer, OneHotEncoder, Pipeline,
                               StandardScaler, StreamTrainer, iter_chunks, load_csv)
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

# Or let StreamTrainer test, then train on every chunk and log the results.
# AUC needs probabilities: the trainer passes it predict_proba scores automatically.
metrics = {"accuracy": Accuracy(), "f1": F1(), "auc": AUC(pos_label="yes")}
trainer = StreamTrainer(make_pipeline(), metrics=metrics, window=100).run(iter_chunks(X, y, chunk_size=25))
vis.plot_metric_over_time(trainer.get("rolling_accuracy"), title="Accuracy", ylabel="accuracy")

# What did the latest tree learn?
tree = trainer.model["model"]
print(tree.export_text(trainer.model["encode"].get_feature_names_out(names)))
print(tree.feature_importances_)
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
  - **Validate, then commit**: a chunk (or a `fit`) is checked completely before anything changes, so a rejected call never alters the tree, its buffer, `classes_` or `n_features_in_`. The first non-empty chunk fixes the number of features, as in the transformers.
  - **Interpretation**: `feature_importances_` is each feature's share of the information gain of the latest tree; `export_text(feature_names)` prints its questions as indented rules.
  - **Vectorised**:
    - the gain of every threshold of a feature is computed at once from cumulative class counts;
    - `predict` moves all samples down the tree together.
  - **Edge cases**:
    - NaN features raise an error asking for an `Imputer`;
    - NaN labels are rejected, and so are labels mixing text and numbers;
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
    - finds new categories with one `np.unique` per column (no Python loop over samples); a column that mixes text and numbers is rejected, as in 2.1;
    - `handle_unknown` defaults to `"ignore"` for streams, and missing values encode as all zeros;
    - `columns` picks which columns to encode (checked: integers inside X, each named once; negative indices count from the end), and `max_categories` keeps the output width fixed.
- **`stats.py`**:
  - `update_moments` (mean and variance) and `update_min_max` are shared helpers, also used by the scalers and imputer. `update_moments` is the vectorised, chunk form of the 2.1 `StreamingStats.merge`; a test checks they agree.
  - `StreamingStats.update_stats(X_chunk)` keeps:
    - mean, variance (exactly 0 for constant columns), min and max per feature;
    - quantiles from a random sample (using the 2.1 `quantile`);
    - histograms (using the 2.1 `histogram`), optionally over the last `window` chunks only.
  - Input goes through the 2.1 numeric check: NaN means missing, while text and infinity are rejected with a message naming the row and column.
- **`metrics.py`**:
  - Every metric has `update`, `result` and `reset`.
  - Accuracy, precision, recall and F1 share one accumulated confusion matrix, counted chunk by chunk with the 2.1 `confusion_matrix`, so streaming results equal batch results.
  - Precision, recall and F1 average with the 2.1 helper, so `average="binary" | "macro" | "micro"` and `pos_label` behave exactly like the 2.1 functions. The default is `"macro"` because streams are often multi-class.
  - `AUC` accumulates score histograms and measures the area with the 2.1 `auc`. It is marked `needs_scores = True`, so `StreamTrainer` gives it probabilities instead of labels.
  - `RollingMetric` uses only the last `window` samples (and reports the `needs_scores` and `pos_label` of the metric it wraps).
- **`pipeline.py`**: extends the 2.1 `Pipeline`.
  - `partial_fit` updates each transformer with the chunk, then transforms it for the next step. Empty chunks are skipped (a transformer that has seen no data could not transform them).
  - It also adds `predict_proba`, `classes_` and `is_fitted`.
- **`stream.py`: `StreamTrainer`**
  - `fit_chunk` and `score_chunk` together run test-then-train evaluation.
  - Per chunk it logs the chunk, cumulative and rolling metrics, the training time, the memory footprint (pickled size, including the buffer), and for the buffered tree whether it retrained, how long that took, and how many samples are buffered.
  - Label metrics get `predict(X)`; score metrics such as `AUC` get the `predict_proba` column of their `pos_label` (0 for a label the model has not seen yet).
  - Metrics are checked when the trainer is created, with clear errors (an object without `update`/`result`/`reset`, a `ConfusionMatrix`, or an `AUC` for a model without `predict_proba`).
  - Chunks that arrive before the first retrain are only used for training; an empty chunk never counts as training.
- **`visualise.py`**:
  - the three required plots (`compare_models` is used to compare buffer sizes), plus `plot_parameter_sweep` for a result against a setting such as `buffer_size`;
  - each returns the figure and accepts `save_path=` and `show=`; chunk axes use whole-number ticks;
  - the confusion matrix comes from the 2.1 `confusion_matrix`.
- **`io.py`**:
  - `load_csv` reads the file with the 2.1 `read_csv`, and numeric columns are parsed by the 2.1 reader itself;
  - `target_column` is a name or an index inside the header (an out-of-range index is an error, never wrapped around);
  - `iter_chunks` uses the 2.1 `batch_iterator`.

Every public class, method and function has a NumPy-style docstring with its parameters, input and output shapes and errors.

## Reuse of Assignment 2.1

Every piece below is imported from the 2.1 package `numcompute`; no 2.1 file was edited and no 2.1 function was copied.

| 2.2 file | 2.1 code it reuses |
|---|---|
| `io.py` | `read_csv` (text and numbers), `DEFAULT_MISSING_VALUES`, `batch_iterator`, `_is_integer` |
| `preprocessing.py` | subclasses `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder` (its `transform` and feature names); input checks, `TransformerMixin`, `NotFittedError`, `safe_divide`, `_is_integer` |
| `stats.py` | `histogram`, `quantile`, `safe_divide`, `NotFittedError`, the numeric input check `_check_numeric` |
| `metrics.py` | `confusion_matrix`, `auc`, the averaging helper behind `precision` / `recall` / `f1`, `_has_nan`, `_kind_group` |
| `tree.py` | input checks, `NotFittedError`, `safe_divide`, `_has_nan`, `_kind_group` |
| `_checks.py` | `_is_integer` (the one integer rule shared by `tree`, `stats`, `metrics` and `preprocessing`) |
| `pipeline.py` | subclasses `Pipeline` |
| `stream.py`, `visualise.py` | `NotFittedError`; `confusion_matrix` |
| `benchmark_stream/stream_loop_vs_vectorised.py` | `compare`, `format_results_table`; the 2.1 `StreamingStats` as a loop baseline |

**No duplicated code.** An automatic check (AST comparison of every function body, docstrings removed) compared all 109 functions of `numcompute_stream` with the 118 functions of `numcompute`:
- No 2.2 function is identical to a 2.1 function, and none duplicates another 2.2 function.
- One near-match remains: the three NumPy lines that turn a confusion matrix into per-class totals (`_FromConfusionMatrix.result` and the 2.1 `_class_counts`). The 2.1 helper takes raw labels, while the streaming metric only keeps the accumulated matrix, so it cannot be reused without changing 2.1.
- The one rule that was repeated in 8 places (a setting must be an integer >= 1) lives in `_checks.py`, with one error message form everywhere.
- There is no hand-written safe division, integer check, NaN check, CSV parsing or confusion counting; these all come from 2.1.

## Results

**(a) Loops vs vectorised code.** 2.1's `compare()` checks that both versions give the same result before reporting a speed-up.

| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| running_mean_and_variance | 0.00147 | 0.07044 | 47.90x |
| best_split_of_a_feature | 0.0005086 | 0.1671 | 328.52x |
| tree_prediction | 0.003202 | 0.04588 | 14.33x |
| confusion_matrix_counts | 0.007117 | 0.05466 | 7.68x |

Inputs: running moments 20,000 x 4 values; best split 2,000 values; tree prediction 20,000 samples (tree of depth 8, 65 leaves); confusion matrix 100,000 labels.
Mean of 3 timed runs after 1 warm-up run; random seed 0. 2026-11-16, Python 3.14.6, NumPy 2.5.3. The exact speed-ups depend on the machine and change between runs (an earlier run on the same laptop, with NumPy 2.5.5, gave 28x, 288x, 8.5x and 7.7x), but their order stays the same: the split search gains most because its loop version is O(n²) per feature.

**(b) Buffer size.** The stream is `Dataset/Housing.csv` (the 2.1 dataset: 544 houses with a known `airconditioning` label), in chunks of 25 rows. The pipeline is the same for every row: encode the text columns, fill missing values, scale, then `DecisionTreeClassifier(max_depth=4, buffer_size=...)`, tested then trained on every chunk. The data has missing cells (also written as `NAN`) and text columns, so the encoder and imputer do real work.

Stream: Dataset/Housing.csv, 544 houses, label `airconditioning`, chunks of 25 rows (22 chunks). A model is scored only once its first buffer has been trained, so the accuracy column for chunks 12-21 compares every buffer size on the same samples.

**File order** (sorted by price, high to low, so the data changes over the stream):

| buffer_size | First scored chunk | Retrains | Mean retrain time (ms) | Accuracy (chunks 12-21) | Accuracy (all scored chunks) | Training time (s) | Peak memory (KB) |
|---|---|---|---|---|---|---|---|
| 50 | 2 | 10 | 10.9 | 0.828 | 0.715 | 0.137 | 14 |
| 100 | 4 | 5 | 22.0 | 0.848 | 0.730 | 0.153 | 21 |
| 200 | 8 | 2 | 25.4 | 0.717 | 0.718 | 0.088 | 41 |
| 300 | 12 | 1 | 25.6 | 0.873 | 0.873 | 0.048 | 60 |
| always predict the most common label | - | - | - | 0.873 | 0.686 | - | - |

**Shuffled order** (seed 0, no change over time):

| buffer_size | First scored chunk | Retrains | Mean retrain time (ms) | Accuracy (chunks 12-21) | Accuracy (all scored chunks) | Training time (s) | Peak memory (KB) |
|---|---|---|---|---|---|---|---|
| 50 | 2 | 10 | 12.8 | 0.693 | 0.686 | 0.152 | 15 |
| 100 | 4 | 5 | 17.8 | 0.676 | 0.685 | 0.113 | 21 |
| 200 | 8 | 2 | 21.1 | 0.697 | 0.695 | 0.070 | 42 |
| 300 | 12 | 1 | 25.3 | 0.779 | 0.779 | 0.053 | 60 |
| always predict the most common label | - | - | - | 0.672 | 0.686 | - | - |

The benchmark also saves four plots in `benchmark_stream/`: rolling accuracy over the stream for every buffer size, buffer sizes 50 vs 200 (`compare_models`), and accuracy and mean retrain time against `buffer_size` (`plot_parameter_sweep`).

- **Retraining cost:**
  - small buffers retrain more often (10 times for 50 samples, 1 time for 300);
  - each retrain of a small buffer is quicker (about 11 ms for 50 samples against 26 ms for 300; timings vary from run to run), and the mean retrain time grows with the buffer in both orders;
  - the memory footprint grows with the buffer (14 KB for 50, 60 KB for 300).
- **Delay:** a model cannot predict before its first buffer is trained, so a larger buffer starts later (chunk 12 for 300, chunk 2 for 50).
- **Accuracy needs care.**
  - About 69% of the houses have no air conditioning, and in the last chunks (the cheaper houses of the price-sorted file) about 87% have none.
  - In file order, the largest buffer only ties "always answer no" on those chunks (0.873), and the smaller buffers are below it.
  - In shuffled order, only the buffer of 300 clearly beats the baseline (0.779 against 0.672).
  - The demo's AUC for `yes` agrees: 0.606 for a buffer of 100 and 0.511 for 300, close to guessing (0.5).
  - So on this small data set, how many samples each tree sees matters more than how often it retrains. The signal is weak, and the comparison covers only 244 samples, so differences of a few points are within noise.

## Tests

- **Counts**: 185 tests in `tests_stream/` cover **100% of the lines (909 statements) and 100% of the branches (268)** of `numcompute_stream`. Together with the 627 Assignment 2.1 tests, all 812 pass, and the 2.1 coverage gate (100% of `numcompute`) still passes.
- **What they check**:
  - streaming results against the 2.1 batch results: every metric and every `average`, the scalers, imputer and encoder, and the running mean and variance against the 2.1 `StreamingStats`;
  - the vectorised split search, prediction and category search against simple loops, and the tree against the course's own tree code;
  - buffered retraining: the buffer fills, retrains at `buffer_size`, is cleared, and `predict` uses the latest tree in between;
  - the trainer's AUC against a hand-written test-then-train loop with `predict_proba`;
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
  - **a rejected call changes nothing** (tree, encoder, statistics, confusion matrix, rolling metrics);
  - edge cases: NaN features, labels and scores; infinity; constant and all-missing columns; empty chunks (also before the first data and inside a pipeline) and chunks bigger than the buffer; new classes or categories mid-stream; wrong shapes, text input, out-of-range columns, unfitted models and invalid settings.
- **Without matplotlib**: the plotting tests are skipped if matplotlib is not installed. The 2.1 `requirements.txt` lists only NumPy.

## Debugging log

Each problem was found by a test, a measurement or a review, diagnosed, fixed, and then pinned by a test. For defects 4, 6 and 10-19 the guarding test was also run against the code without the fix and confirmed to fail.

| # | What went wrong | Diagnosis | Fix | Test that now guards it |
|---|---|---|---|---|
| 1 | The tree grew leaf by leaf, not by buffered retraining as the specification asks | Found when checking the code against the specification | Rewrote `tree.py` with the course's recursive build and a buffer | `test_partial_fit_buffers_until_buffer_size_then_retrains_and_clears` |
| 2 | `StreamTrainer` tried to score a chunk before any model existed | A model without `is_fitted` was treated as ready | `model_ready` also needs one `fit_chunk` first | `test_models_without_a_buffer_are_logged_too` |
| 3 | `StandardScaler` turned a constant column of 0.1 into -1 instead of 0 | Round-off left a tiny variance, and the scaler divided by almost zero | Detect constant columns exactly (min == max), as 2.1 does | `test_constant_and_all_missing_columns_match_the_2_1_scaler` |
| 4 | `ConfusionMatrix` changed its labels before rejecting NaN input | The validation came after the state update | Validate first | `test_a_rejected_update_leaves_the_confusion_matrix_unchanged` |
| 5 | Precision, recall and F1 repeated the 2.1 averaging logic | Found by comparing 2.2 functions with 2.1 functions | Reuse the 2.1 helper (`binary`, `macro`, `micro` now match 2.1) | `test_streaming_scores_equal_the_2_1_batch_scores` |
| 6 | A threshold between values near +1.7e308 and -1.7e308 raised an overflow warning | `high - low` overflows, and only a later guard hid the result | Compute `low / 2 + high / 2`, and keep the guard for neighbouring floats | `test_extreme_values_cannot_overflow_the_threshold` |
| 7 | All-equal features still paid for the cumulative class counts | Measured: 171 ms for 200,000 x 5 constant values | Skip a constant feature right after its sort (6 ms; timings vary by machine) | `test_all_equal_features_are_cheap_and_give_one_leaf` (checks the result and a generous time bound) |
| 8 | The integer check for settings was repeated in 8 places with different messages | The duplicate audit flagged two near-identical constructors | One shared `check_int_at_least` | `test_integer_settings_are_checked_the_same_way_everywhere` |
| 9 | `OneHotEncoder.partial_fit` looped over samples in Python (two loops), against the specification's vectorisation rule, and accepted a column mixing text and numbers that 2.1 rejects | A loop audit of the package: every `for` and comprehension, and what it iterates over | One `np.unique(..., return_index=True)` per column keeps arrival order; missing values are found with vectorised tests | `test_vectorised_category_search_matches_a_loop_in_order_of_arrival`, `test_a_column_mixing_text_and_numbers_is_rejected_like_in_2_1` |
| 10 | The first vectorised version cut "green" to "gree" | An existing test failed: the first chunk fixed the text width (`<U4`), and casting new categories to it cut them short | Let `np.concatenate` widen the type | `test_text_categories_longer_than_the_first_chunk_are_not_cut` (fails with the bug put back) |
| 11 | After a rejected `fit`, a trained tree forgot `n_features_in_`, so `predict` accepted inputs of any width; a rejected `partial_fit` could fix the width or add classes | Edge-case review: the checks also changed the state | Check everything first, then change the state; the first non-empty chunk fixes the width | `test_a_rejected_call_leaves_the_tree_unchanged`, `test_empty_chunks_and_bad_input` |
| 12 | `Pipeline.partial_fit` crashed on an empty first chunk (`NotFittedError` from the scaler) | A transformer skips an empty chunk, then cannot transform it | Skip empty chunks in the pipeline | `test_empty_chunks_are_skipped_even_before_the_first_data` |
| 13 | `StreamTrainer` gave `AUC` the predicted 0/1 labels, not probabilities (0.775 instead of 0.868 on the test stream) | The trainer passed `predict` output to every metric | Metrics marked `needs_scores` get the `predict_proba` column of their `pos_label`; metrics are checked when the trainer is created | `test_auc_is_given_probabilities_not_labels`, `test_metrics_are_checked_when_the_trainer_is_created` |
| 14 | One empty chunk marked the model as trained, so the next chunk was scored before any learning; `metrics={}` with `verbose` crashed (`StopIteration`) | `_trained` was set by any `fit_chunk`; the log line assumed a first metric | Only a non-empty chunk counts as training; the log line works without metrics | `test_empty_chunks_before_the_first_data_are_not_mistaken_for_training`, `test_verbose_log_without_metrics` |
| 15 | `load_csv(target_column=3)` on a 3-column file silently used column 0 as the label | The index was taken modulo the number of columns | Reject indices outside the header | `test_target_column_outside_the_file_is_an_error` |
| 16 | `StreamingStats` accepted infinity: the std and median became NaN and the histogram edges broke | No check of the input values | Reuse the 2.1 numeric check (text: `TypeError`; infinity: `ValueError` naming row and column) | `test_text_and_infinity_are_rejected_and_change_nothing` |
| 17 | `OneHotEncoder(columns=...)` was not checked: a repeated column was encoded twice and a negative index broke the pass-through columns | `columns` was used as given | Check the indices on the first chunk (integers inside X, each once; negative counts from the end) | `test_encoder_columns_are_checked_before_anything_is_learned` |
| 18 | A chunk rejected part-way through `OneHotEncoder.partial_fit` (a column mixing text and numbers) had already updated the columns before it; a bad first chunk left the encoder looking fitted | Columns were updated one at a time, inside the checking loop | Find every column's new categories first, then update them all at once | `test_a_chunk_rejected_part_way_leaves_the_encoder_unchanged` |
| 19 | `RollingMetric.update` stored a NaN label, so `result()` failed until it left the window; text labels after numbers silently turned every stored label into text | The window was only checked when `result()` ran | Check each chunk with a fresh copy of the wrapped metric, and reject a switch between text and numbers, before storing it | `test_a_rejected_update_leaves_the_rolling_window_unchanged` |

## Design decisions and known limits

Every limit below is either fixed above or a deliberate choice, with the reason.

- **The buffered tree forgets older data.** The specification says to retrain on the buffer and clear it. The benchmark shows the cost: a small buffer sees few samples per tree. The benefit is bounded memory and a tree that follows a changing stream.
- **A rejected input changes nothing.** The tree, every transformer, `StreamingStats` and every metric check a chunk completely before they change any state, so one bad chunk cannot corrupt a long-running stream.
- **A pipeline is not transactional.** It updates its steps in order, so if the model rejects a chunk's labels (for example NaN labels), the transformers before it have already counted that chunk's (valid) features. Undoing that would mean copying every step, including the tree's buffer, for every chunk.
- **Quantiles are approximate once the stream is longer than `sample_size`.** They come from a random sample (reservoir sampling), so memory does not grow. They are exact until that many rows have been seen. Raise `sample_size` for more accuracy.
- **`AUC` is approximate.** Scores are counted in 1,000 bins, so memory stays small. Scores in the same bin count as ties, so the error is at most about 1/1000 of the score range (a test checks it against the pairwise definition).
- **Loops over features, never over samples.** `best_split`, the histograms and the encoder loop over the columns (a handful), and everything inside is NumPy. The recursion in `_build` is as deep as `max_depth`.
- **The `Imputer` supports `mean` and `constant` in a stream.** `median` and `most_frequent` need all the data at once, so they raise a clear error in `partial_fit` and work with `fit`.
- **Text columns must be encoded before the tree.** The tree rejects text and NaN with a message that names the fix (`OneHotEncoder`, `Imputer`).
- **Floating-point limits.** Scaling a column that holds both 1e18 and values near 1e9 can hide a difference of 1, because `x - mean` loses it in float64. This is the nature of 64-bit floats, not a bug. The tree itself, used on the raw column, keeps such values apart (tested).
