# NumCompute

A small scientific-computing toolkit written with **plain Python and NumPy only** - no pandas, scikit-learn or other data or machine-learning libraries.

It covers loading and cleaning data, sorting and ranking, descriptive statistics, evaluation metrics, numerical gradients and a simple pipeline. It also includes benchmarks that compare vectorised NumPy code with ordinary Python loops.

Built for **COMP 5004 - Programming Task 1** by group Beta.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Conventions](#conventions)
- [Modules](#modules)
- [Performance](#performance)
- [Demo notebook](#demo-notebook)
- [Running the tests](#running-the-tests)
- [Project layout](#project-layout)
- [Team](#team)
- [Troubleshooting](#troubleshooting)

## Installation

You need **Python 3.10 or newer**. NumPy (2.0 or newer) is installed for you.

```bash
# 1. Get the code
git clone https://github.com/Cyriloo7/2650_COMP_5004_2_Assignment-2.1-Programming-Task-1-group-Beta.git
cd 2650_COMP_5004_2_Assignment-2.1-Programming-Task-1-group-Beta

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
source .venv/bin/activate         # macOS / Linux

# 3. Install the package, plus the tools for running the tests
python -m pip install -e ".[dev]"
```

- `-e` means "editable": changes you make to the code take effect straight away.
- To use the toolkit without the test tools, run `python -m pip install -e .` instead. This installs only NumPy.
- Check that it worked: `python -c "import numcompute; print('ok')"`

## Quickstart

```python
import numpy as np

from numcompute.metrics import accuracy, auc, f1, roc_curve
from numcompute.rank import rank
from numcompute.stats import describe, quantile
from numcompute.utils import euclidean_distance, topk_indices

# 1. Summarise a table that has a missing value (NaN is ignored).
X = np.array([[1.0, 10.0],
              [2.0, np.nan],
              [3.0, 30.0],
              [4.0, 40.0]])
print(np.round(describe(X, axis=0)["mean"], 2))   # one mean per column
print(quantile(X, 0.5, axis=0))                   # one median per column

# 2. Rank values. Tied values share the average of their ranks.
print(rank([10, 20, 20, 30]))

# 3. Nearest neighbours: build a distance matrix, then keep the 2 closest per row.
points = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [0.4, 0.0]])
distances = euclidean_distance(points, points)
print(topk_indices(distances, 2, largest=False))  # each point, then its nearest neighbour

# 4. Score a classifier.
y_true = np.array([0, 1, 1, 0, 1, 0])
scores = np.array([0.2, 0.9, 0.6, 0.6, 0.4, 0.1])
y_pred = (scores >= 0.5).astype(int)
print(round(accuracy(y_true, y_pred), 3), round(f1(y_true, y_pred), 3))
fpr, tpr, _ = roc_curve(y_true, scores)
print(round(auc(fpr, tpr), 3))
```

Output:

```text
[ 2.5  26.67]
[ 2.5 30. ]
[1.  2.5 2.5 4. ]
[[0 3]
 [1 3]
 [2 1]
 [3 0]]
0.667 0.667
0.833
```

For the full workflow - loading a CSV, cleaning it and trying every module - see the [demo notebook](#demo-notebook).

## Conventions

These rules hold across the whole toolkit, so you only need to learn them once.

- **Shapes.** A data table `X` has shape `(n_samples, n_features)`: one row per sample, one column per feature. Labels and scores (`y_true`, `y_pred`, `y_score`) are 1-D arrays with one entry per sample. Distance functions take two tables, `A` of shape `(n, d)` and `B` of shape `(m, d)`.
- **`axis`.** `axis=0` goes down the rows and gives one result per **column** (feature). `axis=1` gives one result per **row** (sample). Negative numbers count from the end, so `-1` is the last axis. Where a function allows `axis=None`, it uses every value. Defaults: `describe` uses `axis=0`; `softmax`, `log_softmax` and `topk_indices` use `axis=-1`; `quantile`, `percentile` and `logsumexp` use `axis=None`.
- **Missing values.** `NaN` means "missing". Statistics functions ignore it, `rank` and `topk_indices` put it last, and functions that cannot work with it raise a `ValueError`. The tables below give the exact rule for each function.
- **Ties.** Equal values are handled by a stated rule: ranks use `average`, `dense` or `ordinal`; `topk_indices` takes the lower index first; ROC curves merge tied scores into one point.
- **Errors.** Wrong shapes, empty arrays and invalid options raise a `ValueError` with a message saying what is wrong.
- **fit / transform.** Preprocessing classes learn from data with `fit(X)`, apply what they learned with `transform(X)`, and `fit_transform(X)` does both in one step.

## Modules

Each module has a short summary and a table of its main functions. The last column gives the NaN and tie rule.

### io

Reads CSV files into NumPy arrays. Blank cells become `NaN` instead of causing an error, the delimiter can be changed (comma, tab and so on), and very large files can be read in chunks so they never have to fit in memory at once.

| Function | What it does | NaN and ties |
|---|---|---|
| `read_csv(path, ...)` | Load a whole CSV file; returns `(X, column_names)`. | Blank or NA cells become `NaN`. |
| `read_csv_chunks(path, chunksize, ...)` | Read a large CSV one chunk at a time. | Same as `read_csv`. |

### preprocessing

Cleans and rescales features before analysis. The imputer fills in missing values, the two scalers put every column on a common scale, and the encoder turns category labels into 0/1 columns. Each class follows the `fit` / `transform` pattern, so what it learns from one dataset can be applied to another.

| Class | What it does | NaN and ties |
|---|---|---|
| `SimpleImputer(...)` | Fill missing values, for example with the column mean. | Replaces `NaN`. |
| `StandardScaler()` | Give each column mean 0 and standard deviation 1. | Fill `NaN` first with the imputer. |
| `MinMaxScaler(...)` | Rescale each column to a range such as 0 to 1. | Fill `NaN` first with the imputer. |
| `OneHotEncoder()` | Turn each category into its own 0/1 column. | n/a |

### sort_search

Sorting, selecting and searching. `stable_sort` and `multi_key_sort` put values in order (by several columns if needed), `topk` finds the k largest or smallest values without sorting everything, `quickselect` is a hand-written version of the same idea, and `binary_search` finds where a value belongs in sorted data.

| Function | What it does | NaN and ties |
|---|---|---|
| `stable_sort(...)` | Sort values; equal values keep their original order. | Ties keep their original order. |
| `multi_key_sort(...)` | Sort rows by several columns. | A tie in one column is broken by the next column. |
| `topk(...)` | The k largest or smallest values and their indices. | See the function's docstring. |
| `quickselect(...)` | The k-th smallest value, written by hand. | n/a |
| `binary_search(...)` | Where a value belongs in sorted data, and whether it is there. | With duplicates, returns the first position. |

### rank

Ranking with a clear rule for ties, and percentiles. `rank` offers three tie methods, and `percentile` returns the value at a given percentile.

| Function | What it does | NaN and ties |
|---|---|---|
| `rank(data, method="average")` | Rank of each value; 1 is the smallest. | Ties follow `method`: `average`, `dense` or `ordinal`. NaN ranks last (all NaNs share one rank). |
| `percentile(data, q, interpolation="linear", axis=None)` | Value at a percentile; `q` is between 0 and 100. | NaN is ignored. |

### stats

Summary statistics. `describe` gives the mean, median, standard deviation, minimum and maximum along an axis. `StreamingStats` keeps a running mean and variance while values arrive one at a time, so the whole dataset never has to be in memory. `histogram` and `quantile` complete the set.

| Function | What it does | NaN and ties |
|---|---|---|
| `describe(X, axis=0)` | Mean, median, std, min and max along an axis, returned as a dict. | NaN is ignored. A column that is all NaN gives NaN. |
| `StreamingStats` (also named `WelfordAccumulator`) | Running mean and variance: `update(x)` adds a value, `merge(other)` combines two accumulators. | NaN values are skipped and not counted. With no values, `mean`, `variance` and `std` raise `ValueError`. |
| `histogram(data, bins=10, range=None)` | Number of values in each bin. | NaN is dropped before binning. |
| `quantile(data, q, interpolation="linear", axis=None)` | Value at a quantile; `q` is between 0 and 1. | NaN is ignored. |

### metrics

Scores for judging predictions: a confusion matrix, accuracy, precision, recall and F1 for classification, mean squared error for regression, and an ROC curve with its area (AUC) for binary classifiers.

| Function | What it does | NaN and ties |
|---|---|---|
| `confusion_matrix(y_true, y_pred, labels=None)` | Counts of (true, predicted) pairs; returns `(matrix, labels)`. | NaN labels raise `ValueError`. |
| `accuracy(y_true, y_pred)` | Fraction of predictions that are correct. | n/a |
| `precision`, `recall`, `f1` (arguments `y_true, y_pred, average="binary", pos_label=1, labels=None`) | `average` is `binary`, `macro` or `micro`. | A class with no predictions or no samples scores `0.0`, never NaN. `binary` needs at most 2 classes, one of them `pos_label`. |
| `mse(y_true, y_pred)` | Mean squared error. | n/a |
| `roc_curve(y_true, y_score)` | ROC curve for 0/1 labels; returns `(fpr, tpr, thresholds)`. | Tied scores are merged into one point, so row order does not matter. NaN scores are not checked (they rank last), so remove them first. |
| `auc(fpr, tpr)` | Area under a curve. | `fpr` must be monotonic with at least 2 points. |

### optim

Numerical derivatives. `grad` estimates the gradient of a function using finite differences (forward or central), and `jacobian` does the same for functions that return a vector.

| Function | What it does | NaN and ties |
|---|---|---|
| `grad(f, x, h=1e-5, method="central")` | Gradient of a scalar function at `x`. | n/a |
| `jacobian(F, x, ...)` | Jacobian of a vector function at `x`, with shape `(m, n)`. | n/a |

### pipeline

Chains steps together. A `Pipeline` runs steps one after another (for example impute, then scale). `FeatureUnion` runs steps side by side on the same input and joins the results column by column.

| Class | What it does | NaN and ties |
|---|---|---|
| `Pipeline(steps)` | Run steps in order with `fit`, `transform`, `fit_transform` and `predict`. | n/a |
| `FeatureUnion(...)` | Run steps in parallel on the same input and join their outputs side by side. | n/a |

### utils

Shared building blocks used by the other modules: pairwise distances, activation functions, numerically stable `softmax` and `logsumexp`, safe division, a top-k helper for nearest neighbours, and a batch iterator.

| Function | What it does | NaN and ties |
|---|---|---|
| `euclidean_distance(A, B)`, `manhattan_distance(A, B)`, `cosine_distance(A, B)` | Distance from every row of `A` to every row of `B`; result has shape `(n, m)`. | A NaN only affects the distances of its own point. |
| `sigmoid(x)`, `relu(x)` | Activation functions. | NaN stays NaN. |
| `softmax(x, axis=-1)`, `log_softmax(x, axis=-1)`, `logsumexp(x, axis=None)` | Stable versions that do not overflow on large values. | `-inf` and `+inf` are handled without warnings. A slice that is all `-inf` gives NaN in `softmax` and `log_softmax`, and `-inf` in `logsumexp`. Empty input raises `ValueError`. |
| `safe_divide(num, denom, fill=0.0)` | Division that returns `fill` where the denominator is 0. | NaN stays NaN. |
| `topk_indices(X, k, axis=-1, largest=True)` | Indices of the k largest values; use `largest=False` for the smallest, such as nearest neighbours. | Ties: lower index first. NaN ranks last. |
| `batch_iterator(X, batch_size)` | Yield the data in consecutive batches. | n/a |

### benchmarking

Timing tools. `benchmark` times one function, `compare` times a vectorised and a loop version of the same computation and reports the speedup, and `format_results_table` prints the results as a Markdown table.

| Function | What it does | NaN and ties |
|---|---|---|
| `benchmark(func, *args, repeats=10, warmup=1)` | Time one function; returns mean, std, min, max and the result. | n/a |
| `compare(vectorized_fn, loop_fn, *args, repeats=10)` | Time two versions and report the speedup. | Raises if the two results differ in shape or value. Matching NaNs count as equal. |
| `format_results_table(results)` | Turn `compare` results into a Markdown table. | n/a |

## Performance

Vectorised code does the looping inside NumPy instead of in Python. The table compares NumCompute functions with plain-Python-loop versions of the same computation, run on the Housing dataset (`Dataset/Housing.csv`: 545 houses, 13 columns).

**Measured on**

- Python 3.14.4 and NumPy 2.5.4
- Linux (x86_64), 1 CPU core
- Each time is the mean of 10 runs (after 1 warm-up run), in seconds

| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| describe, 545 houses x 6 columns | 0.0005014 | 0.0005376 | 1.07x |
| histogram, price in 10 bins | 8.183e-05 | 6.574e-05 | 0.80x |
| rank (average), price with ties | 5.567e-05 | 0.01213 | 217.91x |
| euclidean_distance, 100 houses vs 445 | 0.0006939 | 0.08437 | 121.58x |
| topk_indices, 5 nearest of 445, per house | 0.001144 | 0.004981 | 4.36x |
| confusion_matrix, 545 houses, classes found | 9.523e-05 | 2.768e-05 | 0.29x |
| confusion_matrix, 545 houses, labels= given | 6.249e-05 | 2.494e-05 | 0.40x |
| roc_curve + auc, price vs air conditioning | 8.262e-05 | 0.001445 | 17.50x |

How to read it:

- **Speedup** is the loop time divided by the vectorised time. A value below 1x means the loop was faster.
- The biggest gains are where the loop version compares every house with many others: ranking, pairwise distances and AUC.
- On only 545 rows, the one-pass summaries (`describe`, `histogram`, `confusion_matrix`) are not much faster than a loop (roughly 0.3x to 2x), and `confusion_matrix` is slower, because every NumPy call has a fixed set-up cost. With about ten times as many rows (the same houses repeated), `describe` and `histogram` are already several times faster.
- `confusion_matrix` has to find the class labels itself, which involves a sort. Passing `labels=` skips that step and is faster.
- Every pair is checked to give the same answer before its speedup is reported, so a fast but wrong version cannot appear in the table.
- Exact times differ between machines and between runs. The pattern is what matters.

## Demo notebook

`demo/quickstart.ipynb` walks through the toolkit on a small dataset: reading a CSV with missing values, cleaning it, sorting and searching, ranking, statistics, gradients and a benchmark.

1. Install the package (see [Installation](#installation)). The notebook must use the same environment.
2. Install Jupyter, which is only needed for the notebook: `python -m pip install notebook`
3. From the repository root, open it: `python -m notebook demo/quickstart.ipynb`
4. In the notebook, choose **Run > Run All Cells**.

In VS Code, open `demo/quickstart.ipynb`, click **Select Kernel** (top right), pick the `.venv` interpreter, and choose **Run All**.

Open the notebook from inside the cloned repository so that any relative paths it uses (for example to the `Dataset/` folder) resolve correctly.

## Running the tests

```bash
python -m pytest tests/                    # run every test
python -m pytest tests/ -v                 # one line per test
python -m pytest tests/test_stats.py       # one module only

# with coverage (needs the [dev] install)
python -m pytest tests/ --cov=numcompute --cov-branch --cov-report=term-missing

# the project's quality gate: fail unless coverage is 100%
python -m pytest tests/ --cov=numcompute --cov-branch --cov-fail-under=100
```

Use `python -m pytest` rather than plain `pytest`: it makes sure the tests run in the same environment you installed the package into. Test files are named `test_<module>.py` and live in `tests/`.

## Project layout

```
.
├── numcompute/            the package, one file per module
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/                 unit tests, test_<module>.py
├── demo/
│   └── quickstart.ipynb   end-to-end demo notebook
├── Dataset/               Housing.csv, the sample dataset
├── README.md
└── pyproject.toml         packaging and dependencies
```

## Team

| Team member | Modules |
|---|---|
| Cyril ([@Cyriloo7](https://github.com/Cyriloo7)) | `stats.py`, `metrics.py`, `utils.py`, `benchmarking.py` |
| *Teammate 2 (add name)* | `sort_search.py`, `rank.py`, `optim.py` |
| Binitta ([@binittavar](https://github.com/binittavar)) | `io.py`, `preprocessing.py`, `pipeline.py` |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'numcompute'`** - the package is not installed in the environment you are running. From the repository root, run `python -m pip install -e ".[dev]"`.
- **`pytest: error: unrecognized arguments: --cov`** - the test tools are missing. Run `python -m pip install -e ".[dev]"`.
- **pytest says "no tests ran"** - test files must be named `test_<name>.py` (underscore, not hyphen) and live in `tests/`.
- **`AttributeError: module 'numpy' has no attribute 'trapezoid'`** - your NumPy is older than 2.0. Run `python -m pip install -U "numpy>=2.0"`.
- **PowerShell will not run `Activate.ps1`** - run `Set-ExecutionPolicy -Scope Process RemoteSigned` first, or use Command Prompt and `.venv\Scripts\activate.bat`.
