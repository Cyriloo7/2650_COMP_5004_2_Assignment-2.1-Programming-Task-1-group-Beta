"""Benchmark (a): loop-based vs vectorised versions of the core streaming computations.

Run from the repository root (like the Assignment 2.1 benchmarks):

    python -m benchmark_stream.stream_loop_vs_vectorised

The timing and checking reuse ``numcompute.benchmarking.compare`` from
Assignment 2.1: it raises instead of reporting a speed-up if the two versions
give different results. The loop versions are the course and 2.1 code: the 2.1
``StreamingStats`` (one value at a time), and the course tree's way of trying
one threshold at a time and predicting one sample at a time.
The table is printed and saved to benchmark_stream/stream_loop_vs_vectorised.md.
"""

import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # also works as: python benchmark_stream/<this file>

from numcompute.benchmarking import compare, format_results_table  # noqa: E402  Assignment 2.1
from numcompute.stats import StreamingStats as StreamingStats21    # noqa: E402  Assignment 2.1
from numcompute_stream.metrics import ConfusionMatrix               # noqa: E402
from numcompute_stream.stats import update_moments                  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier, best_split, impurity  # noqa: E402


# ------------------------------------------------------------ loop versions
def loop_running_moments(X):
    """The 2.1 StreamingStats: one accumulator per feature, updated one value at a time."""
    rows = []
    for column in X.T:
        accumulator = StreamingStats21()
        for value in column:
            accumulator.update(value)
        rows.append((accumulator.mean, accumulator.variance))
    return np.array(rows).T  # row 0: means, row 1: variances


def loop_best_split(x, y):
    """Try one threshold at a time, as the course tree's _gain does for each question."""
    parent = impurity(np.bincount(y, minlength=2))
    values = np.unique(x)
    best_threshold, best_gain = np.nan, 0.0
    for low, high in zip(values[:-1], values[1:]):
        threshold = low + (high - low) / 2
        left = x <= threshold
        after = (left.sum() * impurity(np.bincount(y[left], minlength=2))
                 + (~left).sum() * impurity(np.bincount(y[~left], minlength=2))) / len(y)
        if parent - after > best_gain:
            best_threshold, best_gain = threshold, parent - after
    return np.array([best_threshold, best_gain])


def loop_predict(tree, X):
    """The course tree's predict: walk down the tree one sample at a time."""
    labels = []
    for row in X:
        node = tree.root_
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        labels.append(node.label)
    return tree.classes_[np.array(labels)]


def loop_confusion_counts(y_true, y_pred):
    matrix = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


# ------------------------------------------------------ vectorised versions
def running_mean_and_variance(X):
    """The streaming update used by StreamingStats and StandardScaler, in 10 chunks."""
    count, mean, m2 = np.zeros(X.shape[1]), np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for chunk in np.array_split(X, 10):
        count, mean, m2 = update_moments(count, mean, m2, chunk)
    return np.vstack([mean, m2 / count])


def best_split_of_a_feature(x, y):
    """All thresholds at once from cumulative class counts."""
    _, threshold, gain = best_split(x.reshape(-1, 1), y, n_classes=2)
    return np.array([threshold, gain])


def tree_prediction(tree, X):
    """All samples move down the tree together."""
    return tree.predict(X)


def confusion_matrix_counts(y_true, y_pred):
    return ConfusionMatrix().update(y_true, y_pred).result()[0]


def main():
    rng = np.random.default_rng(0)
    results = []

    X = rng.normal(5.0, 2.0, size=(20_000, 4))
    results.append(compare(running_mean_and_variance, loop_running_moments, X, repeats=3))

    x = rng.normal(size=2_000)
    y = (x + rng.normal(scale=0.7, size=2_000) > 0).astype(int)
    results.append(compare(best_split_of_a_feature, loop_best_split, x, y, repeats=3))

    X_train = rng.uniform(0, 10, size=(5_000, 4))
    tree = DecisionTreeClassifier(max_depth=8).fit(X_train, (X_train[:, 0] + X_train[:, 1] > 10).astype(int))
    results.append(compare(tree_prediction, loop_predict, tree, rng.uniform(0, 10, size=(20_000, 4)), repeats=3))

    y_true, y_pred = rng.integers(0, 5, 100_000), rng.integers(0, 5, 100_000)
    results.append(compare(confusion_matrix_counts, loop_confusion_counts, y_true, y_pred, repeats=3))

    table = format_results_table(results)
    notes = (f"\n\nInputs: running moments 20,000 x 4 values; best split 2,000 values; tree prediction 20,000 samples "
             f"(tree of depth {tree.get_depth()}, {tree.get_n_leaves()} leaves); confusion matrix 100,000 labels.\n"
             f"Mean of 3 timed runs after 1 warm-up run; random seed 0. {date.today()}, Python "
             f"{platform.python_version()}, NumPy {np.__version__}.\n")
    print(table + notes)
    (Path(__file__).parent / "stream_loop_vs_vectorised.md").write_text(table + notes)


if __name__ == "__main__":
    main()
