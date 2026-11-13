"""Robustness tests: numerical stability, chunking, all-equal data and bounded memory.

These are the properties a stream must keep however the data arrives: results must not depend on how it is cut
into chunks, must stay exact on widely spread or offset values, must stay fast on all-equal data, and the memory
of the model must stay bounded.
"""

import time
import warnings

import numpy as np
import pytest

from numcompute import preprocessing as base21
from numcompute_stream.io import iter_chunks
from numcompute_stream.metrics import AUC, F1, Accuracy, ConfusionMatrix, RollingMetric
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stats import StreamingStats
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier, best_split


# ---------------------------------------------------- chunking must not matter
@pytest.mark.parametrize("chunk_size", [1, 7, 64, 300])
def test_running_statistics_do_not_depend_on_the_chunk_size(chunk_size):
    X = np.random.default_rng(1).normal(50.0, 4.0, size=(300, 3))
    stats = StreamingStats()
    for chunk in iter_chunks(X, np.zeros(300), chunk_size):
        stats.update_stats(chunk[0])
    np.testing.assert_allclose(stats.mean, X.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(stats.var, X.var(axis=0), rtol=1e-10)
    np.testing.assert_array_equal(stats.min, X.min(axis=0))


@pytest.mark.parametrize("chunk_size", [1, 13, 500])
def test_metrics_do_not_depend_on_the_chunk_size(chunk_size):
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 3, 500)
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, 3, 500))
    results = []
    for metric in (Accuracy(), F1(), ConfusionMatrix()):
        for y_t, y_p in iter_chunks(y_true, y_pred, chunk_size):
            metric.update(y_t, y_p)
        results.append(metric.result())
    assert results[0] == Accuracy().update(y_true, y_pred).result()
    assert results[1] == pytest.approx(F1().update(y_true, y_pred).result())
    np.testing.assert_array_equal(results[2][0], ConfusionMatrix().update(y_true, y_pred).result()[0])


@pytest.mark.parametrize("chunk_size", [1, 5, 10, 50])
def test_buffered_tree_is_the_same_for_any_chunk_size_that_divides_buffer_size(chunk_size):
    X = np.random.default_rng(3).uniform(0, 10, size=(200, 3))
    y = ((X[:, 0] > 5) ^ (X[:, 1] > 6)).astype(int)
    tree = DecisionTreeClassifier(max_depth=4, buffer_size=50)
    for X_chunk, y_chunk in iter_chunks(X, y, chunk_size):
        tree.partial_fit(X_chunk, y_chunk)
    reference = DecisionTreeClassifier(max_depth=4).fit(X[150:], y[150:])  # the last full buffer
    assert tree.n_retrains_ == 4
    np.testing.assert_array_equal(tree.predict(X), reference.predict(X))


def test_the_buffer_stays_below_buffer_size_after_every_chunk():
    """Bounded memory: the buffer is cleared as soon as it reaches buffer_size, whatever the chunk size."""
    X = np.random.default_rng(4).normal(size=(2000, 2))
    y = (X[:, 0] > 0).astype(int)
    tree = DecisionTreeClassifier(buffer_size=120)
    for X_chunk, y_chunk in iter_chunks(X, y, 37):
        tree.partial_fit(X_chunk, y_chunk)
        assert tree.n_buffered_ < 120
    assert tree.n_retrains_ == 13  # every 4 chunks (4 * 37 = 148 >= 120 samples): 54 full chunks -> 13 retrains


def test_streaming_is_deterministic():
    X = np.random.default_rng(5).normal(size=(400, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    runs = [StreamTrainer(Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(buffer_size=100))]))
            .run(iter_chunks(X, y, 25)) for _ in range(2)]
    np.testing.assert_array_equal(runs[0].get("accuracy"), runs[1].get("accuracy"))


# ------------------------------------------------- widely spread and offset values
def test_variance_is_exact_for_a_large_offset_fed_one_sample_at_a_time():
    x = 1e9 + np.arange(1000.0)  # E[x^2] - E[x]^2 loses 5 units of variance here
    stats = StreamingStats()
    for value in x:
        stats.update_stats(np.array([[value]]))
    assert stats.var[0] == pytest.approx(np.var(x), rel=1e-12)
    assert abs((x ** 2).mean() - x.mean() ** 2 - np.var(x)) > 1  # the naive formula would be wrong


def test_imputer_mean_is_exact_for_a_large_offset():
    imputer = Imputer()
    for value in 1e9 + np.arange(1000.0):
        imputer.partial_fit(np.array([[value]]))
    assert imputer.statistics_[0] == pytest.approx(1e9 + 499.5, rel=1e-15)


def test_scaler_on_columns_with_very_different_scales_matches_the_2_1_scaler():
    rng = np.random.default_rng(6)
    X = np.column_stack([1e9 + rng.normal(size=400), 1e-9 * rng.normal(size=400), rng.normal(size=400)])
    scaler = StandardScaler()
    for row in X:
        scaler.partial_fit(row.reshape(1, -1))
    np.testing.assert_allclose(scaler.transform(X), base21.StandardScaler().fit(X).transform(X), atol=1e-5)


def test_tree_separates_1e9_from_1e9_plus_1_in_a_widely_spread_column():
    X = np.array([[-1e18], [0.0], [1e9], [1e9 + 1], [1e18]] * 4)
    y = np.array([0, 0, 0, 1, 1] * 4)
    feature, threshold, _ = best_split(X[2:4], y[2:4], n_classes=2)
    assert feature == 0 and 1e9 < threshold < 1e9 + 1
    np.testing.assert_array_equal(DecisionTreeClassifier(max_depth=5).fit(X, y).predict(X[:5]), [0, 0, 0, 1, 1])


def test_extreme_values_cannot_overflow_the_threshold():
    X, y = np.array([[-1.7e308], [1.7e308]]), np.array([0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an overflow RuntimeWarning must not be raised, even if it is hidden later
        feature, threshold, _ = best_split(X, y, n_classes=2)
    assert feature == 0 and np.isfinite(threshold) and -1.7e308 <= threshold < 1.7e308
    assert DecisionTreeClassifier().fit(X, y).predict(X).tolist() == [0, 1]


# ------------------------------------------------------------------ all-equal data
def test_all_equal_features_are_cheap_and_give_one_leaf():
    X, y = np.ones((100_000, 5)), np.arange(100_000) % 2  # nothing to ask, mixed labels
    start = time.perf_counter()
    tree = DecisionTreeClassifier().fit(X, y)
    assert time.perf_counter() - start < 2.0  # about 0.1 s; the bound is generous to avoid flaky failures
    assert tree.get_n_leaves() == 1 and best_split(X, y, 2)[0] is None


def test_all_equal_labels_and_one_sample_chunks():
    tree = DecisionTreeClassifier(buffer_size=3)
    for i in range(7):  # one-sample chunks, a single class
        tree.partial_fit(np.full((1, 2), float(i)), [4])
    assert tree.n_retrains_ == 2 and tree.n_buffered_ == 1 and tree.predict([[0.0, 0.0]]).tolist() == [4]


def test_ties_pick_the_lowest_feature_and_the_lowest_threshold():
    X = np.column_stack([np.arange(6.0), np.arange(6.0)])  # two identical features
    y = np.array([0, 0, 0, 1, 1, 1])
    assert best_split(X, y, 2)[:2] == (0, 2.5)  # feature 0, not feature 1
    X2 = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert best_split(X2, np.array([0, 1, 0, 1]), 2)[1] == 0.5  # equally good cuts: the lowest wins


# ---------------------------------------------------- one rule for every integer setting
@pytest.mark.parametrize("build, setting", [
    (lambda v: StreamingStats(bins=v), "bins"), (lambda v: StreamingStats(sample_size=v), "sample_size"),
    (lambda v: StreamingStats(window=v), "window"), (lambda v: AUC(n_bins=v), "n_bins"),
    (lambda v: RollingMetric(Accuracy(), window=v), "window"),
    (lambda v: OneHotEncoder(max_categories=v), "max_categories"),
    (lambda v: DecisionTreeClassifier(buffer_size=v), "buffer_size"),
    (lambda v: DecisionTreeClassifier(max_depth=v), "max_depth"),
])
@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "7"])
def test_integer_settings_are_checked_the_same_way_everywhere(build, setting, bad):
    with pytest.raises(ValueError, match=f"{setting} must be .*integer >= 1"):
        build(bad)


def test_optional_integer_settings_accept_none_and_numpy_integers():
    assert StreamingStats(window=None).window is None and StreamingStats(window=np.int64(3)).window == 3
    assert OneHotEncoder(max_categories=None).max_categories is None
    assert DecisionTreeClassifier(buffer_size=np.int32(5), min_samples_split=np.int64(2)).buffer_size == 5
