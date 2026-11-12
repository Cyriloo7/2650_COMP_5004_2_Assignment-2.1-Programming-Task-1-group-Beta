"""Tests for stats.py: streaming statistics must match NumPy on all the data."""

import numpy as np
import pytest

from numcompute.preprocessing import NotFittedError
from numcompute.stats import StreamingStats as StreamingStats21
from numcompute_stream.stats import StreamingStats, safe_divide, update_min_max, update_moments


@pytest.mark.parametrize("n_chunks", [1, 5, 40])
def test_mean_variance_min_max_match_numpy(n_chunks):
    rng = np.random.default_rng(1)
    X = rng.normal(10, 2, size=(400, 3))
    X[rng.random(X.shape) < 0.05] = np.nan
    stats = StreamingStats()
    for chunk in np.array_split(X, n_chunks):
        stats.update_stats(chunk)
    np.testing.assert_allclose(stats.mean, np.nanmean(X, axis=0))
    np.testing.assert_allclose(stats.var, np.nanvar(X, axis=0))
    np.testing.assert_allclose(stats.min, np.nanmin(X, axis=0))
    np.testing.assert_allclose(stats.max, np.nanmax(X, axis=0))
    np.testing.assert_array_equal(stats.n_missing_, np.isnan(X).sum(axis=0))


def test_variance_is_numerically_stable():
    x = 1e9 + np.random.default_rng(2).normal(0, 0.01, size=5000)  # huge offset, tiny spread
    stats = StreamingStats()
    for chunk in np.array_split(x, 50):
        stats.update_stats(chunk)
    assert stats.var[0] == pytest.approx(np.var(x), rel=1e-3)


def test_quantiles_are_exact_until_the_sample_is_full_then_close():
    x = np.random.default_rng(3).normal(size=5000)
    small = StreamingStats(sample_size=1000)
    for chunk in np.array_split(x[:800], 4):
        small.update_stats(chunk)
    np.testing.assert_allclose(small.quantile([0.1, 0.5, 0.9])[:, 0], np.quantile(x[:800], [0.1, 0.5, 0.9]))
    big = StreamingStats(sample_size=1000, random_state=0)
    for chunk in np.array_split(x, 25):
        big.update_stats(chunk)
    assert abs(big.quantile(0.5)[0] - np.median(x)) < 0.1


def test_histogram_and_sliding_window():
    chunks = [np.full(4, 1.0), np.full(3, 5.0), np.full(2, 9.0)]
    windowed = StreamingStats(bins=5, hist_range=(0.0, 10.0), window=2)
    everything = StreamingStats(bins=5, hist_range=(0.0, 10.0))
    for chunk in chunks:
        windowed.update_stats(chunk)
        everything.update_stats(chunk)
    counts, edges = windowed.histogram()
    np.testing.assert_array_equal(edges[0], np.linspace(0, 10, 6))
    np.testing.assert_array_equal(counts[0], [0, 0, 3, 0, 2])  # only the last two chunks
    expected = np.histogram(np.concatenate(chunks), bins=5, range=(0.0, 10.0))[0]
    np.testing.assert_array_equal(everything.histogram()[0][0], expected)


def test_values_outside_the_histogram_range_are_counted():
    stats = StreamingStats(bins=2).update_stats([0.0, 1.0]).update_stats([5.0, -3.0])
    assert stats.out_of_range_[0] == 2 and stats.histogram()[0][0].sum() == 2


def test_empty_chunks_wrong_widths_and_no_data():
    stats = StreamingStats()
    with pytest.raises(NotFittedError):
        _ = stats.mean
    stats.update_stats(np.empty((0, 2)))  # an empty chunk changes nothing
    assert stats.n_features_in_ is None
    stats.update_stats(np.ones((3, 2)))
    with pytest.raises(ValueError):
        stats.update_stats(np.ones((3, 3)))
    assert set(stats.summary()) == {"count", "missing", "mean", "std", "min", "median", "max"}


def test_chunk_merge_equals_the_2_1_streaming_stats():
    x = np.random.default_rng(6).normal(3, 2, size=500)
    x[::17] = np.nan
    accumulator = StreamingStats21()
    for value in x:  # the 2.1 version: one value at a time
        accumulator.update(value)
    count, mean, m2 = np.zeros(1), np.zeros(1), np.zeros(1)
    for chunk in np.array_split(x.reshape(-1, 1), 9):  # the 2.2 version: whole chunks
        count, mean, m2 = update_moments(count, mean, m2, chunk)
    assert count[0] == accumulator.count
    assert mean[0] == pytest.approx(accumulator.mean) and m2[0] / count[0] == pytest.approx(accumulator.variance)


def test_constant_columns_have_exactly_zero_variance():
    stats = StreamingStats().update_stats(np.full((7, 1), 0.1)).update_stats(np.full((3, 1), 0.1))
    assert stats.var[0] == 0.0 and stats.std[0] == 0.0
    low, high = update_min_max(np.full(2, np.nan), np.full(2, np.nan), np.array([[3.0, np.nan], [1.0, np.nan]]))
    assert low.tolist()[0] == 1.0 and high.tolist()[0] == 3.0 and np.isnan(low[1])


def test_update_moments_and_safe_divide():
    count, mean, m2 = update_moments(np.zeros(1), np.zeros(1), np.zeros(1), np.array([[1.0], [3.0]]))
    assert (count[0], mean[0], m2[0]) == (2.0, 2.0, 2.0)
    np.testing.assert_array_equal(safe_divide([1.0, 2.0], [0.0, 4.0]), [0.0, 0.5])


@pytest.mark.parametrize("params", [{"bins": 0}, {"sample_size": 0}, {"window": 0}, {"bins": 2.5}])
def test_invalid_settings(params):
    with pytest.raises(ValueError):
        StreamingStats(**params)


def test_three_dimensional_chunks_are_rejected():
    with pytest.raises(ValueError, match="1D or 2D"):
        StreamingStats().update_stats(np.ones((2, 2, 2)))
