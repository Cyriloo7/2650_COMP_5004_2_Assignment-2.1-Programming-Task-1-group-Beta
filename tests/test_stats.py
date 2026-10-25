"""Unit tests for numcompute.stats."""

import numpy as np
import pytest

from numcompute.stats import (
    StreamingStats,
    WelfordAccumulator,
    describe,
    histogram,
    quantile,
)


class TestDescribe:
    def test_basic_axis0(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        result = describe(X, axis=0)
        np.testing.assert_allclose(result["mean"], [2.5, 25.0])
        np.testing.assert_allclose(result["median"], [2.5, 25.0])
        np.testing.assert_allclose(result["min"], [1.0, 10.0])
        np.testing.assert_allclose(result["max"], [4.0, 40.0])

    def test_axis1(self):
        X = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        result = describe(X, axis=1)
        np.testing.assert_allclose(result["mean"], [2.0, 20.0])
        assert result["mean"].shape == (2,)

    def test_negative_axis_counts_from_the_end(self):
        X = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        np.testing.assert_allclose(
            describe(X, axis=-1)["mean"], describe(X, axis=1)["mean"]
        )

    def test_output_shape_matches_reduced_axis(self):
        X = np.zeros((5, 3))
        result = describe(X, axis=0)
        for key in ("mean", "median", "std", "min", "max"):
            assert result[key].shape == (3,)

    def test_single_nan_does_not_poison_whole_column(self):
        X = np.array([[1.0], [2.0], [np.nan], [4.0]])
        result = describe(X, axis=0)
        assert not np.isnan(result["mean"][0])
        np.testing.assert_allclose(result["mean"], [7.0 / 3.0])

    def test_all_nan_column_returns_nan_without_crashing(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        result = describe(X, axis=0)
        assert np.isnan(result["mean"][1])
        assert np.isnan(result["std"][1])
        assert not np.isnan(result["mean"][0])  # other column unaffected

    def test_all_nan_column_does_not_leak_a_warning(self, recwarn):
        X = np.array([[np.nan], [np.nan]])
        describe(X, axis=0)
        assert len(recwarn) == 0

    def test_empty_axis_raises_value_error(self):
        with pytest.raises(ValueError):
            describe(np.empty((0, 3)), axis=0)

    @pytest.mark.parametrize("bad_axis", [None, 5, -3, 0.5])
    def test_invalid_axis_raises_value_error(self, bad_axis):
        with pytest.raises(ValueError, match="invalid axis"):
            describe(np.zeros((4, 2)), axis=bad_axis)

    def test_single_element_array(self):
        result = describe(np.array([[5.0]]), axis=0)
        np.testing.assert_allclose(result["mean"], [5.0])
        np.testing.assert_allclose(result["std"], [0.0])

    def test_1d_input(self):
        result = describe(np.array([5.0, 1.0, 3.0, 2.0, 4.0]), axis=0)
        assert result["mean"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0

    def test_non_contiguous_input_gives_same_result_as_a_copy(self):
        X = np.arange(20.0).reshape(10, 2)
        X_strided = X[::2]
        assert not X_strided.flags["C_CONTIGUOUS"]
        result = describe(X_strided, axis=0)
        expected = describe(X_strided.copy(), axis=0)
        np.testing.assert_allclose(result["mean"], expected["mean"])
        np.testing.assert_allclose(result["std"], expected["std"])


class TestStreamingStats:
    def test_matches_numpy_mean_and_variance(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=500)
        acc = StreamingStats()
        for x in data:
            acc.update(x)
        assert np.isclose(acc.mean, data.mean())
        assert np.isclose(acc.variance, data.var())

    def test_std_property_matches_numpy_std(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=200)
        acc = StreamingStats()
        for x in data:
            acc.update(x)
        assert np.isclose(acc.std, data.std())

    def test_single_sample_has_zero_variance(self):
        acc = StreamingStats()
        acc.update(42.0)
        assert acc.variance == 0.0
        assert acc.mean == 42.0

    def test_update_returns_self_for_chaining(self):
        acc = StreamingStats()
        result = acc.update(1.0)
        assert result is acc

    def test_variance_with_zero_samples_raises(self):
        acc = StreamingStats()
        with pytest.raises(ValueError):
            _ = acc.variance

    def test_mean_with_zero_samples_raises(self):
        with pytest.raises(ValueError):
            _ = StreamingStats().mean

    def test_std_with_zero_samples_raises(self):
        with pytest.raises(ValueError):
            _ = StreamingStats().std

    def test_nan_samples_are_skipped(self):
        data = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0])
        acc = StreamingStats()
        for x in data:
            acc.update(x)
        valid = data[~np.isnan(data)]
        assert acc.count == len(valid)
        assert np.isclose(acc.mean, valid.mean())
        assert np.isclose(acc.variance, valid.var())

    def test_nan_update_returns_self_for_chaining(self):
        acc = StreamingStats()
        assert acc.update(np.nan) is acc

    def test_stream_of_only_nans_counts_as_zero_samples(self):
        acc = StreamingStats().update(np.nan).update(float("nan"))
        assert acc.count == 0
        with pytest.raises(ValueError):
            _ = acc.mean

    @pytest.mark.parametrize("bad_sample", ["a", None])
    def test_non_numeric_sample_raises_type_error(self, bad_sample):
        with pytest.raises(TypeError):
            StreamingStats().update(bad_sample)

    def test_merge_matches_processing_all_data_at_once(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=1000)
        a, b = StreamingStats(), StreamingStats()
        for x in data[:400]:
            a.update(x)
        for x in data[400:]:
            b.update(x)
        a.merge(b)
        assert a.count == len(data)
        assert np.isclose(a.mean, data.mean())
        assert np.isclose(a.variance, data.var())

    def test_merge_into_a_fresh_empty_accumulator(self):
        source = StreamingStats()
        for x in (1.0, 2.0, 3.0):
            source.update(x)
        empty = StreamingStats()
        empty.merge(source)
        assert empty.count == 3
        assert empty.mean == 2.0

    def test_merging_an_empty_accumulator_in_changes_nothing(self):
        populated = StreamingStats()
        for x in (1.0, 2.0, 3.0):
            populated.update(x)
        populated.merge(StreamingStats())
        assert populated.count == 3
        assert populated.mean == 2.0

    def test_welford_accumulator_is_an_alias_for_streaming_stats(self):
        assert WelfordAccumulator is StreamingStats


class TestHistogram:
    def test_matches_numpy_histogram_default_bins(self):
        data = np.array([1, 2, 2, 3, 3, 3, 4, 5])
        counts, edges = histogram(data)
        np_counts, np_edges = np.histogram(data)
        np.testing.assert_array_equal(counts, np_counts)
        np.testing.assert_allclose(edges, np_edges)

    def test_custom_bin_count(self):
        data = np.arange(100.0)
        counts, edges = histogram(data, bins=4)
        assert len(counts) == 4
        assert len(edges) == 5

    def test_custom_range(self):
        data = np.arange(100.0)
        counts, edges = histogram(data, bins=4, range=(0, 40))
        assert edges[0] == 0
        assert edges[-1] == 40

    def test_nan_values_are_dropped_before_binning(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 3.0])
        counts, edges = histogram(data, bins=3)
        expected_counts, expected_edges = np.histogram([1.0, 2.0, 3.0, 3.0], bins=3)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_allclose(edges, expected_edges)

    def test_all_nan_input_gives_zero_counts(self):
        counts, _ = histogram(np.array([np.nan, np.nan]), bins=4)
        np.testing.assert_array_equal(counts, np.zeros(4, dtype=int))

    @pytest.mark.parametrize("bins", [0, -3])
    def test_non_positive_bins_raises_value_error(self, bins):
        with pytest.raises(ValueError):
            histogram(np.array([1.0, 2.0]), bins=bins)

    def test_infinite_values_without_a_range_raise_value_error(self):
        with pytest.raises(ValueError):
            histogram(np.array([1.0, np.inf]))


class TestQuantile:
    def test_matches_nanpercentile_on_a_0_to_1_scale(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(quantile(data, 0.5), np.nanpercentile(data, 50))

    def test_ignores_nan_values(self):
        data = np.array([1.0, 2.0, 3.0, np.nan])
        assert np.isclose(quantile(data, 0.5), 2.0)

    @pytest.mark.parametrize("method", ["linear", "lower", "higher", "midpoint"])
    def test_each_interpolation_method(self, method):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        got = quantile(data, 0.5, interpolation=method)
        expected = np.nanpercentile(data, 50, method=method)
        assert np.isclose(got, expected)

    def test_array_valued_q(self):
        data = np.arange(1.0, 101.0)
        result = quantile(data, [0.25, 0.5, 0.75])
        expected = np.nanpercentile(data, [25, 50, 75])
        np.testing.assert_allclose(result, expected)

    def test_empty_data_raises_value_error(self):
        with pytest.raises(ValueError):
            quantile(np.array([]), 0.5)

    @pytest.mark.parametrize("bad_q", [1.5, -0.1])
    def test_q_out_of_range_raises_value_error(self, bad_q):
        with pytest.raises(ValueError):
            quantile(np.array([1.0, 2.0, 3.0]), bad_q)

    def test_default_axis_flattens_2d_input(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.isclose(quantile(X, 0.5), 2.5)

    def test_axis_0_gives_one_quantile_per_column(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        np.testing.assert_allclose(quantile(X, 0.5, axis=0), [2.5, 25.0])

    def test_axis_1_gives_one_quantile_per_row(self):
        X = np.array([[1.0, 3.0], [10.0, 30.0], [100.0, 300.0]])
        np.testing.assert_allclose(quantile(X, 0.5, axis=1), [2.0, 20.0, 200.0])

    def test_array_q_with_axis_puts_the_q_dimension_first(self):
        X = np.arange(12.0).reshape(4, 3)
        result = quantile(X, [0.25, 0.75], axis=0)
        assert result.shape == (2, 3)

    def test_nans_are_ignored_along_an_axis(self):
        X = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
        np.testing.assert_allclose(quantile(X, 0.5, axis=0), [3.0, 3.0])

    def test_invalid_axis_raises_value_error(self):
        with pytest.raises(ValueError):
            quantile(np.zeros((3, 2)), 0.5, axis=5)