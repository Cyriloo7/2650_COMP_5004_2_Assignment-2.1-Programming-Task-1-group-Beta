"""
tests/test_rank.py
Unit tests for numcompute.rank

Run from the repo root with:
    python -m pytest tests/test_rank.py -v
"""

import numpy as np
import pytest

from numcompute.rank import rank, percentile
from numcompute.stats import quantile


# ---------- rank(): ties and methods ----------

def test_ordinal_no_ties():
    data = np.array([30, 10, 20])
    result = rank(data, method="ordinal")
    expected = np.array([3, 1, 2], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_all_values_tied_average():
    # Required by spec: all tied -> average rank = mean of 1..n
    data = np.array([5, 5, 5, 5])
    result = rank(data, method="average")
    expected_value = np.mean(np.arange(1, 5))  # mean of 1..4 = 2.5
    np.testing.assert_array_almost_equal(result, np.full(4, expected_value))


def test_dense_vs_ordinal_on_tied_input():
    # Required by spec: 'dense' vs 'ordinal' on the same tied input
    data = np.array([10, 20, 20, 30])
    dense = rank(data, method="dense")
    ordinal = rank(data, method="ordinal")

    np.testing.assert_array_equal(dense, np.array([1, 2, 2, 3], dtype=float))
    assert len(np.unique(ordinal)) == len(ordinal)  # ordinal never ties
    assert ordinal[1] != ordinal[2]


def test_average_matches_hand_computed_ties():
    # 10, 20, 20, 40 -> ordinal 1,2,3,4 -> tied 20s average to 2.5
    data = np.array([10, 20, 20, 40])
    result = rank(data, method="average")
    expected = np.array([1, 2.5, 2.5, 4])
    np.testing.assert_array_almost_equal(result, expected)


def test_rank_nan_sorted_last_and_tied():
    # NaN policy: NaNs rank last and are mutually tied with each other.
    data = np.array([3, np.nan, 1, np.nan, 2])
    avg = rank(data, method="average")
    dense = rank(data, method="dense")
    # positions 1 and 3 are the two NaNs -> should get equal, top ranks
    assert avg[1] == avg[3] == 4.5
    assert dense[1] == dense[3] == 4
    # non-NaN values still rank correctly among themselves
    np.testing.assert_array_equal(
        np.array([avg[0], avg[2], avg[4]]), np.array([3.0, 1.0, 2.0])
    )


def test_rank_does_not_modify_input():
    data = np.array([3.0, 1.0, 2.0, 1.0])
    before = data.copy()
    rank(data, method="average")
    np.testing.assert_array_equal(data, before)


# ---------- rank(): axis ----------

def test_rank_2d_default_axis_ranks_each_column():
    # The default (axis=0) keeps the old behaviour: rank per column.
    data = np.array([
        [30, 1],
        [10, 1],
        [20, 1],
    ])
    result = rank(data, method="ordinal")
    np.testing.assert_array_equal(result[:, 0], np.array([3, 1, 2], dtype=float))
    assert set(result[:, 1]) == {1.0, 2.0, 3.0}
    np.testing.assert_array_equal(result, rank(data, method="ordinal", axis=0))


def test_rank_axis1_ranks_each_row():
    data = np.array([
        [3, 1, 2],
        [10, 30, 20],
    ])
    result = rank(data, method="ordinal", axis=1)
    expected = np.array([
        [3, 1, 2],
        [1, 3, 2],
    ], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_rank_axis1_handles_ties_per_row():
    data = np.array([
        [5, 5, 1],
        [2, 9, 9],
    ])
    result = rank(data, method="average", axis=1)
    expected = np.array([
        [2.5, 2.5, 1.0],
        [1.0, 2.5, 2.5],
    ])
    np.testing.assert_array_equal(result, expected)


def test_rank_negative_axes_match_positive_ones():
    data = np.array([[3.0, 1.0, 2.0], [10.0, 30.0, 20.0], [7.0, 7.0, 0.0]])
    np.testing.assert_array_equal(rank(data, axis=-1), rank(data, axis=1))
    np.testing.assert_array_equal(rank(data, axis=-2), rank(data, axis=0))


def test_rank_transpose_swaps_axis():
    # Ranking the transpose along axis 0 equals ranking the original along axis 1.
    data = np.array([[3.0, 1.0, 2.0, 2.0], [9.0, 8.0, 8.0, 7.0]])
    np.testing.assert_array_equal(
        rank(data, axis=1), rank(data.T, axis=0).T
    )


def test_rank_1d_accepts_axis_0_and_minus_1():
    data = np.array([20, 10, 30])
    expected = np.array([2, 1, 3], dtype=float)
    np.testing.assert_array_equal(rank(data, axis=0), expected)
    np.testing.assert_array_equal(rank(data, axis=-1), expected)


@pytest.mark.parametrize("data, axis", [
    (np.zeros((2, 3)), 2),
    (np.zeros((2, 3)), -3),
    (np.zeros(4), 1),
])
def test_rank_axis_out_of_bounds_raises(data, axis):
    with pytest.raises(ValueError, match="out of bounds"):
        rank(data, axis=axis)


@pytest.mark.parametrize("bad_axis", [1.0, "0", None, True])
def test_rank_axis_must_be_an_integer(bad_axis):
    with pytest.raises(TypeError, match="axis must be an integer"):
        rank(np.zeros((2, 2)), axis=bad_axis)


# ---------- rank(): errors and edge cases ----------

def test_rank_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown method 'bogus'"):
        rank(np.array([1, 2, 3]), method="bogus")


def test_rank_invalid_method_raises_even_for_empty_input():
    with pytest.raises(ValueError, match="Unknown method 'bogus'"):
        rank(np.array([]), method="bogus")


def test_rank_empty_array():
    result = rank(np.array([]), method="average")
    assert result.shape == (0,)


def test_rank_empty_2d_keeps_shape():
    assert rank(np.empty((0, 3))).shape == (0, 3)
    assert rank(np.empty((3, 0)), axis=1).shape == (3, 0)


@pytest.mark.parametrize("shape", [(), (2, 2, 2)])
def test_rank_other_dimensions_raise(shape):
    with pytest.raises(ValueError, match="supports 1D or 2D"):
        rank(np.zeros(shape), method="average")


@pytest.mark.parametrize("bad_data", [["a", "b", "c"], [[1, 2], [3]]])
def test_rank_non_numeric_input_raises_type_error(bad_data):
    with pytest.raises(TypeError, match="rank\\(\\) needs numeric input"):
        rank(bad_data)


# ---------- rank(): non-contiguous input ----------

def test_rank_strided_view_matches_contiguous_copy():
    base = np.array([9.0, 0.0, 7.0, 0.0, 5.0, 0.0, 7.0])
    view = base[::2]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(rank(view), rank(view.copy()))


def test_rank_transposed_and_column_sliced_views_match_copies():
    base = np.arange(24, dtype=float).reshape(4, 6)[:, ::-1] % 5
    for view in (base.T, base[:, ::2], base[::2]):
        copy = np.ascontiguousarray(view)
        for axis in (0, 1):
            np.testing.assert_array_equal(
                rank(view, axis=axis), rank(copy, axis=axis)
            )


# ---------- percentile(): basics ----------

def test_percentile_linear_basic():
    data = np.array([1, 2, 3, 4])
    result = percentile(data, 50, interpolation="linear")
    assert np.isclose(result, 2.5)


def test_percentile_ignores_nan():
    data = np.array([1, 2, np.nan, 4])
    result = percentile(data, 50)
    assert np.isclose(result, 2.0)


def test_percentile_lower_vs_higher_interpolation():
    data = np.array([1, 2, 3, 4])
    lower = percentile(data, 50, interpolation="lower")
    higher = percentile(data, 50, interpolation="higher")
    assert lower == 2
    assert higher == 3


@pytest.mark.parametrize(
    "method", ["linear", "lower", "higher", "midpoint", "nearest"]
)
def test_percentile_every_supported_interpolation_matches_numpy(method):
    data = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    got = percentile(data, [10, 50, 90], interpolation=method)
    expected = np.nanpercentile(data, [10, 50, 90], method=method)
    np.testing.assert_allclose(got, expected)


# ---------- percentile(): axis ----------

def test_percentile_2d_default_flattens():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.isclose(percentile(data, 50), 3.5)


def test_percentile_axis0_gives_one_value_per_column():
    data = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    np.testing.assert_allclose(percentile(data, 50, axis=0), [3.0, 30.0])


def test_percentile_axis1_gives_one_value_per_row():
    data = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    np.testing.assert_allclose(percentile(data, 50, axis=1), [2.0, 20.0])
    np.testing.assert_allclose(percentile(data, 50, axis=-1), [2.0, 20.0])


def test_percentile_axis_with_several_q_puts_q_first():
    data = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    result = percentile(data, [0, 100], axis=1)
    np.testing.assert_allclose(result, [[1.0, 10.0], [3.0, 30.0]])


def test_percentile_axis_ignores_nan_per_column():
    data = np.array([[1.0, np.nan], [3.0, 20.0], [5.0, 40.0]])
    np.testing.assert_allclose(percentile(data, 50, axis=0), [3.0, 30.0])


@pytest.mark.parametrize("axis", [2, -3])
def test_percentile_axis_out_of_bounds_raises(axis):
    with pytest.raises(ValueError, match="out of bounds") as info:
        percentile(np.zeros((2, 3)), 50, axis=axis)
    # The message must be about the axis, not misreported as a bad interpolation.
    assert "interpolation" not in str(info.value)


def test_percentile_axis_must_be_an_integer():
    with pytest.raises(TypeError, match="axis must be an integer"):
        percentile(np.zeros((2, 3)), 50, axis=0.5)


# ---------- percentile(): errors and non-contiguous input ----------

@pytest.mark.parametrize("bad_q", [150, -1, np.nan, [10, 101]])
def test_percentile_out_of_range_raises(bad_q):
    with pytest.raises(ValueError, match="q must be between 0 and 100"):
        percentile(np.array([1, 2, 3]), bad_q)


def test_percentile_invalid_interpolation_raises_clear_error():
    data = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="Invalid interpolation 'Break_mode'"):
        percentile(data, 50, interpolation="Break_mode")


@pytest.mark.parametrize("empty", [[], np.empty((0, 3))])
def test_percentile_empty_input_raises(empty):
    with pytest.raises(ValueError, match="data is empty"):
        percentile(empty, 50)


def test_percentile_and_quantile_raise_the_same_error_for_empty_input():
    with pytest.raises(ValueError, match="data is empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="data is empty"):
        quantile([], 0.5)


@pytest.mark.parametrize("bad", [["a", "b"], [[1, 2], [3]]])
def test_percentile_non_numeric_input_raises_type_error(bad):
    with pytest.raises(TypeError, match="percentile\\(\\) needs numeric input"):
        percentile(bad, 50)


def test_percentile_non_contiguous_views_match_contiguous_copies():
    base = np.arange(30, dtype=float).reshape(5, 6)[::-1] % 7
    for view in (base.T, base[:, ::2], base[::2]):
        copy = np.ascontiguousarray(view)
        for axis in (None, 0, 1):
            np.testing.assert_allclose(
                percentile(view, [25, 75], axis=axis),
                percentile(copy, [25, 75], axis=axis),
            )