"""
tests/test_sort_search.py
Unit tests for numcompute.sort_search

Run with: python -m tests.test_sort_search   (from the repo root)
or:       pytest tests/test_sort_search.py
"""

import bisect

import numpy as np

from numcompute.sort_search import stable_sort, multi_key_sort, topk ,quickselect, binary_search


def _raises(exc_type, func, *args, **kwargs):
    """Small helper so the tests also run without pytest installed."""
    try:
        func(*args, **kwargs)
    except exc_type:
        return True
    return False


# ---------- stable_sort ----------

def test_stable_sort_basic():
    result = stable_sort(np.array([3, 1, 2]))
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_stable_sort_empty_array():
    result = stable_sort(np.array([]))
    assert result.shape == (0,)


def test_stable_sort_axis_and_2d():
    data = np.array([[3, 1], [2, 5], [1, 4]])
    by_col = stable_sort(data, axis=0)
    np.testing.assert_array_equal(by_col, np.array([[1, 1], [2, 4], [3, 5]]))
    by_row = stable_sort(data, axis=1)
    np.testing.assert_array_equal(by_row, np.array([[1, 3], [2, 5], [1, 4]]))


def test_stable_sort_nan_goes_last():
    result = stable_sort(np.array([2.0, np.nan, 1.0]))
    np.testing.assert_array_equal(result[:2], np.array([1.0, 2.0]))
    assert np.isnan(result[2])


def test_stable_sort_non_contiguous_view():
    base = np.array([9, 0, 7, 0, 5, 0, 3])
    view = base[::2]  # strided, non-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(stable_sort(view), np.array([3, 5, 7, 9]))
    # original must be untouched
    np.testing.assert_array_equal(base, np.array([9, 0, 7, 0, 5, 0, 3]))


# ---------- multi_key_sort ----------

def test_multi_key_sort_last_key_is_primary():
    data = np.array([
        [1, 2],
        [0, 2],
        [5, 1],
    ])
    # keys=[0, 1] -> column 1 is PRIMARY, column 0 breaks ties.
    result = multi_key_sort(data, keys=[0, 1])
    expected = np.array([[5, 1], [0, 2], [1, 2]])
    np.testing.assert_array_equal(result, expected)


def test_multi_key_sort_descending():
    data = np.array([[1, 10], [2, 30], [3, 20]])
    result = multi_key_sort(data, keys=[1], ascending=[False])
    np.testing.assert_array_equal(result[:, 1], np.array([30, 20, 10]))


def test_multi_key_sort_mixed_directions():
    data = np.array([
        [1, 5],
        [2, 5],
        [3, 7],
    ])
    # primary = column 1 descending, tie-break = column 0 ascending
    result = multi_key_sort(data, keys=[0, 1], ascending=[True, False])
    expected = np.array([[3, 7], [1, 5], [2, 5]])
    np.testing.assert_array_equal(result, expected)


def test_multi_key_sort_is_stable_on_full_ties():
    data = np.array([[1, 0], [1, 1], [1, 2]])
    result = multi_key_sort(data, keys=[0])
    np.testing.assert_array_equal(result, data)  # original order kept


def test_multi_key_sort_invalid_inputs_raise():
    data = np.array([[1, 2], [3, 4]])
    assert _raises(ValueError, multi_key_sort, np.array([1, 2, 3]), [0])  # not 2D
    assert _raises(ValueError, multi_key_sort, data, [])                  # no keys
    assert _raises(ValueError, multi_key_sort, data, [5])                 # bad column
    assert _raises(ValueError, multi_key_sort, data, [0, 1], [True])      # wrong length


def test_multi_key_sort_empty_rows():
    data = np.empty((0, 3))
    result = multi_key_sort(data, keys=[0])
    assert result.shape == (0, 3)

def test_topk_largest_with_indices():
    values = np.array([5, 1, 9, 3, 7])
    top_values, top_idx = topk(values, 3)
    np.testing.assert_array_equal(top_values, np.array([9, 7, 5]))
    np.testing.assert_array_equal(top_idx, np.array([2, 4, 0]))


def test_topk_smallest():
    values = np.array([5, 1, 9, 3, 7])
    top_values, top_idx = topk(values, 2, largest=False)
    np.testing.assert_array_equal(top_values, np.array([1, 3]))
    np.testing.assert_array_equal(top_idx, np.array([1, 3]))


def test_topk_return_indices_false():
    result = topk(np.array([4, 8, 6]), 2, return_indices=False)
    np.testing.assert_array_equal(result, np.array([8, 6]))


def test_topk_k_zero_or_negative_raises():
    assert _raises(ValueError, topk, np.array([1, 2, 3]), 0)
    assert _raises(ValueError, topk, np.array([1, 2, 3]), -2)


def test_topk_k_equals_length():
    values = np.array([2, 3, 1])
    top_values, _ = topk(values, 3)
    np.testing.assert_array_equal(top_values, np.array([3, 2, 1]))


def test_topk_k_too_large_is_clamped():
    values = np.array([2, 3, 1])
    top_values, top_idx = topk(values, 100)
    np.testing.assert_array_equal(top_values, np.array([3, 2, 1]))
    assert top_idx.size == 3


def test_topk_empty_array():
    top_values, top_idx = topk(np.array([]), 3)
    assert top_values.size == 0 and top_idx.size == 0


def test_topk_all_equal_values():
    values = np.array([4.0, 4.0, 4.0, 4.0])
    top_values, top_idx = topk(values, 2)
    np.testing.assert_array_equal(top_values, np.array([4.0, 4.0]))
    assert len(set(top_idx.tolist())) == 2  # two distinct positions


def test_topk_ignores_nan():
    values = np.array([1.0, np.nan, 5.0, np.nan, 3.0])
    top_values, top_idx = topk(values, 2)
    np.testing.assert_array_equal(top_values, np.array([5.0, 3.0]))
    np.testing.assert_array_equal(top_idx, np.array([2, 4]))  # original positions


def test_topk_all_nan_returns_empty():
    top_values, top_idx = topk(np.array([np.nan, np.nan]), 2)
    assert top_values.size == 0 and top_idx.size == 0


def test_topk_non_contiguous_view():
    base = np.array([1, 99, 8, 99, 5, 99, 6])
    view = base[::2]  # [1, 8, 5, 6]
    top_values, top_idx = topk(view, 2)
    np.testing.assert_array_equal(top_values, np.array([8, 6]))
    np.testing.assert_array_equal(top_idx, np.array([1, 3]))


def test_topk_matches_full_sort_on_random_data():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    top_values, _ = topk(values, 10)
    expected = np.sort(values)[::-1][:10]
    np.testing.assert_allclose(top_values, expected)


def test_topk_rejects_2d():
    assert _raises(ValueError, topk, np.ones((2, 2)), 1)


def test_quickselect_matches_sorted_for_every_k():
    values = np.array([7, 2, 9, 4, 4, 1, 8])
    expected = np.sort(values)
    for k in range(values.size):
        assert quickselect(values, k) == expected[k]


def test_quickselect_does_not_modify_input():
    values = np.array([3, 1, 2])
    quickselect(values, 1)
    np.testing.assert_array_equal(values, np.array([3, 1, 2]))


def test_quickselect_single_element():
    assert quickselect(np.array([42]), 0) == 42


def test_quickselect_all_equal():
    assert quickselect(np.array([5, 5, 5, 5, 5]), 3) == 5


def test_quickselect_already_sorted_and_reversed():
    ascending = np.arange(50)
    assert quickselect(ascending, 25) == 25
    assert quickselect(ascending[::-1], 25) == 25  # also a non-contiguous view


def test_quickselect_random_agrees_with_numpy():
    rng = np.random.default_rng(1)
    values = rng.integers(0, 20, size=300)  # lots of duplicates
    expected = np.sort(values)
    for k in (0, 17, 150, 299):
        assert quickselect(values, k) == expected[k]


def test_quickselect_invalid_inputs_raise():
    assert _raises(ValueError, quickselect, np.array([]), 0)             # empty
    assert _raises(ValueError, quickselect, np.array([1, 2, 3]), 3)      # k too big
    assert _raises(ValueError, quickselect, np.array([1, 2, 3]), -1)     # k negative
    assert _raises(ValueError, quickselect, np.array([1.0, np.nan]), 0)  # NaN
    assert _raises(ValueError, quickselect, np.ones((2, 2)), 0)          # not 1D

def test_binary_search_found():
    index, exists = binary_search(np.array([1, 3, 5, 7]), 5)
    assert (index, exists) == (2, True)


def test_binary_search_value_not_present():
    index, exists = binary_search(np.array([1, 3, 5, 7]), 4)
    assert (index, exists) == (2, False)


def test_binary_search_below_and_above_range():
    arr = np.array([10, 20, 30])
    assert binary_search(arr, 5) == (0, False)
    assert binary_search(arr, 99) == (3, False)


def test_binary_search_empty_array():
    assert binary_search(np.array([]), 1) == (0, False)


def test_binary_search_duplicates_returns_first_occurrence():
    index, exists = binary_search(np.array([1, 2, 2, 2, 3]), 2)
    assert (index, exists) == (1, True)


def test_binary_search_matches_bisect_and_searchsorted():
    arr = np.array([1, 2, 2, 4, 4, 4, 8, 9])
    for x in range(0, 11):
        index, _ = binary_search(arr, x)
        assert index == bisect.bisect_left(arr.tolist(), x)
        assert index == np.searchsorted(arr, x, side="left")


def test_binary_search_nan_and_bad_shape_raise():
    assert _raises(ValueError, binary_search, np.array([1.0, 2.0]), np.nan)
    assert _raises(ValueError, binary_search, np.ones((2, 2)), 1)

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} -> {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")