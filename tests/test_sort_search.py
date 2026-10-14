"""
tests/test_sort_search.py
Unit tests for numcompute.sort_search

Run with: python -m tests.test_sort_search   (from the repo root)
or:       pytest tests/test_sort_search.py
"""

import bisect

import numpy as np

from numcompute.sort_search import stable_sort, multi_key_sort


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