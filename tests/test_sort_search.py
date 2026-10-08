"""
tests/test_sort_search.py
Unit tests for numcompute.sort_search

Run with: python -m tests.test_sort_search   (from the repo root)
or:       pytest tests/test_sort_search.py
"""

import bisect

import numpy as np

from numcompute.sort_search import stable_sort


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