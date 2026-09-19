"""
tests/test_rank.py
Unit tests for numcompute.rank
 
Run with: pytest tests/test_rank.py
(Falls back to plain asserts + a __main__ runner if pytest isn't installed.)
"""
 
import numpy as np
 
from numcompute.rank import rank, percentile
 
 
# ---------- rank() tests ----------
 
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
 
 
def test_rank_2d_applies_per_column():
    data = np.array([
        [30, 1],
        [10, 1],
        [20, 1],
    ])
    result = rank(data, method="ordinal")
    np.testing.assert_array_equal(result[:, 0], np.array([3, 1, 2], dtype=float))
    assert set(result[:, 1]) == {1.0, 2.0, 3.0}
 
 
def test_rank_invalid_method_raises():
    try:
        rank(np.array([1, 2, 3]), method="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
 
 
def test_rank_empty_array():
    result = rank(np.array([]), method="average")
    assert result.shape == (0,)
 
 
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
 
 
def test_rank_3d_raises():
    try:
        rank(np.zeros((2, 2, 2)), method="average")
        assert False, "expected ValueError"
    except ValueError:
        pass
 
 
# ---------- percentile() tests ----------
 
def test_percentile_linear_basic():
    data = np.array([1, 2, 3, 4])
    result = percentile(data, 50, interpolation="linear")
    assert np.isclose(result, 2.5)
 
 
def test_percentile_ignores_nan():
    data = np.array([1, 2, np.nan, 4])
    result = percentile(data, 50)
    assert np.isclose(result, 2.0)
 
 
def test_percentile_out_of_range_raises():
    try:
        percentile(np.array([1, 2, 3]), 150)
        assert False, "expected ValueError"
    except ValueError:
        pass
 
 
def test_percentile_lower_vs_higher_interpolation():
    data = np.array([1, 2, 3, 4])
    lower = percentile(data, 50, interpolation="lower")
    higher = percentile(data, 50, interpolation="higher")
    assert lower == 2
    assert higher == 3
 
 
if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} -> {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")