"""Unit tests for numcompute.utils."""

import numpy as np
import pytest

from numcompute.utils import (
    batch_iterator,
    cosine_distance,
    euclidean_distance,
    log_softmax,
    logsumexp,
    manhattan_distance,
    relu,
    safe_divide,
    sigmoid,
    softmax,
    topk_indices,
)

DISTANCE_FUNCTIONS = [euclidean_distance, manhattan_distance, cosine_distance]
DISTANCE_IDS = ["euclidean", "manhattan", "cosine"]


@pytest.mark.parametrize("fn", DISTANCE_FUNCTIONS, ids=DISTANCE_IDS)
class TestDistanceInputs:
    def test_1d_first_input_raises(self, fn):
        with pytest.raises(ValueError, match="2-D"):
            fn(np.zeros(3), np.zeros((2, 3)))

    def test_1d_second_input_raises(self, fn):
        with pytest.raises(ValueError, match="2-D"):
            fn(np.zeros((2, 3)), np.zeros(3))

    def test_3d_input_raises(self, fn):
        with pytest.raises(ValueError, match="2-D"):
            fn(np.zeros((2, 3, 4)), np.zeros((2, 4)))

    def test_row_sliced_views_match_contiguous_copies(self, fn):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(10, 4))[::2]
        B = rng.normal(size=(9, 4))[::3]
        assert not A.flags["C_CONTIGUOUS"] and not B.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(fn(A, B), fn(A.copy(), B.copy()))

    def test_transposed_view_matches_contiguous_copy(self, fn):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(4, 7)).T
        B = rng.normal(size=(5, 4))
        assert not A.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(fn(A, B), fn(np.ascontiguousarray(A), B))


class TestEuclideanDistance:
    def test_known_values(self):
        A = np.array([[0.0, 0.0]])
        B = np.array([[3.0, 4.0]])
        result = euclidean_distance(A, B)
        np.testing.assert_allclose(result, [[5.0]])

    def test_shape(self):
        A = np.zeros((5, 3))
        B = np.zeros((4, 3))
        assert euclidean_distance(A, B).shape == (5, 4)

    def test_distance_to_self_is_zero(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(6, 4))
        result = euclidean_distance(A, A)
        # tiny floating-point noise is expected, not exactly 0.0
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-6)

    def test_never_produces_nan(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(10, 5))
        B = rng.normal(size=(8, 5))
        result = euclidean_distance(A, B)
        assert not np.any(np.isnan(result))

    def test_mismatched_dimensions_raises(self):
        with pytest.raises(ValueError):
            euclidean_distance(np.zeros((3, 2)), np.zeros((3, 4)))

    def test_near_duplicate_points_far_from_the_origin_are_accurate(self):
        A = np.array([[1e8, 0.0]])
        B = np.array([[1e8 + 1e-3, 0.0]])
        expected = B[0, 0] - A[0, 0]  # separation of the two stored values
        np.testing.assert_allclose(euclidean_distance(A, B), [[expected]], rtol=1e-6)

    def test_distances_are_unchanged_by_a_common_offset(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(8, 3))
        B = rng.normal(size=(6, 3))
        shifted = euclidean_distance(A + 1e8, B + 1e8)
        np.testing.assert_allclose(shifted, euclidean_distance(A, B), rtol=1e-6)

    def test_nan_in_one_point_does_not_affect_other_points(self, recwarn):
        A = np.array([[np.nan, 0.0], [3.0, 4.0]])
        B = np.array([[0.0, 0.0]])
        result = euclidean_distance(A, B)
        assert len(recwarn) == 0
        assert np.isnan(result[0, 0])
        np.testing.assert_allclose(result[1, 0], 5.0)

    @pytest.mark.parametrize(
        "shape_a, shape_b", [((0, 3), (4, 3)), ((4, 3), (0, 3)), ((0, 3), (0, 3))]
    )
    def test_empty_input_gives_an_empty_matrix(self, shape_a, shape_b):
        result = euclidean_distance(np.zeros(shape_a), np.zeros(shape_b))
        assert result.shape == (shape_a[0], shape_b[0])


class TestManhattanDistance:
    def test_known_values(self):
        A = np.array([[0.0, 0.0]])
        B = np.array([[3.0, 4.0]])
        result = manhattan_distance(A, B)
        np.testing.assert_allclose(result, [[7.0]])

    def test_shape(self):
        A = np.zeros((5, 3))
        B = np.zeros((4, 3))
        assert manhattan_distance(A, B).shape == (5, 4)

    def test_mismatched_dimensions_raises(self):
        with pytest.raises(ValueError):
            manhattan_distance(np.zeros((3, 2)), np.zeros((3, 4)))


class TestCosineDistance:
    def test_identical_vectors_have_zero_distance(self):
        A = np.array([[1.0, 2.0, 3.0]])
        result = cosine_distance(A, A)
        np.testing.assert_allclose(result, [[0.0]], atol=1e-10)

    def test_orthogonal_vectors_have_distance_one(self):
        A = np.array([[1.0, 0.0]])
        B = np.array([[0.0, 1.0]])
        np.testing.assert_allclose(cosine_distance(A, B), [[1.0]])

    def test_opposite_vectors_have_distance_two(self):
        A = np.array([[1.0, 0.0]])
        B = np.array([[-1.0, 0.0]])
        np.testing.assert_allclose(cosine_distance(A, B), [[2.0]])

    def test_zero_vector_does_not_produce_nan(self):
        A = np.array([[0.0, 0.0], [1.0, 0.0]])
        B = np.array([[1.0, 0.0]])
        result = cosine_distance(A, B)
        assert not np.any(np.isnan(result))

    def test_mismatched_dimensions_raises(self):
        with pytest.raises(ValueError):
            cosine_distance(np.zeros((3, 2)), np.zeros((3, 4)))


class TestSigmoid:
    def test_known_values(self):
        result = sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, [0.5])

    def test_output_in_open_unit_interval(self):
        x = np.linspace(-20, 20, 41)
        result = sigmoid(x)
        assert np.all(result > 0.0) and np.all(result < 1.0)

    def test_extreme_values_do_not_warn_or_nan(self, recwarn):
        result = sigmoid(np.array([-1000.0, 1000.0]))
        assert len(recwarn) == 0
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_non_numeric_input_raises_value_error(self):
        with pytest.raises(ValueError):
            sigmoid(["a", "b"])


class TestRelu:
    def test_known_values(self):
        x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        np.testing.assert_allclose(relu(x), [0.0, 0.0, 0.0, 0.5, 3.0])

    def test_non_numeric_input_raises_value_error(self):
        with pytest.raises(ValueError):
            relu(["a", "b"])


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        assert np.isclose(softmax(x).sum(), 1.0)

    def test_extreme_values_do_not_warn_or_nan(self, recwarn):
        result = softmax(np.array([1000.0, 1001.0, 1002.0]))
        assert len(recwarn) == 0
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(result, [0.09003057, 0.24472847, 0.66524096])

    def test_axis_parameter_on_2d_input(self):
        X = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = softmax(X, axis=1)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0])

    def test_row_sliced_view_along_axis_0(self):
        X = np.arange(24.0).reshape(6, 4)[::2]
        assert not X.flags["C_CONTIGUOUS"]
        result = softmax(X, axis=0)
        np.testing.assert_allclose(result, softmax(X.copy(), axis=0))
        np.testing.assert_allclose(result.sum(axis=0), np.ones(4))

    def test_transposed_view_along_axis_0(self):
        X = np.arange(12.0).reshape(3, 4).T
        assert not X.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(
            softmax(X, axis=0), softmax(np.ascontiguousarray(X), axis=0)
        )

    def test_negative_infinity_entries_get_zero_probability(self, recwarn):
        result = softmax(np.array([-np.inf, 0.0, 1.0]))
        assert len(recwarn) == 0
        assert result[0] == 0.0
        assert np.isclose(result.sum(), 1.0)

    def test_positive_infinity_entries_share_all_the_probability(self, recwarn):
        first = softmax(np.array([np.inf, 0.0]))
        second = softmax(np.array([np.inf, np.inf, 5.0]))
        assert len(recwarn) == 0
        np.testing.assert_allclose(first, [1.0, 0.0])
        np.testing.assert_allclose(second, [0.5, 0.5, 0.0])

    def test_positive_infinity_next_to_a_large_finite_value_does_not_warn(self, recwarn):
        result = softmax(np.array([np.inf, 1000.0]))
        assert len(recwarn) == 0
        np.testing.assert_allclose(result, [1.0, 0.0])

    def test_entirely_negative_infinity_slice_gives_nan_without_warning(self, recwarn):
        result = softmax(np.array([[-np.inf, -np.inf], [0.0, 0.0]]), axis=1)
        assert len(recwarn) == 0
        assert np.all(np.isnan(result[0]))
        np.testing.assert_allclose(result[1], [0.5, 0.5])

    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            softmax(np.array([]))


class TestLogSoftmax:
    def test_matches_log_of_softmax_for_moderate_values(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(log_softmax(x), np.log(softmax(x)))

    def test_stays_finite_where_softmax_underflows(self):
        result = log_softmax(np.array([0.0, -1000.0]))
        np.testing.assert_allclose(result, [0.0, -1000.0])

    def test_extreme_values_do_not_warn_or_overflow(self, recwarn):
        result = log_softmax(np.array([1000.0, 1001.0, 1002.0]))
        assert len(recwarn) == 0
        np.testing.assert_allclose(np.exp(result).sum(), 1.0)

    def test_axis_parameter_on_2d_input(self):
        X = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = log_softmax(X, axis=1)
        np.testing.assert_allclose(np.exp(result).sum(axis=1), [1.0, 1.0])

    def test_infinities_follow_the_softmax_rules(self, recwarn):
        negative = log_softmax(np.array([-np.inf, 0.0]))
        positive = log_softmax(np.array([np.inf, 0.0]))
        degenerate = log_softmax(np.array([-np.inf, -np.inf]))
        assert len(recwarn) == 0
        np.testing.assert_allclose(negative, [-np.inf, 0.0])
        np.testing.assert_allclose(positive, [0.0, -np.inf])
        assert np.all(np.isnan(degenerate))

    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            log_softmax(np.array([]))


class TestLogSumExp:
    def test_matches_direct_computation_for_small_values(self):
        x = np.array([1.0, 2.0, 3.0])
        expected = np.log(np.sum(np.exp(x)))
        assert np.isclose(logsumexp(x), expected)

    def test_axis_none_returns_a_scalar(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = logsumexp(x)
        assert np.isscalar(result) or np.ndim(result) == 0

    def test_axis_reduces_like_np_sum(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = logsumexp(x, axis=1)
        assert result.shape == (2,)

    def test_extreme_values_do_not_overflow(self, recwarn):
        result = logsumexp(np.array([1000.0, 1001.0, 1002.0]))
        assert len(recwarn) == 0
        assert not np.isnan(result) and not np.isinf(result)

    def test_entirely_negative_infinity_gives_negative_infinity(self, recwarn):
        result = logsumexp(np.array([-np.inf, -np.inf]))
        assert len(recwarn) == 0
        assert result == -np.inf

    def test_positive_infinity_entry_gives_positive_infinity(self, recwarn):
        first = logsumexp(np.array([np.inf, 0.0]))
        second = logsumexp(np.array([np.inf, 1000.0]))
        assert len(recwarn) == 0
        assert first == np.inf and second == np.inf

    def test_negative_infinity_entries_are_ignored(self, recwarn):
        result = logsumexp(np.array([-np.inf, 0.0]))
        assert len(recwarn) == 0
        assert np.isclose(result, 0.0)

    def test_infinities_are_handled_slice_by_slice_along_an_axis(self, recwarn):
        X = np.array([[-np.inf, -np.inf], [0.0, 0.0], [np.inf, 1.0]])
        result = logsumexp(X, axis=1)
        assert len(recwarn) == 0
        np.testing.assert_allclose(result, [-np.inf, np.log(2.0), np.inf])

    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            logsumexp(np.array([]))


class TestSafeDivide:
    def test_normal_division(self):
        result = safe_divide(np.array([10.0, 20.0]), np.array([2.0, 5.0]))
        np.testing.assert_allclose(result, [5.0, 4.0])

    def test_zero_denominator_uses_fill(self):
        result = safe_divide(np.array([10.0, 5.0]), np.array([0.0, 0.0]), fill=-1.0)
        np.testing.assert_allclose(result, [-1.0, -1.0])

    def test_default_fill_is_zero(self):
        result = safe_divide(np.array([5.0]), np.array([0.0]))
        assert result[0] == 0.0

    def test_no_warning_on_zero_division(self, recwarn):
        safe_divide(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        assert len(recwarn) == 0

    def test_non_broadcastable_shapes_raise_value_error(self):
        with pytest.raises(ValueError):
            safe_divide(np.ones(3), np.ones(4))


def _stable_argsort_prefix(X, k, axis, largest):
    """Reference: the first k entries of a stable argsort (of -X for largest)."""
    order = np.argsort(-X if largest else X, axis=axis, kind="stable")
    return np.take(order, np.arange(k), axis=axis)


class TestTopkIndices:
    @pytest.mark.parametrize("largest", [True, False])
    @pytest.mark.parametrize(
        "shape, axis", [((9,), -1), ((5, 8), -1), ((5, 8), 0), ((3, 4, 6), 1)]
    )
    def test_matches_stable_argsort_for_every_k(self, shape, axis, largest):
        X = np.random.default_rng(0).normal(size=shape)
        for k in range(1, X.shape[axis] + 1):
            np.testing.assert_array_equal(
                topk_indices(X, k, axis=axis, largest=largest),
                _stable_argsort_prefix(X, k, axis, largest),
            )

    @pytest.mark.parametrize("largest", [True, False])
    def test_matches_stable_argsort_with_heavy_ties(self, largest):
        X = np.random.default_rng(1).integers(0, 3, size=(20, 12)).astype(float)
        for k in range(1, 13):
            np.testing.assert_array_equal(
                topk_indices(X, k, largest=largest),
                _stable_argsort_prefix(X, k, -1, largest),
            )

    def test_ties_are_broken_by_lower_index_first(self):
        X = np.array([[2.0, 1.0, 2.0, 1.0, 2.0]])
        np.testing.assert_array_equal(topk_indices(X, 2, largest=True), [[0, 2]])
        np.testing.assert_array_equal(topk_indices(X, 3, largest=False), [[1, 3, 0]])

    @pytest.mark.parametrize("largest", [True, False])
    def test_nan_ranks_last_in_both_modes(self, largest):
        X = np.array([[np.nan, 3.0, 1.0, np.nan, 2.0]])
        full = topk_indices(X, 5, largest=largest)
        np.testing.assert_array_equal(full, _stable_argsort_prefix(X, 5, -1, largest))
        assert np.all(np.isnan(X[0, full[0, -2:]]))

    def test_nan_is_not_selected_while_enough_valid_values_exist(self):
        X = np.array([[np.nan, 3.0, 1.0, np.nan, 2.0]])
        np.testing.assert_array_equal(topk_indices(X, 2, largest=True), [[1, 4]])

    def test_nearest_neighbours_on_a_distance_matrix(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [0.4, 0.0]])
        neighbours = topk_indices(euclidean_distance(points, points), 2, largest=False)
        np.testing.assert_array_equal(neighbours[:, 0], [0, 1, 2, 3])  # each point itself
        np.testing.assert_array_equal(neighbours[:, 1], [3, 3, 1, 0])  # then its nearest

    def test_output_shape_for_a_middle_axis(self):
        assert topk_indices(np.zeros((4, 6, 5)), 3, axis=1).shape == (4, 3, 5)

    def test_1d_input(self):
        np.testing.assert_array_equal(topk_indices(np.array([5.0, 1.0, 3.0]), 2), [0, 2])

    def test_non_contiguous_input(self):
        X = np.random.default_rng(3).normal(size=(10, 8))
        for view, axis in [(X[::2], -1), (X.T, 0), (X[:, ::2], -1)]:
            assert not view.flags["C_CONTIGUOUS"]
            np.testing.assert_array_equal(
                topk_indices(view, 2, axis=axis), topk_indices(view.copy(), 2, axis=axis)
            )

    def test_zero_rows_give_an_empty_result(self):
        assert topk_indices(np.zeros((0, 5)), 2).shape == (0, 2)

    def test_numpy_integer_k_is_accepted(self):
        assert topk_indices(np.arange(5.0), np.int64(2)).shape == (2,)

    @pytest.mark.parametrize("bad_k", [0, -1, 7, 2.5, "3", None, True])
    def test_invalid_k_raises_value_error(self, bad_k):
        with pytest.raises(ValueError, match="k must be"):
            topk_indices(np.zeros((2, 6)), bad_k)

    def test_axis_out_of_range_raises_value_error(self):
        with pytest.raises(ValueError):
            topk_indices(np.zeros((3, 4)), 2, axis=5)


class TestBatchIterator:
    def test_covers_every_element_exactly_once_in_order(self):
        X = np.arange(23)
        batches = list(batch_iterator(X, batch_size=5))
        assert [len(b) for b in batches] == [5, 5, 5, 5, 3]
        np.testing.assert_array_equal(np.concatenate(batches), X)

    def test_batch_size_larger_than_array(self):
        batches = list(batch_iterator(np.arange(3), batch_size=10))
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_empty_array_yields_nothing(self):
        assert list(batch_iterator(np.array([]), batch_size=5)) == []

    @pytest.mark.parametrize("bad_size", [0, -1])
    def test_non_positive_batch_size_raises(self, bad_size):
        with pytest.raises(ValueError):
            list(batch_iterator(np.arange(5), batch_size=bad_size))

    @pytest.mark.parametrize("bad_size", [2.5, "3", None, True])
    def test_non_integer_batch_size_raises_value_error(self, bad_size):
        with pytest.raises(ValueError, match="positive integer"):
            list(batch_iterator(np.arange(5), batch_size=bad_size))

    def test_numpy_integer_batch_size_is_accepted(self):
        batches = list(batch_iterator(np.arange(5), batch_size=np.int64(2)))
        assert [len(b) for b in batches] == [2, 2, 1]