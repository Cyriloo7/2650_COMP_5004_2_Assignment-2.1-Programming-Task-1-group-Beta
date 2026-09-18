"""
Unit tests for numcompute.utils.

Run from the project root with:
    pytest tests/test_utils.py -v
"""

import numpy as np
import pytest

from numcompute.utils import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
    sigmoid,
    relu,
    softmax,
    logsumexp,
    safe_divide,
    batch_iterator,
)


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

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


class TestRelu:
    def test_known_values(self):
        x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        np.testing.assert_allclose(relu(x), [0.0, 0.0, 0.0, 0.5, 3.0])


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


# ---------------------------------------------------------------------------
# logsumexp
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# safe_divide
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# batch_iterator
# ---------------------------------------------------------------------------

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