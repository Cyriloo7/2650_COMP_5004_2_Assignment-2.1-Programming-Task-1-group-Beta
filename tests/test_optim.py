"""
tests/test_optim.py
Unit tests for numcompute.optim

Run from the repo root with:
    python -m pytest tests/test_optim.py -v
"""

import numpy as np
import pytest

from numcompute.optim import grad, jacobian


def _sum_of_squares(x):
    return np.sum(x ** 2)


def _counting(func):
    """Wrap `func` so the number of calls made to it is recorded."""
    calls = []

    def wrapper(x):
        calls.append(1)
        return func(x)

    wrapper.calls = calls
    return wrapper


# ---------- grad(): accuracy ----------

def test_grad_sum_of_squares_matches_2x_central():
    # Required by spec: grad on f(x) = sum(x**2) matches 2*x within tolerance
    x = np.array([1.0, 2.0, 3.0])
    result = grad(_sum_of_squares, x, method="central")
    expected = 2 * x
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_grad_sum_of_squares_matches_2x_forward():
    x = np.array([1.0, 2.0, 3.0])
    result = grad(_sum_of_squares, x, method="forward")
    expected = 2 * x
    # forward differences are less accurate (O(h) error), so use a looser tolerance
    np.testing.assert_allclose(result, expected, atol=1e-3)


def test_grad_negative_and_zero_values():
    x = np.array([0.0, -2.0, 5.0])
    result = grad(_sum_of_squares, x, method="central")
    expected = 2 * x
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_grad_scales_h_for_large_magnitude_input():
    # Required by spec: "consider scaling h relative to abs(x[i])".
    # A fixed, unscaled h=1e-5 breaks down completely for large x
    # (rounding error swamps the perturbation). With scaling, the
    # estimate should stay accurate even at x = 1e12.
    x = np.array([1e12])
    result = grad(_sum_of_squares, x, h=1e-5, method="central")
    expected = 2 * x
    relative_error = abs(result[0] - expected[0]) / expected[0]
    assert relative_error < 1e-8, (
        f"expected h-scaling to keep relative error small, got {relative_error:.2e}"
    )


def test_grad_accepts_integer_input_and_does_not_modify_it():
    x = np.array([1, 2, 3])
    result = grad(_sum_of_squares, x)
    np.testing.assert_allclose(result, [2.0, 4.0, 6.0], atol=1e-5)
    np.testing.assert_array_equal(x, np.array([1, 2, 3]))


def test_grad_empty_input_gives_empty_gradient():
    for method in ("central", "forward"):
        result = grad(_sum_of_squares, np.array([]), method=method)
        assert result.shape == (0,)


# ---------- grad(): errors ----------

def test_grad_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown method 'bogus'"):
        grad(np.sum, np.array([1.0, 2.0]), method="bogus")


def test_grad_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1D array for x"):
        grad(np.sum, np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.parametrize("bad_h", [0, 0.0, -1e-5, np.inf, -np.inf, np.nan])
def test_grad_rejects_h_that_is_not_positive_and_finite(bad_h):
    with pytest.raises(ValueError, match="h must be finite and > 0"):
        grad(_sum_of_squares, np.array([1.0, 2.0]), h=bad_h)


@pytest.mark.parametrize("bad_h", ["1e-5", None, True, [1e-5]])
def test_grad_rejects_h_that_is_not_a_number(bad_h):
    with pytest.raises(TypeError, match="h must be a number"):
        grad(_sum_of_squares, np.array([1.0, 2.0]), h=bad_h)


@pytest.mark.parametrize("method", ["central", "forward"])
def test_grad_rejects_f_that_returns_a_vector(method):
    def vector_f(x):
        return x * 2.0

    with pytest.raises(ValueError, match="single number.*shape \\(2,\\)"):
        grad(vector_f, np.array([1.0, 2.0]), method=method)


def test_grad_rejects_f_that_returns_a_list():
    with pytest.raises(ValueError, match="single number"):
        grad(lambda x: [1.0, 2.0], np.array([1.0, 2.0]))


@pytest.mark.parametrize("bad_return", ["text", [[1.0, 2.0], [3.0]]])
def test_grad_rejects_f_that_returns_a_non_number(bad_return):
    with pytest.raises(TypeError, match="single number"):
        grad(lambda x: bad_return, np.array([1.0, 2.0]))


def test_grad_accepts_numpy_scalars_and_zero_d_arrays():
    # np.float64, a plain int and a 0-d array all count as "a single number".
    x = np.array([1.0, 2.0])
    np.testing.assert_allclose(grad(lambda v: np.float64(v[0] * 3), x), [3.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(grad(lambda v: np.array(v[1] * 2.0), x), [0.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(grad(lambda v: 5, x), [0.0, 0.0], atol=1e-6)


# ---------- grad(): call counts and non-contiguous input ----------

def test_grad_calls_f_2n_times_for_central_and_n_plus_1_for_forward():
    x = np.array([1.0, 2.0, 3.0])
    central = _counting(_sum_of_squares)
    grad(central, x, method="central")
    assert len(central.calls) == 2 * 3

    forward = _counting(_sum_of_squares)
    grad(forward, x, method="forward")
    assert len(forward.calls) == 3 + 1


def test_grad_strided_view_matches_contiguous_copy():
    base = np.array([1.0, 99.0, 2.0, 99.0, 3.0])
    view = base[::2]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(
        grad(_sum_of_squares, view), grad(_sum_of_squares, view.copy())
    )
    # the original array must not be touched
    np.testing.assert_array_equal(base, np.array([1.0, 99.0, 2.0, 99.0, 3.0]))


# ---------- jacobian(): accuracy ----------

def test_jacobian_linear_map_matches_matrix_exactly():
    # Required by spec: jacobian on a known linear map matches its matrix exactly
    A = np.array([
        [2.0, 0.0, 1.0],
        [0.0, 3.0, -1.0],
    ])

    def F(x):
        return A @ x

    x = np.array([1.0, 1.0, 1.0])
    result = jacobian(F, x, method="central")
    # For a linear map, the finite-difference estimate should match A
    # almost exactly (only tiny floating-point noise, not truncation error,
    # since the true derivative of a linear function is constant everywhere).
    np.testing.assert_allclose(result, A, atol=1e-6)


def test_jacobian_shape_is_m_by_n():
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])  # 3 outputs, 2 inputs

    def F(x):
        return A @ x

    x = np.array([1.0, 1.0])
    result = jacobian(F, x)
    assert result.shape == (3, 2)


def test_jacobian_forward_method_matches_matrix():
    A = np.array([[1.0, -2.0], [3.0, 4.0]])

    def F(x):
        return A @ x

    x = np.array([2.0, -1.0])
    result = jacobian(F, x, method="forward")
    np.testing.assert_allclose(result, A, atol=1e-3)


def test_jacobian_empty_input_gives_m_by_zero_matrix():
    result = jacobian(lambda x: np.array([1.0, 2.0]), np.array([]))
    assert result.shape == (2, 0)


# ---------- jacobian(): errors ----------

def test_jacobian_rejects_non_1d_output():
    def F(x):
        return np.array([[1.0, 2.0]])  # 2D output, invalid

    with pytest.raises(ValueError, match="return a 1D array"):
        jacobian(F, np.array([1.0, 2.0]))


def test_jacobian_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1D array for x"):
        jacobian(lambda x: x, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_jacobian_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown method 'bogus'"):
        jacobian(lambda x: x, np.array([1.0, 2.0]), method="bogus")


@pytest.mark.parametrize("bad_h", [0, 0.0, -1e-5, np.inf, -np.inf, np.nan])
def test_jacobian_rejects_h_that_is_not_positive_and_finite(bad_h):
    with pytest.raises(ValueError, match="h must be finite and > 0"):
        jacobian(lambda x: x, np.array([1.0, 2.0]), h=bad_h)


@pytest.mark.parametrize("bad_h", ["1e-5", None, True, [1e-5]])
def test_jacobian_rejects_h_that_is_not_a_number(bad_h):
    with pytest.raises(TypeError, match="h must be a number"):
        jacobian(lambda x: x, np.array([1.0, 2.0]), h=bad_h)


# ---------- jacobian(): call counts and non-contiguous input ----------

def test_jacobian_calls_F_2n_plus_1_times_for_central_and_n_plus_1_for_forward():
    # F(x) is called once to learn the output size, so central differences
    # need 2n + 1 calls in total (not 2n), and forward differences n + 1.
    x = np.array([1.0, 2.0, 3.0])

    central = _counting(lambda v: v * 2.0)
    jacobian(central, x, method="central")
    assert len(central.calls) == 2 * 3 + 1

    forward = _counting(lambda v: v * 2.0)
    jacobian(forward, x, method="forward")
    assert len(forward.calls) == 3 + 1


def test_jacobian_strided_view_matches_contiguous_copy():
    A = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]])

    def F(x):
        return A @ x

    base = np.array([1.0, 99.0, 2.0, 99.0, 3.0])
    view = base[::2]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(jacobian(F, view), jacobian(F, view.copy()))