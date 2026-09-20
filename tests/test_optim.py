"""
tests/test_optim.py
Unit tests for numcompute.optim

Run with: pytest tests/test_optim.py
(Falls back to plain asserts + a __main__ runner if pytest isn't installed.)
"""

import numpy as np

from numcompute.optim import grad, jacobian


# ---------- grad() tests ----------

def test_grad_sum_of_squares_matches_2x_central():
    # Required by spec: grad on f(x) = sum(x**2) matches 2*x within tolerance
    def f(x):
        return np.sum(x ** 2)

    x = np.array([1.0, 2.0, 3.0])
    result = grad(f, x, method="central")
    expected = 2 * x
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_grad_sum_of_squares_matches_2x_forward():
    def f(x):
        return np.sum(x ** 2)

    x = np.array([1.0, 2.0, 3.0])
    result = grad(f, x, method="forward")
    expected = 2 * x
    # forward differences are less accurate (O(h) error), so use a looser tolerance
    np.testing.assert_allclose(result, expected, atol=1e-3)


def test_grad_negative_and_zero_values():
    def f(x):
        return np.sum(x ** 2)

    x = np.array([0.0, -2.0, 5.0])
    result = grad(f, x, method="central")
    expected = 2 * x
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_grad_invalid_method_raises():
    def f(x):
        return np.sum(x)

    try:
        grad(f, np.array([1.0, 2.0]), method="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_grad_rejects_non_1d_input():
    def f(x):
        return np.sum(x)

    try:
        grad(f, np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_grad_scales_h_for_large_magnitude_input():
    # Required by spec: "consider scaling h relative to abs(x[i])".
    # A fixed, unscaled h=1e-5 breaks down completely for large x
    # (rounding error swamps the perturbation). With scaling, the
    # estimate should stay accurate even at x = 1e12.
    def f(x):
        return np.sum(x ** 2)

    x = np.array([1e12])
    result = grad(f, x, h=1e-5, method="central")
    expected = 2 * x
    relative_error = abs(result[0] - expected[0]) / expected[0]
    assert relative_error < 1e-8, (
        f"expected h-scaling to keep relative error small, got {relative_error:.2e}"
    )


# ---------- jacobian() tests ----------

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


def test_jacobian_rejects_non_1d_output():
    def F(x):
        return np.array([[1.0, 2.0]])  # 2D output, invalid

    try:
        jacobian(F, np.array([1.0, 2.0]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_jacobian_invalid_method_raises():
    def F(x):
        return x

    try:
        jacobian(F, np.array([1.0, 2.0]), method="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


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