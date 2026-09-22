"""
optim.py
Finite-difference gradient and Jacobian estimation.

Provides
--------
- grad(f, x, h=1e-5, method='central'|'forward')
- jacobian(F, x, h=1e-5, method='central')

Design notes
------------
Both functions nudge one input dimension at a time and measure how
the output changes -- this is the standard finite-difference approach
to estimating derivatives numerically, without knowing the function's
actual formula.

h trade-off (documented per assignment requirement)
----------------------------------------------------
`h` controls how far we nudge each input to measure the slope. There
are two competing sources of error:

- Too LARGE an h: truncation error. The finite-difference formula is
  only an approximation of the true derivative; larger steps make
  that approximation worse (we're measuring the AVERAGE slope over
  too wide a gap, not the slope exactly at the point).
- Too SMALL an h: rounding error. Floating-point numbers have limited
  precision. If h is smaller than the precision of the numbers
  involved, f(x + h) and f(x) may come out numerically identical
  (or the difference gets swamped by floating-point noise), even
  though the true derivative is not zero.

The default h=1e-5 is a reasonable middle ground for central
differences in double precision. Both grad() and jacobian() scale h
per-dimension relative to abs(x[i]) (h_i = h * max(1.0, abs(x[i]))),
rather than using one fixed absolute step for every input. This keeps
the perturbation meaningful in relative terms regardless of an
input's magnitude -- a fixed h=1e-5 step is far too small to register
against an input like x[i]=1e8 (the change gets lost in
floating-point rounding), and needlessly large relative to an input
like x[i]=1e-8.
"""

import numpy as np


def grad(f, x, h=1e-5, method="central"):
    """Estimate the gradient of a scalar-valued function via finite differences.

    Parameters
    ----------
    f : callable
        A function that takes a 1D array of shape (n,) and returns a
        single scalar (float).
    x : array_like, shape (n,)
        The point at which to estimate the gradient.
    h : float, default 1e-5
        Step size used to perturb each input dimension. See the module
        docstring for the rounding-vs-truncation error trade-off.
    method : {'central', 'forward'}, default 'central'
        - 'central': (f(x+h) - f(x-h)) / (2h) -- more accurate (error
          shrinks with h^2), but costs 2 function evaluations per
          dimension.
        - 'forward': (f(x+h) - f(x)) / h -- cheaper (1 extra
          evaluation per dimension, reusing f(x)), but less accurate
          (error shrinks only with h).

    Returns
    -------
    gradient : ndarray, shape (n,)
        Estimated partial derivative of f with respect to each input
        dimension.

    Raises
    ------
    ValueError
        If `method` is not 'central' or 'forward', or if `x` is not 1D.

    Complexity
    ----------
    Time: O(n) function evaluations (2n for central, n+1 for forward),
    each costing whatever f itself costs. Space: O(n).
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"grad() expects a 1D array for x, got {x.ndim}D")

    if method not in ("central", "forward"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'central' or 'forward'"
        )

    n = x.size
    gradient = np.empty(n, dtype=float)

    # A Python loop over dimensions is expected here: each dimension
    # needs its own independent perturbation of x, so this cannot be
    # vectorised away without calling f in a way it may not support
    # (e.g. f may only accept one x at a time, not a batch of them).
    if method == "forward":
        f_x = f(x)  # reused for every dimension, avoids recomputing it n times

    for i in range(n):
        # Scale h relative to the magnitude of this input dimension.
        # A fixed h=1e-5 is too small a nudge for a large x[i] (the
        # relative change gets lost in floating-point noise) and
        # needlessly large for a tiny x[i]. max(1.0, abs(x[i])) keeps
        # the step at least h in absolute size, but grows it
        # proportionally for large inputs.
        h_i = h * max(1.0, abs(x[i]))

        x_plus = x.copy()
        x_plus[i] += h_i

        if method == "central":
            x_minus = x.copy()
            x_minus[i] -= h_i
            gradient[i] = (f(x_plus) - f(x_minus)) / (2 * h_i)
        else:  # forward
            gradient[i] = (f(x_plus) - f_x) / h_i

    return gradient


def jacobian(F, x, h=1e-5, method="central"):
    """Estimate the Jacobian matrix of a vector-valued function via finite differences.

    Parameters
    ----------
    F : callable
        A function that takes a 1D array of shape (n,) and returns a
        1D array of shape (m,) (m outputs).
    x : array_like, shape (n,)
        The point at which to estimate the Jacobian.
    h : float, default 1e-5
        Step size used to perturb each input dimension. See the module
        docstring for the rounding-vs-truncation error trade-off.
    method : {'central', 'forward'}, default 'central'
        Same trade-off as in grad(): 'central' is more accurate but
        costs twice as many evaluations of F.

    Returns
    -------
    J : ndarray, shape (m, n)
        J[j, i] is the estimated partial derivative of the j-th output
        of F with respect to the i-th input dimension.

    Raises
    ------
    ValueError
        If `method` is invalid, `x` is not 1D, or F(x) does not return
        a 1D array.

    Complexity
    ----------
    Time: O(n) evaluations of F (2n for central, n+1 for forward), each
    costing O(m) to produce its output vector. Space: O(m * n) for the
    output matrix.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"jacobian() expects a 1D array for x, got {x.ndim}D")

    if method not in ("central", "forward"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'central' or 'forward'"
        )

    n = x.size
    f_x0 = np.asarray(F(x), dtype=float)

    if f_x0.ndim != 1:
        raise ValueError(
            f"jacobian() expects F(x) to return a 1D array, got {f_x0.ndim}D"
        )

    m = f_x0.size
    J = np.empty((m, n), dtype=float)

    # Same per-dimension loop pattern as grad(): each column of J is
    # one dimension's worth of perturbation, stacked into an (m, n)
    # matrix as the spec requires. h is scaled per-dimension the same
    # way as in grad() -- see that function's comment for why.
    for i in range(n):
        h_i = h * max(1.0, abs(x[i]))

        x_plus = x.copy()
        x_plus[i] += h_i

        if method == "central":
            x_minus = x.copy()
            x_minus[i] -= h_i
            J[:, i] = (np.asarray(F(x_plus), dtype=float) - np.asarray(F(x_minus), dtype=float)) / (2 * h_i)
        else:  # forward
            J[:, i] = (np.asarray(F(x_plus), dtype=float) - f_x0) / h_i

    return J