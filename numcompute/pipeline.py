"""
pipeline.py -- Chaining and combining transformers for NumCompute.

Provides:
    Pipeline(steps)
        -> chains (name, step) pairs: fit, transform, fit_transform and
           predict run X through every step in order; named_steps['name']
           and pipe['name'] / pipe[i] give access to individual steps
    FeatureUnion(transformers)
        -> applies every (name, transformer) to the same X and joins the
           outputs side by side with np.hstack

Requires:
    Python >= 3.9 and NumPy >= 1.23; no other numcompute modules. Steps are
    duck-typed: any object with the right methods works, with no base class
    to inherit from (the numcompute.preprocessing transformers all qualify).

    How this maps to the brief
    --------------------------
    The brief lists "Transformer/Estimator API, Compose" for this module.

    * Compose: ``Pipeline`` is the Compose abstraction. It composes steps one
    after another into a single object with fit / transform / fit_transform /
    predict. ``FeatureUnion`` is its side-by-side counterpart.
    * Transformer/Estimator API: the contract is duck-typed, so no new base
    class is needed (as the team spec says). A transformer is any object with
    callable ``fit(X[, y])`` and ``transform(X)``; an estimator is any object
    with callable ``fit(X[, y])`` and ``predict(X)``. ``Pipeline`` and
    ``FeatureUnion`` check for these methods when they are constructed.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Duck-typing contract helpers
# ----------------------------------------------------------------------------
def _missing_methods(obj, names) -> list:
    """Return the names in ``names`` that ``obj`` lacks as callable attributes."""
    return [m for m in names if not callable(getattr(obj, m, None))]


def _validate_named_steps(steps, owner: str) -> list:
    """Check ``steps`` is a non-empty sequence of ``(str, object)`` pairs with unique names.

    Returns
    -------
    list of tuple
        The steps as a new list (later edits to the caller's list do not
        affect the owner).

    Raises
    ------
    TypeError
        If ``steps`` is not a list/tuple of ``(name, step)`` pairs or a name
        is not a non-empty string.
    ValueError
        If ``steps`` is empty or two steps share a name.

    Complexity: O(k) for ``k`` steps.
    """
    if not isinstance(steps, (list, tuple)):
        raise TypeError(f"{owner} expects a list of (name, step) tuples, got {type(steps).__name__}")
    if len(steps) == 0:
        raise ValueError(f"{owner} needs at least one (name, step) tuple")

    out = []
    for i, item in enumerate(steps):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise TypeError(f"{owner} item {i} must be a (name, step) tuple, got {item!r}")
        name, step = item
        if not isinstance(name, str) or not name:
            raise TypeError(f"{owner} item {i}: name must be a non-empty str, got {name!r}")
        out.append((name, step))

    names = [n for n, _ in out]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{owner} step names must be unique; repeated: {duplicates}")
    return out


def _require(step, name: str, owner: str, methods, role: str) -> None:
    """Raise ``TypeError`` if ``step`` lacks any of ``methods``."""
    missing = _missing_methods(step, methods)
    if missing:
        raise TypeError(
            f"{owner} step {name!r} ({type(step).__name__}) must implement "
            f"{' and '.join(methods)}; missing: {', '.join(missing)}. {role}"
        )


def _fit(step, X, y):
    """Call ``step.fit``, passing ``y`` only when it is given."""
    return step.fit(X) if y is None else step.fit(X, y)


def _fit_transform(step, X, y):
    """Fit ``step`` and return its transform of ``X`` (uses fit_transform if present)."""
    if callable(getattr(step, "fit_transform", None)):
        return step.fit_transform(X) if y is None else step.fit_transform(X, y)
    _fit(step, X, y)
    return step.transform(X)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
class Pipeline:
    """Chain transformers, optionally ending in an estimator.

    Parameters
    ----------
    steps : list of (str, object)
        ``(name, step)`` pairs, applied in order. Every step but the last
        must have ``fit`` and ``transform``; the last needs ``fit`` and
        ``transform`` and/or ``predict``. Names must be unique.

    Attributes
    ----------
    steps : list of (str, object)
        A copy of the validated steps.
    named_steps : dict
        ``{name: step}``, e.g. ``pipe.named_steps['scale']``.

    Raises
    ------
    TypeError
        At construction, if ``steps`` is malformed or a step does not
        implement the methods its position requires.
    ValueError
        At construction, if ``steps`` is empty or names repeat.

    Examples
    --------
    >>> from numcompute.preprocessing import SimpleImputer, StandardScaler
    >>> pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    >>> X_tr = pipe.fit_transform([[1.0], [float("nan")], [3.0]])
    >>> pipe.named_steps["scale"].mean_
    array([2.])
    """

    def __init__(self, steps):
        steps = _validate_named_steps(steps, "Pipeline")
        for name, step in steps[:-1]:
            _require(step, name, "Pipeline", ("fit", "transform"),
                     "Every step except the last must be a transformer.")
        name, last = steps[-1]
        _require(last, name, "Pipeline", ("fit",), "The last step must be fittable.")
        if _missing_methods(last, ("transform",)) and _missing_methods(last, ("predict",)):
            raise TypeError(
                f"Pipeline step {name!r} ({type(last).__name__}) is the last step and must "
                "implement transform or predict; it has neither."
            )
        self.steps = steps

    # -- step access ---------------------------------------------------------
    @property
    def named_steps(self) -> dict:
        """``{name: step}`` for every step, in order.

        Returns
        -------
        dict of str to object
            A new dict on each access, in step order. The step objects are
            the pipeline's own (not copies), so ``pipe.named_steps['scale']``
            shows fitted attributes after ``fit``.
        """
        return dict(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, key):
        """``pipe['name']`` or ``pipe[i]`` returns a step; ``pipe[a:b]`` a sub-Pipeline."""
        if isinstance(key, slice):
            return Pipeline(self.steps[key])
        if isinstance(key, str):
            try:
                return self.named_steps[key]
            except KeyError:
                raise KeyError(f"no step named {key!r}; steps are {[n for n, _ in self.steps]}") from None
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
            raise TypeError(f"Pipeline indices must be str, int or slice, got {type(key).__name__}")
        return self.steps[key][1]

    def __repr__(self) -> str:
        inner = ", ".join(f"({n!r}, {s!r})" for n, s in self.steps)
        return f"Pipeline([{inner}])"

    # -- helpers -------------------------------------------------------------
    def _transform_until_last(self, X):
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X

    def _last(self, method: str, caller: str):
        name, last = self.steps[-1]
        if not callable(getattr(last, method, None)):
            raise TypeError(
                f"Pipeline.{caller} needs the last step to implement {method}, but step "
                f"{name!r} ({type(last).__name__}) does not."
            )
        return last

    # -- public API ----------------------------------------------------------
    def fit(self, X, y=None) -> "Pipeline":
        """Fit every step in order, feeding each the output of the previous one.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data for the first step.
        y : array-like, optional
            Targets, passed to every step's ``fit`` when not ``None``.

        Returns
        -------
        self : Pipeline

        Raises
        ------
        Whatever a step raises while fitting.

        Complexity
        ----------
        Sum of the steps' fit (and, for all but the last, transform) costs;
        only one intermediate array is alive at a time.
        """
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = _fit_transform(step, Xt, y)
        _fit(self.steps[-1][1], Xt, y)
        return self

    def transform(self, X):
        """Apply every step's ``transform`` in order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_out : array, shape (n_samples, n_features_out)
            Output of the last step.

        Raises
        ------
        TypeError
            If the last step has no ``transform`` (e.g. it is a model; use
            :meth:`predict`).
        NotFittedError
            (from the steps) if the pipeline has not been fitted.

        Complexity
        ----------
        Sum of the steps' transform costs.
        """
        last = self._last("transform", "transform")
        return last.transform(self._transform_until_last(X))

    def fit_transform(self, X, y=None):
        """Fit every step and return the transformed training data.

        Equivalent to ``self.fit(X, y).transform(X)`` but passes over the
        data once, using each step's ``fit_transform`` when it has one.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, optional
            Targets, passed to every step when not ``None``.

        Returns
        -------
        X_out : array, shape (n_samples, n_features_out)

        Raises
        ------
        TypeError
            If the last step has no ``transform``.

        Complexity
        ----------
        Sum of the steps' fit + transform costs.
        """
        last = self._last("transform", "fit_transform")
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = _fit_transform(step, Xt, y)
        return _fit_transform(last, Xt, y)

    def predict(self, X):
        """Transform ``X`` through all but the last step, then call ``predict`` on the last.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : array, shape (n_samples,) or as returned by the last step

        Raises
        ------
        TypeError
            If the last step has no ``predict``.
        NotFittedError
            (from the steps) if the pipeline has not been fitted.

        Complexity
        ----------
        Sum of the transform costs plus the model's predict cost.
        """
        last = self._last("predict", "predict")
        return last.predict(self._transform_until_last(X))


# ----------------------------------------------------------------------------
# FeatureUnion
# ----------------------------------------------------------------------------
class FeatureUnion:
    """Apply several transformers to the same ``X`` and stack their outputs.

    ``transform(X)`` returns
    ``np.hstack([t.transform(X) for _, t in transformers])``.

    Parameters
    ----------
    transformers : list of (str, object)
        ``(name, transformer)`` pairs; each must have ``fit`` and
        ``transform``. Names must be unique.

    Attributes
    ----------
    transformers : list of (str, object)
        A copy of the validated pairs.
    named_transformers : dict
        ``{name: transformer}``.

    Raises
    ------
    TypeError
        At construction, if ``transformers`` is malformed or a member lacks
        ``fit``/``transform``.
    ValueError
        At construction, if ``transformers`` is empty or names repeat.

    Notes
    -----
    Every output must be 2-D with the same number of rows; otherwise
    ``ValueError`` names the transformer at fault. Outputs are joined with
    ``np.hstack``, so NumPy's type promotion applies: joining a string
    output with a numeric one gives a string array. Keep the members'
    outputs numeric (e.g. one-hot encode categories) to get a float result.

    Examples
    --------
    >>> from numcompute.preprocessing import MinMaxScaler, StandardScaler
    >>> union = FeatureUnion([("z", StandardScaler()), ("mm", MinMaxScaler())])
    >>> union.fit_transform([[0.0], [2.0]])
    array([[-1.,  0.],
           [ 1.,  1.]])
    """

    def __init__(self, transformers):
        transformers = _validate_named_steps(transformers, "FeatureUnion")
        for name, t in transformers:
            _require(t, name, "FeatureUnion", ("fit", "transform"),
                     "Every FeatureUnion member must be a transformer.")
        self.transformers = transformers

    @property
    def named_transformers(self) -> dict:
        """``{name: transformer}`` for every member, in order.

        Returns
        -------
        dict of str to object
            A new dict on each access, in member order. The transformer
            objects are the union's own (not copies), so fitted attributes
            are visible after ``fit``.
        """
        return dict(self.transformers)

    def __repr__(self) -> str:
        inner = ", ".join(f"({n!r}, {t!r})" for n, t in self.transformers)
        return f"FeatureUnion([{inner}])"

    def _stack(self, outputs, n_rows: int):
        """Check every output is 2-D with ``n_rows`` rows, then ``np.hstack`` them."""
        for (name, _), out in zip(self.transformers, outputs):
            shape = np.shape(out)
            if len(shape) != 2:
                raise ValueError(
                    f"FeatureUnion member {name!r} returned a {len(shape)}-D output with "
                    f"shape {shape}; every member must return a 2-D array"
                )
            if shape[0] != n_rows:
                raise ValueError(
                    f"FeatureUnion member {name!r} returned {shape[0]} rows, expected "
                    f"{n_rows} (the number of rows in X)"
                )
        return np.hstack(outputs)

    def fit(self, X, y=None) -> "FeatureUnion":
        """Fit every member on the same ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, optional
            Passed to every member's ``fit`` when not ``None``.

        Returns
        -------
        self : FeatureUnion

        Raises
        ------
        Exception
            Any error raised by a member's ``fit`` is passed through unchanged
            (for example ``ValueError`` or ``TypeError`` for bad input).
            Members fitted before the failing one stay fitted.

        Complexity
        ----------
        Sum of the members' fit costs.
        """
        for _, t in self.transformers:
            _fit(t, X, y)
        return self

    def transform(self, X):
        """Transform ``X`` with every member and join the results column-wise.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray, shape (n_samples, sum of members' output widths)
            Member outputs side by side, in the order given.

        Raises
        ------
        ValueError
            If a member returns a non-2-D output or the wrong number of rows.
        NotFittedError
            (from the members) if the union has not been fitted.

        Complexity
        ----------
        Sum of the members' transform costs plus O(n · total width) to stack.
        """
        n_rows = np.shape(X)[0]
        return self._stack([t.transform(X) for _, t in self.transformers], n_rows)

    def fit_transform(self, X, y=None):
        """Fit every member and return the stacked transformed ``X``.

        Equivalent to ``self.fit(X, y).transform(X)``, using each member's
        ``fit_transform`` when it has one.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, optional

        Returns
        -------
        X_out : ndarray, shape (n_samples, sum of members' output widths)

        Raises
        ------
        ValueError
            If a member returns a non-2-D output or the wrong number of rows.

        Complexity
        ----------
        Sum of the members' fit + transform costs plus the stacking cost.
        """
        n_rows = np.shape(X)[0]
        return self._stack([_fit_transform(t, X, y) for _, t in self.transformers], n_rows)
