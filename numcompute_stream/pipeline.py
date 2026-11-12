"""Pipeline: the Assignment 2.1 Pipeline plus streaming (``partial_fit``).

``numcompute.pipeline.Pipeline`` (reused, not changed) already checks the
steps, chains ``fit`` / ``transform`` / ``predict`` and supports
``pipe["name"]``, ``pipe[:-1]`` and ``len(pipe)``. This subclass adds what a
stream needs: ``partial_fit``, ``predict_proba``, ``classes_`` and
``is_fitted``.
"""

from numcompute.pipeline import Pipeline as BasePipeline  # Assignment 2.1 (reused)


class Pipeline(BasePipeline):
    """A list of (name, step) pairs - transformers followed by a model - trained chunk by chunk.

    ``partial_fit(X_chunk, y_chunk)`` updates every step in turn. Each
    transformer first updates its running statistics with the chunk and then
    transforms the chunk for the next step ("update, then transform"), so a
    chunk is always transformed with statistics that already include it.

    Example (from the specification)
    --------------------------------
    >>> pipe = Pipeline([
    ...     ("scale", StandardScaler()),
    ...     ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
    ... ])
    >>> pipe.partial_fit(X_chunk, y_chunk)
    """

    def __init__(self, steps):
        super().__init__(steps)  # the 2.1 checks: names, transformers, last step
        for name, step in self.steps:
            if not callable(getattr(step, "partial_fit", None)):
                raise TypeError(f"Pipeline step {name!r} ({type(step).__name__}) has no partial_fit method, "
                                "so the pipeline cannot learn from a stream.")

    def partial_fit(self, X, y=None):
        """Update every step with one chunk."""
        for _, transformer in self.steps[:-1]:
            transformer.partial_fit(X, y)
            X = transformer.transform(X)
        self.steps[-1][1].partial_fit(X, y)
        return self

    def predict_proba(self, X):
        """Class probabilities from the last step, after the transformers."""
        last = self._last("predict_proba", "predict_proba")
        return last.predict_proba(self._transform_until_last(X))

    @property
    def classes_(self):
        return self.steps[-1][1].classes_

    @property
    def is_fitted(self):
        """False while the last step cannot predict yet (e.g. a tree still filling its first buffer)."""
        return bool(getattr(self.steps[-1][1], "is_fitted", True))
