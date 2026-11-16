"""StreamTrainer: train a model or pipeline chunk by chunk and log the results.

For each chunk the trainer first *tests* the current model on it and then
*trains* on it (test-then-train, also called prequential evaluation). Every
sample is predicted before the model has learned from it, so the logged
scores are honest estimates of how the model does on new data.

A buffered decision tree cannot predict until its first buffer is full, so
chunks that arrive before that are used for training only (their scores are
logged as NaN).
"""

import copy
import pickle
import time

import numpy as np

from numcompute.preprocessing import NotFittedError  # Assignment 2.1 (reused)

from .metrics import F1, Accuracy, ConfusionMatrix, RollingMetric


def model_size_kb(model):
    """Memory footprint of a model, including any data buffer, as its pickled size.

    Parameters
    ----------
    model : object
        Any picklable model or pipeline.

    Returns
    -------
    float
        Size in kilobytes (1 KB = 1024 bytes).
    """
    return len(pickle.dumps(model)) / 1024


def _final_step(model):
    """The model itself, or the last step of a Pipeline."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


class StreamTrainer:
    """Train a model on a stream of chunks and log metrics, time, memory and retraining.

    Parameters
    ----------
    model : object with partial_fit and predict
        A model, or a Pipeline that ends with a model.
    metrics : dict of str -> metric, optional
        Streaming metrics from ``metrics.py`` (default: accuracy and
        macro-F1). Label metrics get ``model.predict(X)``; metrics with
        ``needs_scores`` (``AUC``) get the ``predict_proba`` column of their
        ``pos_label``, so the model must then have ``predict_proba``.
        ``ConfusionMatrix`` is not accepted: its result is a matrix, not a number.
    window : int, default 500
        Number of most recent samples used by the rolling metrics.
    verbose : bool, default False
        Print one line per chunk.

    Attributes
    ----------
    history : list of dict
        One record per chunk with ``chunk``, ``n_samples``, ``train_time``
        (seconds) and ``memory_kb``; for every metric its value on the chunk,
        ``cumulative_<name>`` (all scored chunks so far) and ``rolling_<name>``
        (last ``window`` samples). For a buffered tree also ``retrained``
        (1 if the tree was retrained on this chunk), ``retrain_time`` (seconds,
        NaN otherwise), ``n_retrains`` and ``buffered`` (samples waiting).
    n_samples_seen : int
        Samples trained on so far.
    last_y_true, last_y_pred : ndarray of shape (n_samples,) or None
        True and predicted labels of the latest scored chunk.

    Example
    -------
    >>> trainer = StreamTrainer(pipe, metrics={"accuracy": Accuracy(), "auc": AUC(pos_label="yes")})
    >>> trainer.run(iter_chunks(X, y, chunk_size=25))
    >>> trainer.get("rolling_accuracy")
    """

    def __init__(self, model, metrics=None, window=500, verbose=False):
        if not (hasattr(model, "partial_fit") and hasattr(model, "predict")):
            raise TypeError("model must have partial_fit and predict methods.")
        metrics = {"accuracy": Accuracy(), "f1": F1()} if metrics is None else dict(metrics)
        for name, metric in metrics.items():
            if not all(callable(getattr(metric, method, None)) for method in ("update", "result", "reset")):
                raise TypeError(f"metric {name!r} must have update, result and reset methods.")
            if isinstance(metric, ConfusionMatrix):
                raise TypeError(f"metric {name!r}: ConfusionMatrix.result() is a matrix, not a number; keep it "
                                "outside the trainer (e.g. update it with last_y_true and last_y_pred).")
            if getattr(metric, "needs_scores", False) and not callable(getattr(model, "predict_proba", None)):
                raise TypeError(f"metric {name!r} needs predicted probabilities, but the model has no "
                                "predict_proba method.")
        self.model = model
        self.metrics = metrics
        self.rolling = {name: RollingMetric(copy.deepcopy(metric), window) for name, metric in metrics.items()}
        self.verbose = verbose
        self.history = []
        self.n_samples_seen = 0
        self.last_y_true = self.last_y_pred = None
        self._trained = False  # becomes True after the first non-empty chunk is trained on

    @property
    def model_ready(self):
        """True once the model can predict: it has learned from data, and (for a
        buffered tree) its first buffer has been filled and trained on."""
        return self._trained and bool(getattr(self.model, "is_fitted", True))

    def fit_chunk(self, X, y):
        """Train the model on one chunk with ``partial_fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        dict
            ``train_time`` (seconds) and ``memory_kb``; for a buffered model
            also ``retrained``, ``n_retrains``, ``retrain_time`` and ``buffered``.
        """
        final = _final_step(self.model)
        retrains_before = getattr(final, "n_retrains_", None)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        record = {"train_time": time.perf_counter() - start, "memory_kb": model_size_kb(self.model)}
        self._trained = self._trained or len(y) > 0
        if retrains_before is not None:  # a buffered model: log when it retrained
            retrained = final.n_retrains_ > retrains_before
            record.update({"retrained": int(retrained), "n_retrains": final.n_retrains_,
                           "retrain_time": final.last_train_time_ if retrained else np.nan,
                           "buffered": final.n_buffered_})
        self.n_samples_seen += len(y)
        return record

    def score_chunk(self, X, y):
        """Test the current model on one chunk and update every metric.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            True labels of the chunk.

        Returns
        -------
        dict of str -> float
            For every metric ``name``: ``name`` (this chunk only),
            ``cumulative_<name>`` and ``rolling_<name>``.

        Raises
        ------
        NotFittedError
            If the model cannot predict yet.
        """
        if not self.model_ready:
            raise NotFittedError("The model cannot predict yet (e.g. the tree's first buffer is not full); "
                                 "call fit_chunk first.")
        y = np.asarray(y)
        y_pred = self.model.predict(X)
        self.last_y_true, self.last_y_pred = y, y_pred
        proba = None
        scores = {}
        for name, metric in self.metrics.items():
            if getattr(metric, "needs_scores", False):  # e.g. AUC: probabilities of its pos_label
                proba = self.model.predict_proba(X) if proba is None else proba
                output = self._column_of(proba, metric.pos_label)
            else:
                output = y_pred
            scores[name] = copy.deepcopy(metric).reset().update(y, output).result()  # this chunk only
            scores[f"cumulative_{name}"] = metric.update(y, output).result()
            scores[f"rolling_{name}"] = self.rolling[name].update(y, output).result()
        return scores

    def process_chunk(self, X, y):
        """Test on the chunk (if the model can predict yet), then train on it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        dict
            The record appended to ``history``.
        """
        record = {"chunk": len(self.history), "n_samples": len(y)}
        if self.model_ready:
            record.update(self.score_chunk(X, y))
        record.update(self.fit_chunk(X, y))
        self.history.append(record)
        if self.verbose:
            print(self._describe(record))
        return record

    def run(self, chunks):
        """Process every chunk of a stream, in order.

        Parameters
        ----------
        chunks : iterable of (X_chunk, y_chunk)
            For example ``iter_chunks(X, y, chunk_size)``.

        Returns
        -------
        self
        """
        for X, y in chunks:
            self.process_chunk(X, y)
        return self

    def get(self, key):
        """Values of one logged quantity over all chunks.

        Parameters
        ----------
        key : str
            A key of the records, e.g. ``"accuracy"``, ``"rolling_f1"`` or ``"memory_kb"``.

        Returns
        -------
        ndarray of shape (n_chunks,)
            NaN where the quantity was not logged (e.g. before the first retrain).
        """
        return np.array([record.get(key, np.nan) for record in self.history], dtype=float)

    def _column_of(self, proba, label):
        """Probabilities of ``label`` (all 0 if the model has not seen that label yet)."""
        classes = list(self.model.classes_)
        return proba[:, classes.index(label)] if label in classes else np.zeros(len(proba))

    def _describe(self, record):
        """One log line for a chunk record (used when ``verbose``)."""
        text = f"chunk {record['chunk']:3d} | n={record['n_samples']}"
        name = next(iter(self.metrics), None)  # the first metric is shown (None: no metrics to show)
        if name in record:
            text += (f" | {name} {record[name]:.3f} | cumulative {record['cumulative_' + name]:.3f}"
                     f" | rolling {record['rolling_' + name]:.3f}")
        elif name is not None:
            text += " | not scored (no model yet)"
        if record.get("retrained"):
            text += f" | retrained in {1000 * record['retrain_time']:.1f} ms"
        elif "buffered" in record:
            text += f" | buffered {record['buffered']}"
        return text + f" | model {record['memory_kb']:.0f} KB"
