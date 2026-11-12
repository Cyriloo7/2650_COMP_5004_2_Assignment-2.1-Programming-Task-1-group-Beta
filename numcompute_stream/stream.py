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

from .metrics import F1, Accuracy, RollingMetric


def model_size_kb(model):
    """Memory footprint of a model (including any data buffer), as its pickled size in KB."""
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
    metrics : dict of name -> metric, optional
        Streaming metrics from ``metrics.py`` (default: accuracy and macro-F1).
    window : int, default 500
        Number of most recent samples used by the rolling metrics.
    verbose : bool, default False
        Print one line per chunk.

    After training, ``history`` holds one dict per chunk with, for every
    metric: its value on the chunk, ``cumulative_<name>`` (all scored chunks
    so far) and ``rolling_<name>`` (last ``window`` samples), plus
    ``train_time`` (seconds) and ``memory_kb``. For a buffered tree it also
    logs ``retrained`` (1 if the tree was retrained on this chunk),
    ``retrain_time`` (seconds, NaN otherwise), ``n_retrains`` and
    ``buffered`` (samples waiting in the buffer).
    """

    def __init__(self, model, metrics=None, window=500, verbose=False):
        if not (hasattr(model, "partial_fit") and hasattr(model, "predict")):
            raise TypeError("model must have partial_fit and predict methods.")
        self.model = model
        self.metrics = metrics if metrics is not None else {"accuracy": Accuracy(), "f1": F1()}
        self.rolling = {name: RollingMetric(copy.deepcopy(metric), window) for name, metric in self.metrics.items()}
        self.verbose = verbose
        self.history = []
        self.n_samples_seen = 0
        self.last_y_true = self.last_y_pred = None
        self._trained = False  # becomes True after the first fit_chunk

    @property
    def model_ready(self):
        """True once the model can predict: it has been given data, and (for a buffered
        tree) its first buffer has been filled and trained on."""
        return self._trained and bool(getattr(self.model, "is_fitted", True))

    def fit_chunk(self, X, y):
        """Train on one chunk; return the training time, model size and retraining information."""
        final = _final_step(self.model)
        retrains_before = getattr(final, "n_retrains_", None)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        record = {"train_time": time.perf_counter() - start, "memory_kb": model_size_kb(self.model)}
        self._trained = True
        if retrains_before is not None:  # a buffered model: log when it retrained
            retrained = final.n_retrains_ > retrains_before
            record.update({"retrained": int(retrained), "n_retrains": final.n_retrains_,
                           "retrain_time": final.last_train_time_ if retrained else np.nan,
                           "buffered": final.n_buffered_})
        self.n_samples_seen += len(y)
        return record

    def score_chunk(self, X, y):
        """Test the current model on one chunk, update every metric and return the values."""
        if not self.model_ready:
            raise NotFittedError("The model cannot predict yet (e.g. the tree's first buffer is not full); "
                                 "call fit_chunk first.")
        y_pred = self.model.predict(X)
        self.last_y_true, self.last_y_pred = np.asarray(y), y_pred
        scores = {}
        for name, metric in self.metrics.items():
            scores[name] = copy.deepcopy(metric).reset().update(y, y_pred).result()  # this chunk only
            scores[f"cumulative_{name}"] = metric.update(y, y_pred).result()
            scores[f"rolling_{name}"] = self.rolling[name].update(y, y_pred).result()
        return scores

    def process_chunk(self, X, y):
        """Test on the chunk (if the model can predict yet), then train on it."""
        record = {"chunk": len(self.history), "n_samples": len(y)}
        if self.model_ready:
            record.update(self.score_chunk(X, y))
        record.update(self.fit_chunk(X, y))
        self.history.append(record)
        if self.verbose:
            print(self._describe(record))
        return record

    def run(self, chunks):
        """Process every (X_chunk, y_chunk) pair of an iterable, in order."""
        for X, y in chunks:
            self.process_chunk(X, y)
        return self

    def get(self, key):
        """Values of one logged quantity over all chunks (NaN where it was not logged)."""
        return np.array([record.get(key, np.nan) for record in self.history], dtype=float)

    def _describe(self, record):
        text = f"chunk {record['chunk']:3d} | n={record['n_samples']}"
        name = next(iter(self.metrics))
        if name in record:
            text += (f" | {name} {record[name]:.3f} | cumulative {record['cumulative_' + name]:.3f}"
                     f" | rolling {record['rolling_' + name]:.3f}")
        else:
            text += " | not scored (no model yet)"
        if record.get("retrained"):
            text += f" | retrained in {1000 * record['retrain_time']:.1f} ms"
        elif "buffered" in record:
            text += f" | buffered {record['buffered']}"
        return text + f" | model {record['memory_kb']:.0f} KB"
