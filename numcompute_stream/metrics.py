"""Streaming classification metrics.

Every metric is updated chunk by chunk and has the same three methods:

    metric.update(y_true_chunk, y_pred_chunk)   add one chunk
    metric.result()                             value for everything seen so far
    metric.reset()                              start again

Accuracy, precision, recall and F1 all keep one accumulated confusion matrix,
so the streaming result is exactly the value you would get from all the data
at once (before any update, result() is NaN). The scores come from the
Assignment 2.1 code, which is reused and not changed: each chunk is counted
with ``confusion_matrix``, and the averaging (``"binary"``, ``"macro"``,
``"micro"``) uses the same helper as the 2.1 ``precision`` / ``recall`` /
``f1``. The 2.1 conventions hold: ``ConfusionMatrix.result()`` returns
``(matrix, labels)``, macro-F1 is the mean of the per-class F1 scores, and a
class with no predictions scores 0.
"""

import copy

import numpy as np

from numcompute.metrics import _VALID_AVERAGES, _average_ratio, _has_nan, auc, confusion_matrix  # Assignment 2.1
from numcompute.preprocessing import _kind_group                                                  # Assignment 2.1
from numcompute.utils import _is_integer                                                          # Assignment 2.1


def _check_pair(y_true, y_pred):
    """Both inputs must be 1-D and the same length."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must be 1-D with the same length, got shapes "
                         f"{y_true.shape} and {y_pred.shape}.")
    return y_true, y_pred


class ConfusionMatrix:
    """Counts of (true label, predicted label) pairs, accumulated over time.

    ``result()`` returns ``(matrix, labels)``: rows are true labels and
    columns predicted labels, in sorted label order. New labels can appear at
    any time - the matrix grows to include them. Each chunk is counted with
    the vectorised 2.1 ``confusion_matrix`` (which also rejects NaN labels).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.labels = None
        self.matrix = None
        return self

    def update(self, y_true, y_pred):
        y_true, y_pred = _check_pair(y_true, y_pred)
        if _has_nan(y_true) or _has_nan(y_pred):  # checked before any state changes
            raise ValueError("y_true and y_pred must not contain NaN.")
        if len(y_true) == 0:
            return self
        seen = np.unique(np.concatenate([y_true, y_pred]))
        if self.labels is None:
            self.labels = seen
            self.matrix = np.zeros((len(seen), len(seen)), dtype=int)
        elif not np.isin(seen, self.labels).all():  # a new label: grow the matrix
            if {_kind_group(seen.dtype), _kind_group(self.labels.dtype)} == {"text", "number"}:
                raise ValueError("Labels must all be text or all be numbers.")  # NumPy would turn numbers into text
            labels = np.union1d(self.labels, seen)
            matrix = np.zeros((len(labels), len(labels)), dtype=int)
            old = np.searchsorted(labels, self.labels)  # where the old labels moved to
            matrix[np.ix_(old, old)] = self.matrix
            self.labels, self.matrix = labels, matrix
        chunk_matrix, _ = confusion_matrix(y_true, y_pred, labels=self.labels)
        self.matrix += chunk_matrix
        return self

    def result(self):
        if self.labels is None:
            return np.zeros((0, 0), dtype=int), np.array([])
        return self.matrix.copy(), self.labels.copy()


class _FromConfusionMatrix:
    """Shared code: the metrics below are all computed from an accumulated confusion matrix."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.confusion = ConfusionMatrix()
        return self

    def update(self, y_true, y_pred):
        self.confusion.update(y_true, y_pred)
        return self

    def result(self):
        matrix, labels = self.confusion.result()
        if matrix.sum() == 0:
            return np.nan
        true_positives = np.diag(matrix).astype(float)
        predicted_totals = matrix.sum(axis=0).astype(float)  # how often each label was predicted
        actual_totals = matrix.sum(axis=1).astype(float)     # how often each label really occurred
        return self._score(true_positives, predicted_totals, actual_totals, labels)


class Accuracy(_FromConfusionMatrix):
    """Share of correct predictions: the diagonal of the confusion matrix over its total."""

    def _score(self, true_positives, predicted_totals, actual_totals, labels):
        return float(true_positives.sum() / actual_totals.sum())


class _Averaged(_FromConfusionMatrix):
    """Precision, recall and F1, averaged exactly like the 2.1 functions.

    average : {"binary", "macro", "micro"}, default "macro"
        "binary" scores ``pos_label`` only (at most 2 labels), "macro" is the
        mean over all labels seen, "micro" uses the summed counts.
    """

    def __init__(self, average="macro", pos_label=1):
        if average not in _VALID_AVERAGES:
            raise ValueError(f"average must be one of {_VALID_AVERAGES}, got {average!r}.")
        self.average = average
        self.pos_label = pos_label
        super().__init__()

    def _ratio(self, numerator, denominator, labels):
        return _average_ratio(numerator, denominator, self.average, labels, self.pos_label, type(self).__name__)


class Precision(_Averaged):
    """TP / (TP + FP) = true positives / times predicted."""

    def _score(self, true_positives, predicted_totals, actual_totals, labels):
        return self._ratio(true_positives, predicted_totals, labels)


class Recall(_Averaged):
    """TP / (TP + FN) = true positives / times the label occurred."""

    def _score(self, true_positives, predicted_totals, actual_totals, labels):
        return self._ratio(true_positives, actual_totals, labels)


class F1(_Averaged):
    """F1 = 2TP / (2TP + FP + FN) per label (equal to 2PR / (P + R), as in the 2.1 ``f1``).

    Macro-F1 is the mean of the per-label F1 scores; micro-F1 equals the accuracy.
    """

    def _score(self, true_positives, predicted_totals, actual_totals, labels):
        return self._ratio(2 * true_positives, predicted_totals + actual_totals, labels)


class AUC:
    """Area under the ROC curve (binary), accumulated over time.

    ``update(y_true, y_score)`` takes the predicted probability of
    ``pos_label``. Scores are counted in ``n_bins`` bins between 0 and 1, so
    the memory stays small however long the stream is. ``result()`` builds the
    ROC curve from these counts and measures its area with the 2.1 ``auc``
    (trapezoidal rule), so scores in the same bin count as ties (half a
    point), as in the rank-based definition of AUC.
    """

    def __init__(self, pos_label=1, n_bins=1000):
        if not _is_integer(n_bins) or n_bins < 1:
            raise ValueError(f"n_bins must be an integer >= 1, got {n_bins!r}.")
        self.pos_label = pos_label
        self.n_bins = n_bins
        self.reset()

    def reset(self):
        self.positive = np.zeros(self.n_bins)  # score histogram of positive samples
        self.negative = np.zeros(self.n_bins)  # score histogram of negative samples
        return self

    def update(self, y_true, y_score):
        y_true, y_score = _check_pair(y_true, y_score)
        y_score = y_score.astype(float)
        if _has_nan(y_true):
            raise ValueError("y_true must not contain NaN.")
        if not np.all((y_score >= 0) & (y_score <= 1)):  # also rejects NaN
            raise ValueError("y_score must contain probabilities between 0 and 1 (no NaN).")
        bins = np.minimum((y_score * self.n_bins).astype(int), self.n_bins - 1)
        is_positive = y_true == self.pos_label
        self.positive += np.bincount(bins[is_positive], minlength=self.n_bins)
        self.negative += np.bincount(bins[~is_positive], minlength=self.n_bins)
        return self

    def result(self):
        n_positive, n_negative = self.positive.sum(), self.negative.sum()
        if n_positive == 0 or n_negative == 0:
            return np.nan
        # ROC curve: lower the threshold one bin at a time, from the highest scores down.
        tpr = np.concatenate([[0.0], np.cumsum(self.positive[::-1]) / n_positive])
        fpr = np.concatenate([[0.0], np.cumsum(self.negative[::-1]) / n_negative])
        return auc(fpr, tpr)


class RollingMetric:
    """Any metric above, computed over the most recent ``window`` samples only.

    The last ``window`` (y_true, y_pred) pairs are kept, and ``result()``
    recomputes a fresh copy of the wrapped metric on them.

    Example
    -------
    >>> rolling_f1 = RollingMetric(F1(), window=500)
    """

    def __init__(self, metric, window=500):
        if not _is_integer(window) or window < 1:
            raise ValueError(f"window must be an integer >= 1, got {window!r}.")
        self.metric = metric
        self.window = window
        self.reset()

    def reset(self):
        self.y_true = None
        self.y_pred = None
        return self

    def update(self, y_true, y_pred):
        y_true, y_pred = _check_pair(y_true, y_pred)
        if self.y_true is not None:
            y_true = np.concatenate([self.y_true, y_true])
            y_pred = np.concatenate([self.y_pred, y_pred])
        self.y_true, self.y_pred = y_true[-self.window:], y_pred[-self.window:]
        return self

    def result(self):
        if self.y_true is None or len(self.y_true) == 0:
            return np.nan
        return copy.deepcopy(self.metric).reset().update(self.y_true, self.y_pred).result()
