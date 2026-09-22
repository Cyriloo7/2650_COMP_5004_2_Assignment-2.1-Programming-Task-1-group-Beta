"""
metrics.py -- Classification and regression metrics for NumCompute.

Provides:
    confusion_matrix(y_true, y_pred, labels=None) -> (matrix, labels)
    accuracy(y_true, y_pred)
    precision(y_true, y_pred, average='binary'|'macro'|'micro', pos_label=1)
    recall(y_true, y_pred, average='binary'|'macro'|'micro', pos_label=1)
    f1(y_true, y_pred, average='binary'|'macro'|'micro', pos_label=1)
    mse(y_true, y_pred)
    roc_curve(y_true, y_score)   -- bonus, binary classification only
    auc(fpr, tpr)                -- bonus
"""

import numpy as np

from .utils import safe_divide

_VALID_AVERAGES = ("binary", "macro", "micro")


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Count how often every (true class, predicted class) pair occurred.

    Built without looping over samples: both label arrays are converted
    to small integer indices, combined into a single number per sample,
    and counted with `np.bincount`.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
    y_pred : array_like, shape (n_samples,)
    labels : array_like, optional
        The set of labels to index the matrix by. If not given, uses
        every distinct value seen in `y_true` or `y_pred`, sorted.
        If given explicitly, does not need to already be sorted.

    Returns
    -------
    matrix : ndarray of int, shape (n_labels, n_labels)
        matrix[i, j] = number of samples whose true label is labels[i]
        and predicted label is labels[j].
    labels : ndarray, shape (n_labels,)
        The labels indexing the rows/columns of `matrix`, sorted.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes, or either contains
        a value not present in `labels` (when `labels` is given
        explicitly).

    Complexity
    ----------
    O(n + n_labels^2) time and space, where n = len(y_true).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"confusion_matrix(): y_true has shape {y_true.shape} but "
            f"y_pred has shape {y_pred.shape}"
        )

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.unique(np.asarray(labels))  # sorts + de-duplicates

    if not (np.all(np.isin(y_true, labels)) and np.all(np.isin(y_pred, labels))):
        raise ValueError(
            "confusion_matrix(): y_true/y_pred contain a value not present in `labels`"
        )

    n = len(labels)
    t_idx = np.searchsorted(labels, y_true)
    p_idx = np.searchsorted(labels, y_pred)
    matrix = np.bincount(t_idx * n + p_idx, minlength=n * n).reshape(n, n)
    return matrix, labels


def accuracy(y_true, y_pred):
    """
    Fraction of predictions that exactly match the true label.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
    y_pred : array_like, shape (n_samples,)

    Returns
    -------
    float
        A value in [0, 1].

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes, or are empty.

    Complexity
    ----------
    O(n) time and space.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"accuracy(): y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("accuracy(): y_true is empty")

    return float(np.mean(y_true == y_pred))


def _resolve_binary_pos_index(labels, pos_label):
    """Find pos_label's position in `labels`, for average='binary'."""
    if len(labels) != 2:
        raise ValueError(
            f"average='binary' requires exactly 2 classes, got {len(labels)}: "
            f"{list(labels)}. Use average='macro' or 'micro' for more than 2 classes."
        )
    matches = np.where(labels == pos_label)[0]
    if len(matches) == 0:
        raise ValueError(f"pos_label={pos_label!r} not found among labels {list(labels)}")
    return int(matches[0])


def _class_counts(y_true, y_pred, labels):
    """
    Shared helper for precision/recall/f1: per-class true positives and
    the two per-class totals precision/recall are each divided by.
    """
    matrix, all_labels = confusion_matrix(y_true, y_pred, labels)
    true_positives = np.diag(matrix).astype(float)
    predicted_totals = matrix.sum(axis=0).astype(float)  # true_positives + false_positives
    actual_totals = matrix.sum(axis=1).astype(float)      # true_positives + false_negatives
    return true_positives, predicted_totals, actual_totals, all_labels


def _average_ratio(numerator, denominator, average, all_labels, pos_label, fn_name):
    """
    Shared averaging logic for precision() and recall() (metrics that
    are each a single ratio per class, safe to average directly).
    """
    if average not in _VALID_AVERAGES:
        raise ValueError(f"{fn_name}(): average must be one of {_VALID_AVERAGES}, got {average!r}")

    if average == "binary":
        idx = _resolve_binary_pos_index(all_labels, pos_label)
        return float(safe_divide(numerator[idx], denominator[idx], fill=0.0))
    if average == "micro":
        return float(safe_divide(numerator.sum(), denominator.sum(), fill=0.0))
    # 'macro': the plain mean of each class's own ratio
    per_class = safe_divide(numerator, denominator, fill=0.0)
    return float(per_class.mean())


def precision(y_true, y_pred, average="binary", pos_label=1, labels=None):
    """
    What fraction of the samples predicted as a class actually are that
    class: true_positives / (true_positives + false_positives).

    A class the model never predicted at all has a predicted total of 0;
    this returns 0.0 for that class rather than raising or producing NaN
    (a documented design choice, not an accident).

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
    y_pred : array_like, shape (n_samples,)
    average : {'binary', 'macro', 'micro'}, default='binary'
        'binary': precision of the single class named by `pos_label`
        (requires exactly 2 classes total).
        'macro': unweighted mean of each class's own precision.
        'micro': one global ratio, pooling every class's counts first.
    pos_label : default=1
        Which class counts as "positive" when average='binary'.
    labels : array_like, optional
        Passed through to `confusion_matrix`.

    Returns
    -------
    float
        A value in [0, 1].

    Raises
    ------
    ValueError
        If `average` isn't one of the three options above, or
        average='binary' is used with anything other than 2 classes.

    Complexity
    ----------
    O(n + n_labels^2), dominated by building the confusion matrix.
    """
    true_positives, predicted_totals, _, all_labels = _class_counts(y_true, y_pred, labels)
    return _average_ratio(true_positives, predicted_totals, average, all_labels, pos_label, "precision")


def recall(y_true, y_pred, average="binary", pos_label=1, labels=None):
    """
    What fraction of a class's actual samples were correctly identified:
    true_positives / (true_positives + false_negatives).

    Same `average` options, `pos_label`, zero-division convention, and
    complexity as `precision` -- see its docstring for details.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
    y_pred : array_like, shape (n_samples,)
    average : {'binary', 'macro', 'micro'}, default='binary'
    pos_label : default=1
    labels : array_like, optional

    Returns
    -------
    float
        A value in [0, 1].

    Raises
    ------
    ValueError
        If `average` isn't one of the three options, or average='binary'
        is used with anything other than 2 classes.
    """
    true_positives, _, actual_totals, all_labels = _class_counts(y_true, y_pred, labels)
    return _average_ratio(true_positives, actual_totals, average, all_labels, pos_label, "recall")


def f1(y_true, y_pred, average="binary", pos_label=1, labels=None):
    """
    Harmonic mean of precision and recall: 2*P*R / (P+R).

    For average='macro', this computes precision and recall PER CLASS
    first, combines those into a per-class F1, and only then averages
    the F1 scores -- NOT the other way around (computing macro-precision
    and macro-recall first and combining those two numbers). The two
    approaches can give meaningfully different results; averaging the
    per-class F1 scores is the standard definition.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
    y_pred : array_like, shape (n_samples,)
    average : {'binary', 'macro', 'micro'}, default='binary'
    pos_label : default=1
    labels : array_like, optional

    Returns
    -------
    float
        A value in [0, 1]. A class with precision + recall == 0 gets an
        F1 of 0.0, rather than raising or producing NaN.

    Raises
    ------
    ValueError
        If `average` isn't one of the three options, or average='binary'
        is used with anything other than 2 classes.

    Complexity
    ----------
    O(n + n_labels^2), dominated by building the confusion matrix.
    """
    if average not in _VALID_AVERAGES:
        raise ValueError(f"f1(): average must be one of {_VALID_AVERAGES}, got {average!r}")

    true_positives, predicted_totals, actual_totals, all_labels = _class_counts(y_true, y_pred, labels)
    prec_per_class = safe_divide(true_positives, predicted_totals, fill=0.0)
    rec_per_class = safe_divide(true_positives, actual_totals, fill=0.0)
    f1_per_class = safe_divide(2 * prec_per_class * rec_per_class, prec_per_class + rec_per_class, fill=0.0)

    if average == "binary":
        idx = _resolve_binary_pos_index(all_labels, pos_label)
        return float(f1_per_class[idx])
    if average == "micro":
        p_micro = safe_divide(true_positives.sum(), predicted_totals.sum(), fill=0.0)
        r_micro = safe_divide(true_positives.sum(), actual_totals.sum(), fill=0.0)
        return float(safe_divide(2 * p_micro * r_micro, p_micro + r_micro, fill=0.0))
    return float(f1_per_class.mean())  # 'macro'


def mse(y_true, y_pred):
    """
    Mean squared error, for regression.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
    y_pred : array_like, shape (n_samples,)

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes, or are empty.

    Complexity
    ----------
    O(n) time and space.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"mse(): y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("mse(): y_true is empty")

    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true, y_score):
    """
    Trace true-positive rate vs. false-positive rate across every
    possible decision threshold, for binary classification.

    Built without looping over thresholds: sorting once by score and
    taking a running (cumulative) count does the same job.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Must contain only the values 0 and 1, with at least one of each.
    y_score : array_like, shape (n_samples,)
        A higher score means "more confident this is class 1".

    Returns
    -------
    fpr : ndarray
        False-positive rate at each threshold, starting at 0.
    tpr : ndarray
        True-positive rate at each threshold, starting at 0.
    thresholds : ndarray
        The score value at each point (the first entry is +inf,
        representing "classify nothing as positive").

    Raises
    ------
    ValueError
        If y_true/y_score have different shapes, or y_true isn't
        strictly binary (0/1) with both classes present.

    Complexity
    ----------
    O(n log n), dominated by the sort.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"roc_curve(): y_true has shape {y_true.shape} but y_score has shape {y_score.shape}"
        )
    if not np.all(np.isin(y_true, (0, 1))):
        raise ValueError("roc_curve(): y_true must contain only 0 and 1")

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve(): y_true must contain at least one 0 and one 1")

    order = np.argsort(-y_score, kind="stable")  # highest score first
    y_sorted = y_true[order]

    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg

    # Prepend the (0, 0) point: "classify nothing as positive".
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    thresholds = np.concatenate(([np.inf], y_score[order]))

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """
    Area under a curve given by (fpr, tpr) points, via the trapezoidal rule.

    Typically used on the output of `roc_curve` to get a single-number
    summary of classifier quality (0.5 = random guessing, 1.0 = perfect).

    Parameters
    ----------
    fpr : array_like
    tpr : array_like
        Same shape as `fpr`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If fpr and tpr have different shapes.

    Complexity
    ----------
    O(n) time and space.
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError(f"auc(): fpr has shape {fpr.shape} but tpr has shape {tpr.shape}")

    # NumPy 2.x removed np.trapz entirely (AttributeError, not a warning) --
    # np.trapezoid is the replacement.
    return float(np.trapezoid(tpr, fpr))