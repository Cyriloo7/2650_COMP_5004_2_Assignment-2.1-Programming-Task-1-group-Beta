"""Classification and regression metrics.

Contents:
    confusion_matrix: counts of (true class, predicted class) pairs.
    accuracy, precision, recall, f1: classification scores.
    mse: mean squared error.
    roc_curve, auc: ROC curve and area under a curve, for binary classification.
"""

import numpy as np

from .utils import safe_divide

_VALID_AVERAGES = ("binary", "macro", "micro")


def _has_nan(values):
    """Return True if a float array contains NaN (other dtypes cannot)."""
    return bool(np.issubdtype(values.dtype, np.floating) and np.isnan(values).any())


def confusion_matrix(y_true, y_pred, labels=None):
    """Count how often each (true class, predicted class) pair occurs.

    Built without a Python loop over samples: labels are mapped to integer
    indices, combined into one index per sample, and counted with `np.bincount`.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
    y_pred : array_like, shape (n_samples,)
    labels : array_like, optional
        Labels that index the matrix, in any order. Defaults to the sorted
        distinct values found in `y_true` and `y_pred`.

    Returns
    -------
    matrix : ndarray of int, shape (n_labels, n_labels)
        matrix[i, j] counts samples with true label labels[i] and predicted
        label labels[j].
    labels : ndarray, shape (n_labels,)
        The sorted labels indexing the rows and columns.

    Raises
    ------
    ValueError
        If `y_true` or `y_pred` is not 1-D, they differ in shape, either
        contains NaN, or either contains a value not in `labels`.

    Complexity
    ----------
    O(n + n_labels^2) time and space, where n = len(y_true).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"confusion_matrix(): y_true and y_pred must be 1-D, got shapes "
            f"{y_true.shape} and {y_pred.shape}"
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"confusion_matrix(): y_true has shape {y_true.shape} but "
            f"y_pred has shape {y_pred.shape}"
        )
    if _has_nan(y_true) or _has_nan(y_pred):
        raise ValueError("confusion_matrix(): y_true and y_pred must not contain NaN")

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
    """Fraction of predictions that exactly match the true label.

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
        If `y_true` and `y_pred` differ in shape, or are empty.

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
    if len(labels) > 2:
        raise ValueError(
            f"average='binary' requires at most 2 classes, got {len(labels)}: "
            f"{list(labels)}. Use average='macro' or 'micro' for more than 2 classes."
        )
    matches = np.where(labels == pos_label)[0]
    if len(matches) == 0:
        raise ValueError(f"pos_label={pos_label!r} not found among labels {list(labels)}")
    return int(matches[0])


def _class_counts(y_true, y_pred, labels):
    """Per-class true positives and the totals precision and recall divide by."""
    matrix, all_labels = confusion_matrix(y_true, y_pred, labels)
    true_positives = np.diag(matrix).astype(float)
    predicted_totals = matrix.sum(axis=0).astype(float)  # true_positives + false_positives
    actual_totals = matrix.sum(axis=1).astype(float)      # true_positives + false_negatives
    return true_positives, predicted_totals, actual_totals, all_labels


def _average_ratio(numerator, denominator, average, all_labels, pos_label, fn_name):
    """Average a per-class ratio (precision or recall) as requested."""
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
    """Fraction of a class's predictions that are correct: TP / (TP + FP).

    A class that is never predicted has precision 0.0 rather than NaN.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
    y_pred : array_like, shape (n_samples,)
    average : {'binary', 'macro', 'micro'}, default='binary'
        'binary': the class given by `pos_label` (at most 2 classes may be
        present, and `pos_label` must be one of them). 'macro': unweighted
        mean over classes. 'micro': one ratio from the pooled counts.
    pos_label : default=1
        Positive class for average='binary'.
    labels : array_like, optional
        Passed to `confusion_matrix`.

    Returns
    -------
    float
        A value in [0, 1].

    Raises
    ------
    ValueError
        If `average` is invalid, or average='binary' is used with more than
        2 classes or with a `pos_label` that is not among them.

    Complexity
    ----------
    O(n + n_labels^2), dominated by building the confusion matrix.
    """
    true_positives, predicted_totals, _, all_labels = _class_counts(y_true, y_pred, labels)
    return _average_ratio(true_positives, predicted_totals, average, all_labels, pos_label, "precision")


def recall(y_true, y_pred, average="binary", pos_label=1, labels=None):
    """Fraction of a class's actual samples that are found: TP / (TP + FN).

    Parameters and conventions are the same as for `precision`; a class with
    no actual samples has recall 0.0 rather than NaN.

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
        If `average` is invalid, or average='binary' is used with more than
        2 classes or with a `pos_label` that is not among them.

    Complexity
    ----------
    O(n + n_labels^2), dominated by building the confusion matrix.
    """
    true_positives, _, actual_totals, all_labels = _class_counts(y_true, y_pred, labels)
    return _average_ratio(true_positives, actual_totals, average, all_labels, pos_label, "recall")


def f1(y_true, y_pred, average="binary", pos_label=1, labels=None):
    """Harmonic mean of precision and recall: 2PR / (P + R).

    For average='macro' the per-class F1 scores are averaged. This differs
    from the F1 of macro precision and macro recall, which is not the
    standard definition.

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
        A value in [0, 1]; 0.0 where precision + recall is 0.

    Raises
    ------
    ValueError
        If `average` is invalid, or average='binary' is used with more than
        2 classes or with a `pos_label` that is not among them.

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
    """Mean squared error, for regression.

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
        If `y_true` and `y_pred` differ in shape, or are empty.

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
    """ROC curve for binary classification, with one point per distinct score.

    Tied scores are merged into a single point, so the curve (and its area)
    does not depend on the order of the samples. Built from one sort and
    cumulative sums rather than a loop over thresholds.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Labels; only 0 and 1 are allowed, with at least one of each.
    y_score : array_like, shape (n_samples,)
        Scores; a higher value means "more likely to be class 1".

    Returns
    -------
    fpr, tpr : ndarray, shape (n_thresholds + 1,)
        False- and true-positive rates, starting at (0, 0).
    thresholds : ndarray, shape (n_thresholds + 1,)
        Score at each point; the first entry is +inf ("predict nothing as
        positive"). n_thresholds is the number of distinct scores.

    Raises
    ------
    ValueError
        If `y_true` and `y_score` differ in shape, or `y_true` is not 0/1
        with both classes present.

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
    scores_sorted = y_score[order]
    y_sorted = y_true[order]

    # Keep the last sample of each run of equal scores: one point per threshold.
    run_ends = np.append(
        np.flatnonzero(scores_sorted[1:] != scores_sorted[:-1]), len(y_sorted) - 1
    )
    tpr = np.cumsum(y_sorted)[run_ends] / n_pos
    fpr = np.cumsum(1 - y_sorted)[run_ends] / n_neg

    # Prepend the (0, 0) point: "classify nothing as positive".
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    thresholds = np.concatenate(([np.inf], scores_sorted[run_ends]))

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """Area under the curve through the points (fpr, tpr), by the trapezoidal rule.

    Typically applied to the output of `roc_curve` (0.5 is chance, 1.0 is
    perfect). `fpr` may be increasing or decreasing; the area is positive
    either way.

    Parameters
    ----------
    fpr : array_like, shape (n,)
        Monotonic x-coordinates.
    tpr : array_like, shape (n,)
        y-coordinates.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If `fpr` and `tpr` differ in shape, have fewer than 2 points, or
        `fpr` is not monotonic.

    Complexity
    ----------
    O(n) time and space.
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError(f"auc(): fpr has shape {fpr.shape} but tpr has shape {tpr.shape}")
    if fpr.size < 2:
        raise ValueError(f"auc(): at least 2 points are needed, got {fpr.size}")

    step = np.diff(fpr)
    if np.all(step >= 0):
        direction = 1.0
    elif np.all(step <= 0):
        direction = -1.0
    else:
        raise ValueError("auc(): fpr must be monotonic (all non-decreasing or all non-increasing)")

    # np.trapz was removed in NumPy 2.x; np.trapezoid replaces it.
    return direction * float(np.trapezoid(tpr, fpr))