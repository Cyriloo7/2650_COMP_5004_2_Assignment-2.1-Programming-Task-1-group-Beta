"""Decision tree classifier for data streams, retrained from a buffer.

How the tree is built (``fit``)
-------------------------------
The recipe of the course's classification tree (``classification-tree.html``):
try every question, keep the one with the biggest information gain, split,
repeat on each side, and stop when a rule says so. Because the features here
are numbers rather than yes/no answers, a question is
``x[feature] <= threshold``; the rows that answer "yes" go left.

* impurity : Gini ``1 - sum(p^2)`` or entropy ``-sum(p * log2(p))``
* gain     : impurity before the split minus the size-weighted impurity after it
* stop     : at ``max_depth``, when a node has fewer than ``min_samples_split``
             samples, when all its samples have the same class, or when no
             question has a gain above 0
* leaf     : predicts the majority class (a tie goes to the first class) and
             keeps the class proportions for ``predict_proba``

How it learns from a stream (``partial_fit``)
---------------------------------------------
Each chunk is added to an internal buffer. Once the buffer holds at least
``buffer_size`` samples, a new tree is trained from scratch on the buffered
data and the buffer is cleared. Between retrains, ``predict`` uses the most
recently trained tree. A chunk is checked completely before anything changes,
so a rejected chunk leaves the tree, its buffer and ``classes_`` untouched.

Vectorisation
-------------
* The gain of every possible threshold of a feature is computed at once from
  cumulative class counts, instead of one threshold at a time.
* Prediction sends all samples down the tree together: one NumPy comparison
  per node instead of a Python loop per sample.

Interpretation
--------------
``feature_importances_`` (total information gain per feature) and
``export_text()`` (the learned questions as indented rules) show what the
latest tree has learned.
"""

import time

import numpy as np

from numcompute.metrics import _has_nan                                           # Assignment 2.1 (reused)
from numcompute.preprocessing import NotFittedError, _check_numeric, _kind_group  # Assignment 2.1 (reused)
from numcompute.utils import safe_divide                                          # Assignment 2.1 (reused)

from ._checks import check_int_at_least


def impurity(class_counts, criterion="gini"):
    """Gini impurity or entropy of each row of class counts.

    The same formulas as ``gini(y)`` and ``entropy(y)`` in the course code,
    applied to counts so that many groups can be scored at once. An empty
    group has impurity 0.

    Parameters
    ----------
    class_counts : array-like of shape (..., n_classes)
        Number of samples of each class in each group.
    criterion : {"gini", "entropy"}, default "gini"

    Returns
    -------
    ndarray of shape (...)
        One impurity per group (entropy is measured in bits).

    Raises
    ------
    ValueError
        If ``criterion`` is not "gini" or "entropy".
    """
    counts = np.asarray(class_counts, dtype=float)
    totals = counts.sum(axis=-1, keepdims=True)
    p = safe_divide(counts, totals)  # an empty group gives p = 0
    if criterion == "gini":
        values = 1.0 - np.sum(p ** 2, axis=-1)
    elif criterion == "entropy":
        log_p = np.log2(p, out=np.zeros_like(p), where=p > 0)  # log2(0) is undefined: 0 * log(0) counts as 0
        values = -np.sum(p * log_p, axis=-1)
    else:
        raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}.")
    return np.where(totals[..., 0] > 0, values, 0.0)


def best_split(X, y, n_classes, criterion="gini", features=None):
    """Find the question ``X[:, feature] <= threshold`` with the biggest information gain.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Numeric features without NaN.
    y : ndarray of shape (n_samples,)
        Class indices 0 .. n_classes - 1.
    n_classes : int
        Number of classes (the length of the class-count vectors).
    criterion : {"gini", "entropy"}, default "gini"
    features : sequence of int or None, default None
        Features to try (None = all of them).

    Returns
    -------
    feature : int or None
        Best feature, or None when no question has a gain above 0.
    threshold : float or None
        Lies halfway between two neighbouring values of the feature.
    gain : float
        Impurity of the node minus the size-weighted impurity of its children.

    Notes
    -----
    Ties are broken deterministically: the lowest threshold of the lowest
    feature wins. Time O(F * n log n) for F features tried: the sort
    dominates, and the gain of all n - 1 thresholds of a feature comes from
    one cumulative sum. Extra memory O(n * n_classes). A constant feature is
    skipped right after its sort, so all-equal data costs little. Thresholds
    are computed as ``low / 2 + high / 2``, which cannot overflow
    (``high - low`` can) and keeps values such as 1e9 and 1e9 + 1 apart even
    in widely spread data.
    """
    n = len(y)
    parent = impurity(np.bincount(y, minlength=n_classes), criterion)
    best_feature, best_threshold, best_gain = None, None, 0.0
    for feature in range(X.shape[1]) if features is None else features:
        order = np.argsort(X[:, feature], kind="stable")
        values = X[order, feature]
        if values[0] == values[-1]:
            continue                                      # constant feature: nothing to ask, skip the counting below
        one_hot = np.eye(n_classes)[y[order]]             # (n, n_classes)
        left = np.cumsum(one_hot, axis=0)[:-1]            # class counts left of each cut
        right = one_hot.sum(axis=0) - left                # class counts right of each cut
        n_left = np.arange(1, n)
        gain = parent - (n_left * impurity(left, criterion) + (n - n_left) * impurity(right, criterion)) / n
        valid = values[1:] > values[:-1]                  # a cut must fall between two different values
        gain = np.where(valid, gain, -np.inf)             # (at least one cut is valid: the feature is not constant)
        i = int(np.argmax(gain))                          # first maximum = lowest threshold
        if gain[i] > best_gain:
            threshold = values[i] / 2 + values[i + 1] / 2  # halving first: the sum cannot overflow
            if not values[i] <= threshold < values[i + 1]:  # neighbouring floats have no midpoint between them
                threshold = values[i]
            best_feature, best_threshold, best_gain = feature, threshold, gain[i]
    return best_feature, best_threshold, best_gain


def _merge_labels(known, labels):
    """Return the sorted union of ``known`` (an array or None) and ``labels``, without changing either.

    Parameters
    ----------
    known : ndarray of shape (n_known,) or None
    labels : array-like of shape (n_labels,)

    Returns
    -------
    ndarray of shape (n_union,) or None
        None only when both inputs hold no label.

    Raises
    ------
    ValueError
        If text and number labels are mixed (NumPy would silently turn the numbers into text).
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return known
    try:
        labels = np.unique(labels)
    except TypeError:  # e.g. an object array holding both "a" and 1
        raise ValueError("Labels must all be text or all be numbers; got a mix of both.") from None
    if known is None:
        return labels
    if {_kind_group(labels.dtype), _kind_group(known.dtype)} == {"text", "number"}:
        raise ValueError(f"Labels must all be text or all be numbers; got {labels.dtype} after {known.dtype}.")
    return np.union1d(known, labels)


class Node:
    """A question with two children, or a leaf with an answer (as in the course code).

    Attributes
    ----------
    feature : int or None
        Feature of the question ``x[feature] <= threshold`` (decision nodes only).
    threshold : float or None
        Threshold of the question (decision nodes only).
    left, right : Node or None
        The subtrees for the "yes" and "no" answers (decision nodes only).
    label : int or None
        Index of the predicted class in the tree's classes (leaves only).
    proba : ndarray of shape (n_classes,) or None
        Class proportions at the leaf (leaves only).
    n_samples : int
        Number of training samples that reached the node.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None, proba=None, n_samples=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
        self.proba = proba
        self.n_samples = n_samples

    @property
    def is_leaf(self):
        """True for a leaf (a node that answers instead of asking a question)."""
        return self.label is not None


class DecisionTreeClassifier:
    """Depth-limited decision tree (Gini or entropy) with buffered retraining for streams.

    Parameters
    ----------
    max_depth : int, default 5
        Maximum depth of the tree (the root has depth 0).
    min_samples_split : int, default 2
        A node with fewer samples becomes a leaf.
    max_features : int, "sqrt" or None, default None
        Number of randomly chosen features tried at each node (None = all).
    criterion : {"gini", "entropy"}, default "gini"
    buffer_size : int, default 500
        ``partial_fit`` retrains the tree once the buffer holds at least
        this many samples.
    random_state : int or None, default None
        Seed for choosing features when ``max_features`` is set.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or None
        All labels seen so far, sorted. Columns of ``predict_proba`` follow it.
    n_features_in_ : int or None
        Number of features, fixed by the first non-empty chunk (or by ``fit``).
    root_ : Node or None
        The current tree (None until the first training).
    feature_importances_ : ndarray of shape (n_features,)
        Share of the current tree's total information gain won by each
        feature (sums to 1; all zeros for a tree that is a single leaf).
    n_retrains_ : int
        Number of times a tree has been trained.
    last_train_time_ : float or None
        Seconds taken by the latest training.

    Notes
    -----
    Training on n buffered samples costs O(max_depth * F * n log n): every
    level of the tree touches each sample once. Prediction sends all samples
    down together, O(max_depth) NumPy comparisons per sample. Memory stays
    bounded: the buffer never holds more than ``buffer_size + chunk - 1``
    samples (it is cleared at every retrain), plus the nodes of one tree.

    Example
    -------
    >>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
    >>> for X_chunk, y_chunk in stream:
    ...     tree.partial_fit(X_chunk, y_chunk)
    >>> tree.predict(X_new)
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, criterion="gini",
                 buffer_size=500, random_state=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}.")
        check_int_at_least("max_depth", max_depth, 1)
        check_int_at_least("min_samples_split", min_samples_split, 2)
        check_int_at_least("buffer_size", buffer_size, 1)
        if not (max_features is None or max_features == "sqrt"):
            try:
                check_int_at_least("max_features", max_features, 1)
            except ValueError:
                raise ValueError(
                    f"max_features must be None, 'sqrt' or an integer >= 1, got {max_features!r}.") from None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.buffer_size = buffer_size
        self.random_state = random_state
        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.n_retrains_ = 0
        self.last_train_time_ = None
        self._X_buffer, self._y_buffer = [], []  # chunks waiting for the next retrain
        self._rng = np.random.default_rng(random_state)

    # --------------------------------------------------------------- training
    def fit(self, X, y):
        """Train a tree from scratch on (X, y), replacing the current tree.

        The buffer is emptied, and ``classes_`` and ``n_features_in_`` restart
        from this data. If the data is rejected, nothing changes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric features without NaN or infinity (n_samples >= 1).
        y : array-like of shape (n_samples,)
            Class labels: all numbers or all text, no NaN.

        Returns
        -------
        self
        """
        X, y = self._check_training_data(X, y, reset=True)
        if len(y) == 0:
            raise ValueError("fit needs at least one sample.")
        classes = _merge_labels(None, y)  # checked before anything changes
        self.n_features_in_, self.classes_ = X.shape[1], classes
        self._X_buffer, self._y_buffer = [], []
        self._train(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """Add one chunk to the buffer; retrain once the buffer holds ``buffer_size`` samples.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
            Numeric features without NaN or infinity. The first non-empty
            chunk fixes ``n_features``; later chunks (even empty ones) must
            match it. An empty chunk changes nothing.
        y_chunk : array-like of shape (n_samples,)
            Class labels: all numbers or all text, no NaN.
        classes : array-like of shape (n_classes,), optional
            Labels to know about in advance (labels that appear later are
            added automatically).

        Returns
        -------
        self

        Notes
        -----
        The chunk is checked completely before anything changes, so a
        rejected chunk leaves the tree, the buffer and ``classes_`` untouched.
        """
        X, y = self._check_training_data(X_chunk, y_chunk)
        known = _merge_labels(self.classes_, y)
        if classes is not None:
            known = _merge_labels(known, classes)
        # Every check has passed: only now is the state updated.
        if self.n_features_in_ is None and len(y) > 0:
            self.n_features_in_ = X.shape[1]  # the first non-empty chunk fixes the number of features
        self.classes_ = known
        if len(y) > 0:
            self._X_buffer.append(X)
            self._y_buffer.append(y)
        if self.n_buffered_ >= self.buffer_size:
            self._train(np.vstack(self._X_buffer), np.concatenate(self._y_buffer))
            self._X_buffer, self._y_buffer = [], []  # the buffer is cleared after every retrain
        return self

    @property
    def n_buffered_(self):
        """Number of samples waiting in the buffer for the next retrain (int)."""
        return sum(len(y) for y in self._y_buffer)

    @property
    def is_fitted(self):
        """True once a tree has been trained, i.e. ``predict`` can be used."""
        return self.root_ is not None

    def _train(self, X, y):
        """Train a new tree on (X, y) and replace the current one."""
        start = time.perf_counter()
        self._tree_classes = self.classes_.copy()  # the class order used by this tree's leaves
        gains = np.zeros(X.shape[1])               # information gain won by each feature, weighted by node size
        self.root_ = self._build(X, np.searchsorted(self._tree_classes, y), 0, gains)
        total = gains.sum()
        self.feature_importances_ = gains / total if total > 0 else gains
        self.last_train_time_ = time.perf_counter() - start
        self.n_retrains_ += 1

    def _build(self, X, y, depth, gains):
        """Grow the tree on (X, y): stop and make a leaf, or split on the best question and recurse."""
        counts = np.bincount(y, minlength=len(self._tree_classes))
        if (depth >= self.max_depth                       # deep enough
                or len(y) < self.min_samples_split        # too few samples to split
                or np.count_nonzero(counts) == 1):        # everyone agrees
            return self._leaf(counts)
        feature, threshold, gain = best_split(X, y, len(self._tree_classes), self.criterion, self._choose_features())
        if feature is None:                               # no question helps
            return self._leaf(counts)
        gains[feature] += len(y) * gain                   # impurity decrease of this split, for feature_importances_
        goes_left = X[:, feature] <= threshold
        return Node(feature=feature, threshold=threshold, n_samples=len(y),
                    left=self._build(X[goes_left], y[goes_left], depth + 1, gains),
                    right=self._build(X[~goes_left], y[~goes_left], depth + 1, gains))

    @staticmethod
    def _leaf(counts):
        """A leaf answering the majority class (a tie goes to the first class) with the class proportions."""
        return Node(label=int(np.argmax(counts)), proba=counts / counts.sum(), n_samples=int(counts.sum()))

    def _choose_features(self):
        """Features to try at a node: all of them, or a random subset (``max_features``)."""
        n = self.n_features_in_
        if self.max_features is None:
            return None
        k = max(1, int(np.sqrt(n))) if self.max_features == "sqrt" else min(int(self.max_features), n)
        return np.sort(self._rng.choice(n, size=k, replace=False))

    # ------------------------------------------------------------- prediction
    def predict_proba(self, X):
        """Class probabilities of each sample, from the leaf it reaches in the latest tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric features without NaN or infinity.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Rows sum to 1; columns follow ``classes_``. A class first seen
            after the latest retrain has probability 0 until the next one.

        Raises
        ------
        NotFittedError
            Before the first training (the message says how full the buffer is).
        """
        self._check_trained()
        X = self._check_input(X)
        proba = np.zeros((len(X), len(self.classes_)))
        columns = np.searchsorted(self.classes_, self._tree_classes)  # classes may have grown since training
        for leaf, rows in self._route(X):
            proba[np.ix_(rows, columns)] = leaf.proba
        return proba

    def predict(self, X):
        """Most likely class of each sample (a tie goes to the first class in ``classes_``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric features without NaN or infinity.

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels taken from ``classes_``.

        Raises
        ------
        NotFittedError
            Before the first training.
        """
        proba = self.predict_proba(X)  # checks first that a tree exists
        return self.classes_[np.argmax(proba, axis=1)]

    def _route(self, X):
        """Send all rows of X down the tree together; return (leaf, row_indices) pairs."""
        groups = []
        stack = [(self.root_, np.arange(len(X)))]
        while stack:
            node, rows = stack.pop()
            if len(rows) == 0:
                continue
            if node.is_leaf:
                groups.append((node, rows))
            else:
                goes_left = X[rows, node.feature] <= node.threshold
                stack.append((node.left, rows[goes_left]))
                stack.append((node.right, rows[~goes_left]))
        return groups

    # ----------------------------------------------------------- inspection
    def get_depth(self):
        """Depth of the current tree (int; 0 for a single leaf or before training)."""
        def depth(node):
            return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))
        return depth(self.root_) if self.root_ is not None else 0

    def get_n_leaves(self):
        """Number of leaves of the current tree (int; 0 before training)."""
        def leaves(node):
            return 1 if node.is_leaf else leaves(node.left) + leaves(node.right)
        return leaves(self.root_) if self.root_ is not None else 0

    def export_text(self, feature_names=None, decimals=3):
        """The current tree as indented text rules, one line per question or answer.

        Parameters
        ----------
        feature_names : sequence of str, optional
            One name per feature (default ``"x0", "x1", ...``).
        decimals : int, default 3
            Significant digits shown for thresholds.

        Returns
        -------
        str
            For example::

                |--- area <= 0.52
                |   |--- class: no  (n=61, 80% no)
                |--- area >  0.52
                |   |--- class: yes  (n=39, 72% yes)

        Raises
        ------
        NotFittedError
            Before the first training.
        ValueError
            If ``feature_names`` does not have one name per feature.
        """
        self._check_trained()
        check_int_at_least("decimals", decimals, 1)
        names = [f"x{j}" for j in range(self.n_features_in_)] if feature_names is None else list(feature_names)
        if len(names) != self.n_features_in_:
            raise ValueError(f"feature_names has {len(names)} names, but the tree uses "
                             f"{self.n_features_in_} features.")
        lines = []

        def write(node, indent):  # a walk over the nodes (at most 2 ** (max_depth + 1) - 1), never over samples
            if node.is_leaf:
                label, share = self._tree_classes[node.label], node.proba[node.label]
                lines.append(f"{indent}|--- class: {label}  (n={node.n_samples}, {share:.0%} {label})")
                return
            name, threshold = names[node.feature], f"{node.threshold:.{decimals}g}"
            lines.append(f"{indent}|--- {name} <= {threshold}")
            write(node.left, indent + "|   ")
            lines.append(f"{indent}|--- {name} >  {threshold}")
            write(node.right, indent + "|   ")

        write(self.root_, "")
        return "\n".join(lines)

    # ----------------------------------------------------------------- checks
    def _check_trained(self):
        """Raise NotFittedError (saying how full the buffer is) if no tree has been trained yet."""
        if self.root_ is None:
            raise NotFittedError(
                f"This DecisionTreeClassifier has not been trained yet: the buffer holds {self.n_buffered_} of "
                f"{self.buffer_size} samples. Call fit, or partial_fit until the buffer is full.")

    def _check_input(self, X, reset=False):
        """Check X without changing the tree: 2-D, numeric, no NaN or infinity, and the expected width."""
        X = _check_numeric(X, "DecisionTreeClassifier")  # 2-D, numeric, no infinity (2.1 checks)
        if np.isnan(X).any():
            raise ValueError("DecisionTreeClassifier: X contains NaN values; fill them first, "
                             "e.g. with an Imputer step in a Pipeline.")
        if not reset and self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but DecisionTreeClassifier is using "
                             f"{self.n_features_in_} features.")
        return X

    def _check_training_data(self, X, y, reset=False):
        """Check (X, y) without changing the tree; ``reset`` skips the width check (for ``fit``)."""
        X = self._check_input(X, reset=reset)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"y must be 1-D with one label per row of X; got X {X.shape} and y {y.shape}.")
        if _has_nan(y):
            raise ValueError("y contains NaN labels; drop those rows first.")
        return X, y
