"""Tests for tree.py: the course's tree-building recipe and buffered retraining."""

import numpy as np
import pytest

from numcompute.preprocessing import NotFittedError
from numcompute_stream.tree import DecisionTreeClassifier, best_split, impurity


def rule_data(n=2000, seed=0):
    """Class 1 when x0 > 5 and x1 > 3 (needs two levels of questions); x2 is noise."""
    X = np.random.default_rng(seed).uniform(0, 10, size=(n, 3))
    return X, ((X[:, 0] > 5) & (X[:, 1] > 3)).astype(int)


class CourseTree:
    """The course's DecisionTree for 0/1 features (classification-tree.html, tab 05), used as a reference."""

    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    @staticmethod
    def gini(y):
        if len(y) == 0:
            return 0.0
        p = np.bincount(y, minlength=2) / len(y)
        return 1.0 - (p ** 2).sum()

    def _best_feature(self, X, y):
        best_f, best_gain = None, 0.0
        for f in range(X.shape[1]):
            mask = X[:, f] == 1
            if mask.all() or not mask.any():
                continue
            gain = self.gini(y) - (mask.sum() * self.gini(y[mask]) + (~mask).sum() * self.gini(y[~mask])) / len(y)
            if gain > best_gain:
                best_f, best_gain = f, gain
        return best_f

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return ("leaf", np.bincount(y).argmax())
        f = self._best_feature(X, y)
        if f is None:
            return ("leaf", np.bincount(y).argmax())
        mask = X[:, f] == 1
        return ("node", f, self._build(X[mask], y[mask], depth + 1), self._build(X[~mask], y[~mask], depth + 1))

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def predict(self, X):
        out = []
        for x in X:
            node = self.root
            while node[0] == "node":
                node = node[2] if x[node[1]] == 1 else node[3]
            out.append(node[1])
        return np.array(out)


# --------------------------------------------------------------- building
def test_impurity_matches_the_course_formulas():
    np.testing.assert_allclose(impurity([[6, 6], [12, 0], [0, 0]]), [0.5, 0.0, 0.0])  # Gini
    np.testing.assert_allclose(impurity([[6, 6], [12, 0]], "entropy"), [1.0, 0.0])    # entropy, in bits
    np.testing.assert_allclose(impurity([[1, 1, 1, 1]], "entropy"), [2.0])
    with pytest.raises(ValueError):
        impurity([[1, 1]], "mse")


def test_same_predictions_as_the_course_tree_on_yes_no_features():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(60, 4))
    y = ((X[:, 0] & X[:, 1]) | (rng.random(60) < 0.1)).astype(int)
    every_input = np.array([[(i >> b) & 1 for b in range(4)] for i in range(16)])
    for depth in (1, 2, 3):
        ours = DecisionTreeClassifier(max_depth=depth).fit(X.astype(float), y)
        course = CourseTree(max_depth=depth).fit(X, y)
        np.testing.assert_array_equal(ours.predict(every_input.astype(float)), course.predict(every_input))


def test_best_split_finds_the_obvious_threshold():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [10.0, 5.0], [11.0, 5.0], [12.0, 5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    feature, threshold, gain = best_split(X, y, n_classes=2)
    assert feature == 0 and threshold == 6.5 and gain == pytest.approx(0.5)
    assert best_split(X[:, [1]], y, n_classes=2)[0] is None  # a constant feature cannot split


def test_vectorised_split_search_matches_a_loop_over_thresholds():
    rng = np.random.default_rng(1)
    X = np.round(rng.normal(size=(80, 2)), 1)  # rounding creates repeated values
    y = (X[:, 0] + rng.normal(scale=0.5, size=80) > 0).astype(int)
    parent = impurity(np.bincount(y))
    best = (None, None, 0.0)
    for feature in range(2):
        values = np.unique(X[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2:
            left, right = y[X[:, feature] <= threshold], y[X[:, feature] > threshold]
            child = (len(left) * impurity(np.bincount(left, minlength=2))
                     + len(right) * impurity(np.bincount(right, minlength=2))) / len(y)
            if parent - child > best[2] + 1e-12:
                best = (feature, threshold, parent - child)
    feature, threshold, gain = best_split(X, y, n_classes=2)
    assert feature == best[0] and threshold == pytest.approx(best[1]) and gain == pytest.approx(best[2])


def test_gini_and_entropy_both_learn():
    X, y = rule_data()
    for criterion in ("gini", "entropy"):
        assert np.mean(DecisionTreeClassifier(criterion=criterion).fit(X, y).predict(X) == y) > 0.97


def test_stopping_rules():
    X, y = rule_data()
    assert DecisionTreeClassifier(max_depth=1).fit(X, y).get_depth() == 1                  # deep enough
    assert DecisionTreeClassifier(min_samples_split=5000).fit(X, y).get_n_leaves() == 1   # too few samples
    assert DecisionTreeClassifier().fit(X, np.zeros(len(X), dtype=int)).get_n_leaves() == 1  # everyone agrees
    constant = DecisionTreeClassifier().fit(np.ones((10, 2)), [0, 1] * 5)                  # no question helps
    assert constant.get_n_leaves() == 1 and constant.predict([[1.0, 1.0]])[0] == 0         # a tie -> first class


def test_max_features_is_reproducible():
    X, y = rule_data()
    a = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert DecisionTreeClassifier(max_features=2).fit(X, y).get_n_leaves() > 1


def test_vectorised_prediction_matches_a_per_sample_loop():
    X, y = rule_data()
    tree = DecisionTreeClassifier().fit(X, y)
    expected = []
    for row in X[:300]:  # the course's predict loop, one sample at a time
        node = tree.root_
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        expected.append(tree.classes_[node.label])
    np.testing.assert_array_equal(tree.predict(X[:300]), expected)


# ------------------------------------------------------ buffered retraining
def test_partial_fit_buffers_until_buffer_size_then_retrains_and_clears():
    X, y = rule_data(1000)
    tree = DecisionTreeClassifier(buffer_size=400)
    tree.partial_fit(X[:250], y[:250])
    assert not tree.is_fitted and tree.n_buffered_ == 250 and tree.n_retrains_ == 0
    with pytest.raises(NotFittedError, match="250 of 400"):
        tree.predict(X[:5])
    tree.partial_fit(X[250:500], y[250:500])  # the buffer now holds 500 >= 400
    assert tree.is_fitted and tree.n_buffered_ == 0 and tree.n_retrains_ == 1
    assert tree.last_train_time_ > 0


def test_retraining_uses_only_the_buffered_data():
    X = np.random.default_rng(1).uniform(0, 10, size=(1000, 1))
    first_rule = (X[:500, 0] > 5).astype(int)
    second_rule = 1 - (X[500:, 0] > 5).astype(int)  # the rule flips
    tree = DecisionTreeClassifier(buffer_size=500).partial_fit(X[:500], first_rule)
    assert tree.predict([[9.0]])[0] == 1
    tree.partial_fit(X[500:], second_rule)
    assert tree.predict([[9.0]])[0] == 0  # the new tree only knows the latest buffer


def test_predict_uses_the_last_tree_between_retrains():
    X, y = rule_data(900)
    tree = DecisionTreeClassifier(buffer_size=500).partial_fit(X[:500], y[:500])
    before = tree.predict(X)
    tree.partial_fit(X[500:800], 1 - y[500:800])  # buffered only: no retrain yet
    assert tree.n_retrains_ == 1 and tree.n_buffered_ == 300
    np.testing.assert_array_equal(tree.predict(X), before)


def test_a_chunk_bigger_than_the_buffer_is_used_in_full():
    X, y = rule_data(1500)
    tree = DecisionTreeClassifier(buffer_size=500).partial_fit(X, y)
    assert tree.n_retrains_ == 1 and tree.n_buffered_ == 0
    np.testing.assert_array_equal(tree.predict(X), DecisionTreeClassifier().fit(X, y).predict(X))


def test_fit_trains_from_scratch_and_empties_the_buffer():
    X, y = rule_data()
    tree = DecisionTreeClassifier(buffer_size=10_000).partial_fit(X[:100], y[:100])
    tree.fit(X, y)
    assert tree.is_fitted and tree.n_buffered_ == 0 and np.mean(tree.predict(X) == y) > 0.97


def test_new_classes_can_appear_between_retrains():
    tree = DecisionTreeClassifier(buffer_size=2).partial_fit([[0.0], [1.0]], ["a", "b"])
    assert tree.predict_proba([[0.0]]).shape == (1, 2)
    tree.partial_fit([[5.0]], ["c"])  # buffered: the current tree has never seen "c"
    proba = tree.predict_proba([[0.0]])
    assert tree.classes_.tolist() == ["a", "b", "c"] and proba.shape == (1, 3) and proba[0, 2] == 0.0


def test_probabilities_sum_to_one_and_ties_are_deterministic():
    tree = DecisionTreeClassifier(buffer_size=2).partial_fit([[0.0], [0.0]], ["x", "y"])
    np.testing.assert_allclose(tree.predict_proba([[0.0], [3.0]]), [[0.5, 0.5], [0.5, 0.5]])
    assert tree.predict([[0.0]])[0] == "x"  # a tie goes to the first class


def test_empty_chunks_and_bad_input():
    tree = DecisionTreeClassifier(buffer_size=10)
    tree.partial_fit(np.empty((0, 2)), np.empty(0))  # an empty chunk changes nothing
    assert not tree.is_fitted and tree.n_buffered_ == 0
    with pytest.raises(ValueError, match="Imputer"):
        tree.partial_fit([[np.nan, 1.0]], [0])
    with pytest.raises(ValueError):
        tree.partial_fit([[1.0, 2.0, 3.0]], [0])  # 3 features after a chunk with 2
    with pytest.raises(ValueError):
        tree.partial_fit([[1.0, 2.0]], [0, 1])  # X and y lengths differ
    with pytest.raises(ValueError):
        tree.partial_fit([[1.0, 2.0]], [np.nan])  # a missing label
    with pytest.raises(TypeError):
        tree.partial_fit([["a", "b"]], [0])  # text features must be encoded first
    tree.partial_fit([[1.0, 2.0]], [0])
    with pytest.raises(ValueError):
        tree.partial_fit([[1.0, 2.0]], ["zero"])  # labels switch from numbers to text
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(np.empty((0, 2)), np.empty(0))


@pytest.mark.parametrize("params", [{"criterion": "mse"}, {"max_depth": 0}, {"min_samples_split": 1},
                                    {"buffer_size": 0}, {"buffer_size": 2.5}, {"max_features": "half"}])
def test_invalid_settings(params):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**params)


def test_classes_can_be_registered_in_advance():
    tree = DecisionTreeClassifier(buffer_size=1).partial_fit([[0.0], [1.0]], [0, 1], classes=[0, 1, 2])
    assert tree.classes_.tolist() == [0, 1, 2] and tree.predict_proba([[0.0]]).shape == (1, 3)


def test_neighbouring_floats_without_a_midpoint_still_split_correctly():
    low = np.nextafter(1.0, 2.0)   # the last bit of low is odd, so the halfway point
    high = np.nextafter(low, 2.0)  # rounds up onto high: there is no float in between
    X, y = np.array([[low], [high]]), np.array([0, 1])
    feature, threshold, _ = best_split(X, y, n_classes=2)
    assert feature == 0 and threshold == low  # x <= threshold still separates the two values
    assert DecisionTreeClassifier().fit(X, y).predict(X).tolist() == [0, 1]
