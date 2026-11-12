"""Tests for pipeline.py: the 2.1 Pipeline extended with partial_fit."""

import numpy as np
import pytest

from numcompute import preprocessing as base21
from numcompute.pipeline import Pipeline as Pipeline21
from numcompute.preprocessing import NotFittedError
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def make_data(n=1200, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, 3))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


def test_specification_example():
    X, y = make_data()
    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500))])
    for rows in np.array_split(np.arange(1200), 4):  # chunks of 300
        pipe.partial_fit(X[rows], y[rows])
    assert pipe.is_fitted and pipe["model"].n_retrains_ == 2 and pipe["model"].n_buffered_ == 0
    assert np.mean(pipe.predict(X) == y) > 0.85
    assert pipe.predict_proba(X[:2]).shape == (2, 2) and pipe.classes_.tolist() == [0, 1]


def test_partial_fit_equals_running_the_steps_by_hand():
    X, y = make_data()
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=300))])
    scaler, tree = StandardScaler(), DecisionTreeClassifier(buffer_size=300)
    for rows in np.array_split(np.arange(1200), 6):
        pipe.partial_fit(X[rows], y[rows])
        scaler.partial_fit(X[rows])
        tree.partial_fit(scaler.transform(X[rows]), y[rows])
    np.testing.assert_array_equal(pipe.predict(X), tree.predict(scaler.transform(X)))


def test_not_fitted_until_the_tree_has_trained_once():
    X, y = make_data()
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=500))])
    pipe.partial_fit(X[:200], y[:200])
    assert not pipe.is_fitted
    with pytest.raises(NotFittedError):
        pipe.predict(X[:5])
    pipe.partial_fit(X[200:600], y[200:600])
    assert pipe.is_fitted


def test_mixed_data_with_text_and_missing_values():
    X = np.array([[1.0, "a"], [np.nan, "b"], [3.0, "a"], [4.0, None]] * 25, dtype=object)
    y = np.array([0, 1, 0, 1] * 25)
    pipe = Pipeline([("encode", OneHotEncoder(columns=[1], max_categories=3)), ("impute", Imputer()),
                     ("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=40))])
    pipe.partial_fit(X[:40], y[:40]).partial_fit(X[40:], y[40:])
    features = pipe[:-1].transform(X)  # the transformers only (2.1 slicing)
    assert features.shape == (100, 4) and not np.isnan(features).any()
    assert np.mean(pipe.predict(X) == y) == 1.0


def test_reuses_the_2_1_pipeline():
    X, y = make_data()
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier())]).fit(X, y)
    assert isinstance(pipe, Pipeline21) and len(pipe) == 2
    assert isinstance(pipe["tree"], DecisionTreeClassifier) and pipe.named_steps["scale"] is pipe["scale"]
    assert np.mean(pipe.predict(X) == y) > 0.9


def test_invalid_steps_are_rejected():
    with pytest.raises(ValueError):
        Pipeline([])
    with pytest.raises(TypeError):
        Pipeline([("tree", DecisionTreeClassifier()), ("scale", StandardScaler())])  # a tree cannot transform
    with pytest.raises(TypeError):
        Pipeline(["not a tuple"])
    with pytest.raises(ValueError):
        Pipeline([("a", StandardScaler()), ("a", DecisionTreeClassifier())])
    with pytest.raises(TypeError, match="partial_fit"):
        Pipeline([("scale", base21.StandardScaler()), ("tree", DecisionTreeClassifier())])  # 2.1 scaler: batch only
