"""Tests for stream.py: test-then-train logging with StreamTrainer."""

from pathlib import Path

import numpy as np
import pytest

from numcompute.io import DEFAULT_MISSING_VALUES
from numcompute.preprocessing import NotFittedError
from numcompute_stream.io import iter_chunks, load_csv
from numcompute_stream.metrics import Accuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stream import StreamTrainer, model_size_kb
from numcompute_stream.tree import DecisionTreeClassifier


def make_data(n=1000, seed=0):
    X = np.random.default_rng(seed).uniform(0, 10, size=(n, 2))
    return X, (X[:, 0] > 5).astype(int)


class MajorityModel:
    """A model without a buffer: always predicts the most common label seen so far."""

    def __init__(self):
        self.counts = {}

    def partial_fit(self, X, y):
        for label in y:
            self.counts[label] = self.counts.get(label, 0) + 1
        return self

    def predict(self, X):
        return np.full(len(X), max(self.counts, key=self.counts.get))


def test_history_logs_metrics_memory_and_retraining():
    X, y = make_data()
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=400), window=300).run(iter_chunks(X, y, 200))
    assert len(trainer.history) == 5 and trainer.n_samples_seen == 1000
    accuracy = trainer.get("accuracy")
    assert np.isnan(accuracy[:2]).all() and not np.isnan(accuracy[2:]).any()  # chunks 0-1 fill the first buffer
    np.testing.assert_array_equal(trainer.get("retrained"), [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(trainer.get("buffered"), [200, 0, 200, 0, 200])
    assert np.array_equal(np.isnan(trainer.get("retrain_time")), trainer.get("retrained") == 0)
    assert np.all(trainer.get("memory_kb") > 0) and trainer.get("n_retrains")[-1] == 2
    for key in ("accuracy", "cumulative_accuracy", "rolling_accuracy", "f1", "cumulative_f1", "rolling_f1"):
        assert key in trainer.history[-1]


def test_cumulative_accuracy_covers_all_scored_chunks():
    X, y = make_data()
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=250)).run(iter_chunks(X, y, 250))
    accuracy, sizes = trainer.get("accuracy")[1:], trainer.get("n_samples")[1:]
    assert trainer.get("cumulative_accuracy")[-1] == pytest.approx(np.sum(accuracy * sizes) / sizes.sum())


def test_fit_chunk_and_score_chunk_can_be_used_directly():
    X, y = make_data()
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=500), metrics={"accuracy": Accuracy()})
    with pytest.raises(NotFittedError):
        trainer.score_chunk(X[:10], y[:10])  # nothing learned yet
    record = trainer.fit_chunk(X[:500], y[:500])
    assert record["retrained"] == 1 and record["buffered"] == 0 and record["retrain_time"] > 0
    scores = trainer.score_chunk(X[500:], y[500:])
    assert 0.9 < scores["accuracy"] <= 1.0 and len(trainer.last_y_pred) == 500


def test_models_without_a_buffer_are_logged_too():
    X, y = make_data()
    trainer = StreamTrainer(MajorityModel()).run(iter_chunks(X, y, 250))
    assert set(trainer.history[0]) == {"chunk", "n_samples", "train_time", "memory_kb"}  # chunk 0: training only
    assert not np.isnan(trainer.get("accuracy")[1:]).any()


def test_verbose_prints_one_line_per_chunk(capsys):
    X, y = make_data()
    StreamTrainer(DecisionTreeClassifier(buffer_size=500), verbose=True).run(iter_chunks(X, y, 500))
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and "not scored" in lines[0] and "retrained" in lines[0]
    assert "chunk   1" in lines[1] and "accuracy" in lines[1]


def test_invalid_model_and_model_size():
    with pytest.raises(TypeError):
        StreamTrainer(object())
    assert model_size_kb(DecisionTreeClassifier()) > 0


def test_verbose_shows_the_buffer_filling_up(capsys):
    X, y = make_data()
    StreamTrainer(DecisionTreeClassifier(buffer_size=600), verbose=True).run(iter_chunks(X, y, 250))
    lines = capsys.readouterr().out.strip().splitlines()
    assert "buffered 250" in lines[0] and "retrained" in lines[2]  # 250, 500, then 750 >= 600 samples


def test_housing_stream_end_to_end():
    """The pipeline of the demo on the real data: text columns, missing values, a buffered tree."""
    path = Path(__file__).resolve().parents[1] / "Dataset" / "Housing.csv"
    X, y, names = load_csv(path, target_column="airconditioning", missing_values=(*DEFAULT_MISSING_VALUES, "NAN"))
    text_names = ("mainroad", "guestroom", "basement", "hotwaterheating", "prefarea", "furnishingstatus")
    text = [names.index(n) for n in text_names]
    pipe = Pipeline([("encode", OneHotEncoder(columns=text, max_categories=3)), ("impute", Imputer()),
                     ("scale", StandardScaler()), ("model", DecisionTreeClassifier(max_depth=4, buffer_size=100))])
    trainer = StreamTrainer(pipe, window=100).run(iter_chunks(X, y, 25))
    assert len(trainer.history) == 22 and trainer.n_samples_seen == 544
    assert np.flatnonzero(trainer.get("retrained")).tolist() == [3, 7, 11, 15, 19]  # every 100 samples
    assert np.isnan(trainer.get("accuracy")[:4]).all() and not np.isnan(trainer.get("accuracy")[4:]).any()
    assert pipe.classes_.tolist() == ["no", "yes"] and pipe.predict_proba(X[:3]).shape == (3, 2)


def test_verbose_log_of_a_model_without_a_buffer(capsys):
    X, y = make_data()
    StreamTrainer(MajorityModel(), verbose=True).run(iter_chunks(X, y, 500))
    lines = capsys.readouterr().out.strip().splitlines()
    assert "not scored" in lines[0] and "accuracy" in lines[1]
    assert all("retrained" not in line and "buffered" not in line for line in lines)  # nothing to report
