"""Tests for metrics.py: streaming results must equal the 2.1 batch results on all the data."""

import numpy as np
import pytest

from numcompute import metrics as metrics21
from numcompute_stream.metrics import AUC, F1, Accuracy, ConfusionMatrix, Precision, Recall, RollingMetric

Y_TRUE = np.array(["a", "a", "b", "b", "c", "c", "a", "b"])
Y_PRED = np.array(["a", "b", "b", "b", "c", "a", "a", "c"])


def stream(metric, y_true, y_pred, n_chunks=3):
    """Feed the data to the metric in chunks and return the result."""
    for rows in np.array_split(np.arange(len(y_true)), n_chunks):
        metric.update(y_true[rows], y_pred[rows])
    return metric.result()


def noisy_labels(n=500, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, n)
    return y_true, np.where(rng.random(n) < 0.6, y_true, rng.integers(0, n_classes, n))


def test_accuracy_streaming_equals_the_2_1_function():
    assert stream(Accuracy(), Y_TRUE, Y_PRED) == pytest.approx(metrics21.accuracy(Y_TRUE, Y_PRED))


def test_confusion_matrix_accumulates_like_the_2_1_function():
    matrix, labels = stream(ConfusionMatrix(), Y_TRUE, Y_PRED)
    expected_matrix, expected_labels = metrics21.confusion_matrix(Y_TRUE, Y_PRED)
    np.testing.assert_array_equal(matrix, expected_matrix)
    assert labels.tolist() == expected_labels.tolist() == ["a", "b", "c"]


def test_confusion_matrix_grows_when_new_labels_appear():
    matrix, labels = ConfusionMatrix().update([1, 1], [1, 2]).update([0], [0]).result()
    assert labels.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(matrix, [[1, 0, 0], [0, 1, 1], [0, 0, 0]])


@pytest.mark.parametrize("average", ["macro", "micro"])
@pytest.mark.parametrize("streaming, batch", [(Precision, metrics21.precision), (Recall, metrics21.recall),
                                              (F1, metrics21.f1)])
def test_streaming_scores_equal_the_2_1_batch_scores(streaming, batch, average):
    y_true, y_pred = noisy_labels()
    assert stream(streaming(average=average), y_true, y_pred, n_chunks=9) == pytest.approx(
        batch(y_true, y_pred, average=average))


def test_binary_average_with_pos_label():
    y_true, y_pred = np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 0])
    for streaming, batch in ((Precision, metrics21.precision), (Recall, metrics21.recall), (F1, metrics21.f1)):
        assert stream(streaming(average="binary", pos_label=1), y_true, y_pred, n_chunks=2) == pytest.approx(
            batch(y_true, y_pred, average="binary", pos_label=1))


def test_macro_f1_is_the_mean_of_per_label_f1_not_the_f1_of_means():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    per_label_f1 = np.array([4 / 6, 2 / 5, 4 / 5])  # 2TP / (2TP + FP + FN) for labels 0, 1, 2
    precision = Precision().update(y_true, y_pred).result()
    recall = Recall().update(y_true, y_pred).result()
    macro_f1 = F1().update(y_true, y_pred).result()
    assert macro_f1 == pytest.approx(per_label_f1.mean())
    assert abs(macro_f1 - 2 * precision * recall / (precision + recall)) > 0.01


def test_zero_division_reset_and_empty_results():
    assert Precision(average="binary").update([0, 0, 1], [0, 0, 0]).result() == 0.0  # 1 is never predicted
    assert np.isnan(Accuracy().result()) and np.isnan(F1().result())
    assert np.isnan(Accuracy().update([1], [1]).reset().result())
    with pytest.raises(ValueError):
        Accuracy().update([1, 2], [1])
    with pytest.raises(ValueError):
        F1(average="weighted")
    with pytest.raises(ValueError):
        F1(average="binary").update([0, 1, 2], [0, 1, 2]).result()  # the 2.1 rule: binary needs 2 labels


def test_nan_labels_and_scores_are_rejected():
    for metric in (Accuracy(), ConfusionMatrix(), F1()):
        with pytest.raises(ValueError):
            metric.update([1.0, np.nan], [1.0, 0.0])
    with pytest.raises(ValueError):
        Accuracy().update([1.0, 0.0], [1.0, np.nan])
    with pytest.raises(ValueError):
        AUC().update([0, 1], [0.5, np.nan])
    with pytest.raises(ValueError):
        AUC().update([0.0, np.nan], [0.5, 0.5])
    with pytest.raises(ValueError):
        ConfusionMatrix().update([1, 2], [1, 2]).update(["a"], ["a"])  # labels switch from numbers to text


def test_a_rejected_update_leaves_the_confusion_matrix_unchanged():
    matrix = ConfusionMatrix().update([0, 1], [0, 1])
    with pytest.raises(ValueError):
        matrix.update([2.0, np.nan], [2.0, 0.0])  # label 2 is new, but the NaN must be rejected before any change
    counts, labels = matrix.result()
    assert labels.tolist() == [0, 1] and counts.sum() == 2


def test_auc_matches_the_pairwise_definition_and_handles_ties():
    assert AUC().update([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).result() == pytest.approx(0.75)
    assert AUC().update([0, 1], [0.5, 0.5]).result() == pytest.approx(0.5)  # a tie counts half
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 2000)
    scores = np.clip(rng.normal(0.4 + 0.2 * y, 0.2), 0, 1)
    pos, neg = scores[y == 1], scores[y == 0]
    exact = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])
    assert stream(AUC(), y, scores, n_chunks=10) == pytest.approx(exact, abs=1e-3)
    assert np.isnan(AUC().update([1, 1], [0.2, 0.9]).result())  # needs both classes
    with pytest.raises(ValueError):
        AUC().update([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError):
        AUC(n_bins=0)


def test_auc_agrees_with_the_2_1_batch_roc_curve():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 500)
    scores = rng.random(500) * 0.6 + 0.4 * y
    fpr, tpr, _ = metrics21.roc_curve(y, scores)
    assert stream(AUC(), y, scores, n_chunks=7) == pytest.approx(metrics21.auc(fpr, tpr), abs=2e-3)


def test_rolling_metric_only_uses_the_last_window():
    rolling = RollingMetric(Accuracy(), window=4).update([1, 1, 1, 1], [1, 1, 1, 1]).update([1, 1], [0, 0])
    assert rolling.result() == pytest.approx(0.5)  # last four: two right, two wrong
    expected = F1().update(Y_TRUE[-5:], Y_PRED[-5:]).result()
    assert stream(RollingMetric(F1(), window=5), Y_TRUE, Y_PRED) == pytest.approx(expected)
    assert np.isnan(RollingMetric(Accuracy()).result())
    for window in (0, 2.5):
        with pytest.raises(ValueError):
            RollingMetric(Accuracy(), window=window)


def test_empty_chunks_change_nothing():
    matrix, labels = ConfusionMatrix().update([], []).result()
    assert matrix.shape == (0, 0) and len(labels) == 0
    assert np.isnan(Accuracy().update([], []).result())
