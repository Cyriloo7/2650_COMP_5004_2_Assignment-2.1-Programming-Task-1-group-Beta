"""
Unit tests for numcompute.metrics.

Run from the project root with:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from numcompute.metrics import (
    confusion_matrix,
    accuracy,
    precision,
    recall,
    f1,
    mse,
    roc_curve,
    auc,
)


# ---------------------------------------------------------------------------
# confusion_matrix()
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_matches_a_manual_loop(self):
        y_true = np.array([0, 1, 2, 2, 0, 1, 1, 2, 0, 1])
        y_pred = np.array([0, 2, 2, 2, 0, 0, 1, 2, 0, 1])
        matrix, labels = confusion_matrix(y_true, y_pred)

        expected = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred):
            expected[t, p] += 1

        np.testing.assert_array_equal(matrix, expected)
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_default_labels_are_sorted_unique_values(self):
        y_true = np.array([5, 3, 3])
        y_pred = np.array([3, 3, 5])
        _, labels = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(labels, [3, 5])

    def test_explicit_unsorted_labels_still_give_correct_result(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        matrix, labels = confusion_matrix(y_true, y_pred, labels=[2, 0, 1])
        np.testing.assert_array_equal(labels, [0, 1, 2])
        np.testing.assert_array_equal(matrix, np.eye(3, dtype=int))

    def test_value_not_in_explicit_labels_raises(self):
        with pytest.raises(ValueError):
            confusion_matrix(np.array([0, 1, 5]), np.array([0, 1, 1]), labels=[0, 1])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            confusion_matrix(np.array([0, 1, 1]), np.array([0, 1]))


# ---------------------------------------------------------------------------
# accuracy()
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 1, 0, 2])
        assert accuracy(y, y) == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.0

    def test_partial_correct(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.75

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            accuracy(np.array([0, 1]), np.array([0, 1, 1]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            accuracy(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# precision / recall / f1
# ---------------------------------------------------------------------------

class TestPrecisionRecallF1:
    def test_perfect_predictions_binary(self):
        y = np.array([0, 1, 0, 1, 1])
        for fn in (precision, recall, f1):
            assert fn(y, y, average="binary") == 1.0

    def test_all_wrong_predictions_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        for fn in (precision, recall, f1):
            assert fn(y_true, y_pred, average="binary") == 0.0

    def test_class_never_predicted_gives_zero_precision_not_nan(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # class 1 never predicted
        result = precision(y_true, y_pred, average="binary", pos_label=1)
        assert result == 0.0
        assert not np.isnan(result)

    def test_macro_is_the_unweighted_mean_of_per_class_scores(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 2])
        # class 0: precision = 2/3 (one false positive from class 1)
        # class 1: precision = 1/1 = 1.0 (only one prediction, correct)
        # class 2: precision = 2/2 = 1.0
        expected_macro_precision = (2 / 3 + 1.0 + 1.0) / 3
        assert np.isclose(
            precision(y_true, y_pred, average="macro"), expected_macro_precision
        )

    def test_micro_equals_accuracy_in_multiclass_single_label_case(self):
        y_true = np.array([0, 1, 2, 2, 0, 1, 1, 2, 0, 1])
        y_pred = np.array([0, 2, 2, 2, 0, 0, 1, 2, 0, 1])
        acc = accuracy(y_true, y_pred)
        assert np.isclose(precision(y_true, y_pred, average="micro"), acc)
        assert np.isclose(recall(y_true, y_pred, average="micro"), acc)
        assert np.isclose(f1(y_true, y_pred, average="micro"), acc)

    def test_macro_f1_averages_per_class_f1_not_macro_precision_and_recall(self):
        # Hand-built so the two approaches clearly disagree:
        #   correct way  (mean of per-class F1)        ~= 0.1818
        #   wrong way    (F1 of macro-P and macro-R)    == 0.55
        y_true = np.array([0] + [1] * 10)
        y_pred = np.array([0] + [0] * 9 + [1])

        result = f1(y_true, y_pred, average="macro")
        wrong_way = 2 * 0.55 * 0.55 / (0.55 + 0.55)

        assert np.isclose(result, 0.18181818181818182)
        assert not np.isclose(result, wrong_way)

    def test_binary_with_more_than_two_classes_raises(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        with pytest.raises(ValueError):
            precision(y_true, y_pred, average="binary")

    def test_binary_with_pos_label_not_among_the_two_classes_raises(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        with pytest.raises(ValueError):
            precision(y_true, y_pred, average="binary", pos_label=5)

    @pytest.mark.parametrize("fn", [precision, recall, f1])
    def test_invalid_average_raises(self, fn):
        y = np.array([0, 1])
        with pytest.raises(ValueError):
            fn(y, y, average="bogus")


# ---------------------------------------------------------------------------
# mse()
# ---------------------------------------------------------------------------

class TestMse:
    def test_known_value(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        assert np.isclose(mse(y_true, y_pred), 0.375)

    def test_perfect_predictions_give_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            mse(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# roc_curve() / auc()
# ---------------------------------------------------------------------------

class TestRocCurveAndAuc:
    def test_curve_starts_at_origin_and_ends_at_one_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        assert fpr[0] == 0.0 and tpr[0] == 0.0
        assert fpr[-1] == 1.0 and tpr[-1] == 1.0

    def test_perfectly_separated_scores_give_auc_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        assert auc(fpr, tpr) == 1.0

    def test_perfectly_inverted_scores_give_auc_zero(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        assert auc(fpr, tpr) == 0.0

    def test_non_binary_y_true_raises(self):
        with pytest.raises(ValueError):
            roc_curve(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))

    def test_single_class_present_raises(self):
        with pytest.raises(ValueError):
            roc_curve(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            roc_curve(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))

    def test_auc_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            auc(np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0]))