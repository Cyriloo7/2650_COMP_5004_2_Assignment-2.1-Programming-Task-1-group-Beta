"""Unit tests for numcompute.metrics."""

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

    @pytest.mark.parametrize(
        "bad", [np.zeros((2, 2), dtype=int), np.int64(1)], ids=["2-D", "scalar"]
    )
    def test_non_1d_input_raises_value_error(self, bad):
        with pytest.raises(ValueError, match="1-D"):
            confusion_matrix(bad, bad)

    @pytest.mark.parametrize(
        "y_true, y_pred",
        [([0.0, np.nan], [0.0, 1.0]), ([0.0, 1.0], [np.nan, 1.0])],
        ids=["nan-in-y_true", "nan-in-y_pred"],
    )
    def test_nan_labels_raise_value_error(self, y_true, y_pred):
        with pytest.raises(ValueError, match="NaN"):
            confusion_matrix(y_true, y_pred)

    def test_string_labels_are_supported(self):
        matrix, labels = confusion_matrix(["cat", "dog", "cat"], ["cat", "cat", "cat"])
        np.testing.assert_array_equal(labels, ["cat", "dog"])
        np.testing.assert_array_equal(matrix, [[2, 0], [1, 0]])


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

    @pytest.mark.parametrize("fn", [precision, recall, f1])
    @pytest.mark.parametrize("average", ["binary", "macro", "micro"])
    def test_single_class_present_is_scored_for_every_average(self, fn, average):
        y = np.array([1, 1, 1])
        assert fn(y, y, average=average) == 1.0

    @pytest.mark.parametrize("fn", [precision, recall, f1])
    def test_binary_single_class_with_a_non_default_pos_label(self, fn):
        y = np.array([0, 0, 0])
        assert fn(y, y, average="binary", pos_label=0) == 1.0

    def test_binary_with_only_the_negative_class_present_raises(self):
        y = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="pos_label"):
            precision(y, y, average="binary", pos_label=1)

    def test_binary_with_three_classes_reports_the_limit(self):
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="at most 2"):
            recall(y, y, average="binary")


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

    @pytest.mark.parametrize("y_true", [[0, 1], [1, 0]])
    def test_fully_tied_scores_give_auc_half_in_both_row_orders(self, y_true):
        fpr, tpr, _ = roc_curve(np.array(y_true), np.array([0.5, 0.5]))
        assert auc(fpr, tpr) == 0.5

    def test_fully_tied_scores_with_many_samples_give_auc_half(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        fpr, tpr, _ = roc_curve(y_true, np.full(8, 0.3))
        assert auc(fpr, tpr) == 0.5

    def test_one_point_per_distinct_score(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_score = np.array([0.8, 0.8, 0.4, 0.4, 0.1, 0.1])
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        np.testing.assert_array_equal(thresholds, [np.inf, 0.8, 0.4, 0.1])
        np.testing.assert_allclose(fpr, [0, 1 / 3, 2 / 3, 1])
        np.testing.assert_allclose(tpr, [0, 1 / 3, 2 / 3, 1])

    def test_thresholds_start_at_positive_infinity(self):
        _, _, thresholds = roc_curve(np.array([0, 1]), np.array([0.2, 0.7]))
        assert thresholds[0] == np.inf

    def test_auc_matches_the_pairwise_definition_on_random_data_with_ties(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            y_true = rng.integers(0, 2, size=60)
            y_true[:2] = [0, 1]  # guarantee both classes
            y_score = rng.integers(0, 5, size=60).astype(float)  # few values: many ties
            pos, neg = y_score[y_true == 1], y_score[y_true == 0]
            expected = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(
                pos[:, None] == neg[None, :]
            )
            fpr, tpr, _ = roc_curve(y_true, y_score)
            np.testing.assert_allclose(auc(fpr, tpr), expected)

    def test_auc_does_not_change_when_rows_are_shuffled(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=80)
        y_true[:2] = [0, 1]
        y_score = rng.integers(0, 4, size=80).astype(float)
        baseline = auc(*roc_curve(y_true, y_score)[:2])
        for _ in range(10):
            order = rng.permutation(80)
            shuffled = auc(*roc_curve(y_true[order], y_score[order])[:2])
            np.testing.assert_allclose(shuffled, baseline)

    def test_infinite_scores_are_merged_like_any_other_tie(self, recwarn):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_score = np.array([np.inf, np.inf, 0.0, 0.0, -np.inf, -np.inf])
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        assert len(recwarn) == 0
        np.testing.assert_array_equal(thresholds, [np.inf, np.inf, 0.0, -np.inf])
        np.testing.assert_allclose(auc(fpr, tpr), 0.5)

    def test_non_monotonic_fpr_raises_value_error(self):
        with pytest.raises(ValueError, match="monotonic"):
            auc(np.array([0.0, 1.0, 0.5]), np.array([0.0, 1.0, 0.5]))

    @pytest.mark.parametrize("n_points", [0, 1])
    def test_fewer_than_two_points_raises_value_error(self, n_points):
        with pytest.raises(ValueError, match="at least 2"):
            auc(np.zeros(n_points), np.zeros(n_points))

    def test_decreasing_fpr_gives_the_same_positive_area(self):
        fpr = np.array([0.0, 0.25, 1.0])
        tpr = np.array([0.0, 0.75, 1.0])
        assert auc(fpr, tpr) > 0
        np.testing.assert_allclose(auc(fpr[::-1], tpr[::-1]), auc(fpr, tpr))

    def test_constant_fpr_has_zero_area(self):
        assert auc(np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.5, 1.0])) == 0.0


@pytest.mark.parametrize(
    "make_view", [lambda a: a[::2], lambda a: a[::-1]], ids=["every-other", "reversed"]
)
class TestNonContiguousInput:
    def test_classification_metrics_match_contiguous_copies(self, make_view):
        rng = np.random.default_rng(0)
        y_true, y_pred = rng.integers(0, 2, size=40), rng.integers(0, 2, size=40)
        true_view, pred_view = make_view(y_true), make_view(y_pred)
        assert not true_view.flags["C_CONTIGUOUS"]
        true_copy, pred_copy = true_view.copy(), pred_view.copy()
        for fn in (accuracy, precision, recall, f1):
            assert fn(true_view, pred_view) == fn(true_copy, pred_copy)
        matrix, labels = confusion_matrix(true_view, pred_view)
        matrix_copy, labels_copy = confusion_matrix(true_copy, pred_copy)
        np.testing.assert_array_equal(matrix, matrix_copy)
        np.testing.assert_array_equal(labels, labels_copy)

    def test_mse_matches_contiguous_copy(self, make_view):
        rng = np.random.default_rng(1)
        true_view = make_view(rng.normal(size=40))
        pred_view = make_view(rng.normal(size=40))
        assert mse(true_view, pred_view) == mse(true_view.copy(), pred_view.copy())

    def test_roc_curve_and_auc_match_contiguous_copies(self, make_view):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, size=40)
        y_true[:2] = [0, 1]
        true_view = make_view(y_true)
        score_view = make_view(np.round(rng.random(40), 1))  # rounded: many ties
        got = roc_curve(true_view, score_view)
        expected = roc_curve(true_view.copy(), score_view.copy())
        for actual_array, expected_array in zip(got, expected):
            np.testing.assert_array_equal(actual_array, expected_array)
        assert auc(*got[:2]) == auc(*expected[:2])