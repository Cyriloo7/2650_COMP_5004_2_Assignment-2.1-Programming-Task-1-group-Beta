"""Tests for visualise.py: the plots are created and can be saved."""

import pytest

matplotlib = pytest.importorskip("matplotlib")  # the 2.1 requirements only list numpy
matplotlib.use("Agg")  # draw without a screen

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from numcompute_stream import visualise as vis  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_metric_over_time_single_and_multiple_lines(tmp_path):
    fig = vis.plot_metric_over_time([np.nan, 0.6, 0.8], title="Accuracy", ylabel="accuracy",
                                    save_path=tmp_path / "a.png")
    assert (tmp_path / "a.png").exists() and fig.axes[0].get_title() == "Accuracy"
    fig = vis.plot_metric_over_time({"tree": [0.5, 0.6], "forest": [0.6, 0.7]}, drift_at=1)
    assert len(fig.axes[0].get_lines()) == 3  # two models + the drift line


def test_compare_models(tmp_path):
    fig = vis.compare_models([0.5, 0.6, 0.7], [0.6, 0.7, 0.9], labels=("tree", "forest"),
                             save_path=tmp_path / "c.png")
    assert (tmp_path / "c.png").exists()
    legend = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert legend[0].startswith("tree") and "mean 0.733" in legend[1]
    with pytest.raises(ValueError):
        vis.compare_models([1, 2], [1, 2, 3])


def test_plot_predictions_vs_ground_truth(tmp_path):
    fig = vis.plot_predictions_vs_ground_truth(["a", "b", "b", "c"], ["a", "b", "c", "c"],
                                               save_path=tmp_path / "p.png")
    assert (tmp_path / "p.png").exists() and len(fig.axes) == 2
    assert "75.0%" in fig.axes[0].get_title()
    with pytest.raises(ValueError):
        vis.plot_predictions_vs_ground_truth([1, 2], [1])


def test_show_option_and_drift_line_in_comparisons():
    fig = vis.compare_models([0.5, 0.6], [0.6, 0.7], drift_at=1, show=True)  # show() does nothing on the Agg backend
    assert len(fig.axes[0].get_lines()) == 3  # two models + the drift line
