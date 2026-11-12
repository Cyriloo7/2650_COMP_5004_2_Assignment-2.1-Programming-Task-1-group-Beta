"""Plots for streaming experiments (matplotlib).

Each function returns the matplotlib Figure. Use ``save_path`` to save it to
a file and ``show=True`` to display it in a script (in Jupyter the figure is
shown inline automatically).
"""

import matplotlib.pyplot as plt
import numpy as np

from numcompute.metrics import confusion_matrix  # Assignment 2.1 (reused)


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Value", xlabel="Chunk",
                          drift_at=None, save_path=None, show=False):
    """Plot a metric (e.g. accuracy) across chunks.

    ``metric_values`` is one sequence of values, or a dict {label: values}
    to draw several lines. ``drift_at`` marks a chunk with a dashed line
    (e.g. a known concept drift). NaN values (e.g. the first, training-only
    chunk) are left as gaps.
    """
    series = metric_values if isinstance(metric_values, dict) else {ylabel: metric_values}
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in series.items():
        ax.plot(np.asarray(values, dtype=float), marker="o", markersize=3, label=label)
    if drift_at is not None:
        ax.axvline(drift_at, color="grey", linestyle="--", label="drift")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    if len(series) > 1 or drift_at is not None:
        ax.legend()
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title="Model comparison",
                   ylabel="Accuracy", drift_at=None, save_path=None, show=False):
    """Compare two models on the same streaming metric, chunk by chunk (e.g. trees with different buffer_size values).

    The legend shows each model's mean; the shaded area is the gap between them.
    """
    metric1, metric2 = np.asarray(metric1, dtype=float), np.asarray(metric2, dtype=float)
    if metric1.shape != metric2.shape:
        raise ValueError(f"metric1 and metric2 must have the same length, got {len(metric1)} and {len(metric2)}.")
    fig, ax = plt.subplots(figsize=(8, 4))
    for values, label in ((metric1, labels[0]), (metric2, labels[1])):
        ax.plot(values, marker="o", markersize=3, label=f"{label} (mean {np.nanmean(values):.3f})")
    ax.fill_between(np.arange(len(metric1)), metric1, metric2, color="grey", alpha=0.2)
    if drift_at is not None:
        ax.axvline(drift_at, color="grey", linestyle="--", label="drift")
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs ground truth",
                                     save_path=None, show=False):
    """Compare predictions with the true labels of one chunk (e.g. the latest).

    Left: each sample's true class (circle) and predicted class (cross, red
    when wrong). Right: the confusion matrix of the chunk.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}.")
    matrix, labels = confusion_matrix(y_true, y_pred)
    true_index, pred_index = np.searchsorted(labels, y_true), np.searchsorted(labels, y_pred)
    wrong = true_index != pred_index
    samples = np.arange(len(y_true))

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    left.scatter(samples, true_index, facecolors="none", edgecolors="tab:blue", label="true")
    left.scatter(samples[~wrong], pred_index[~wrong], marker="x", color="tab:green", label="predicted (correct)")
    left.scatter(samples[wrong], pred_index[wrong], marker="x", color="tab:red", label="predicted (wrong)")
    left.set_yticks(range(len(labels)), [str(label) for label in labels])
    left.set(xlabel="Sample in chunk", title=f"Per sample (accuracy {np.mean(~wrong):.1%})")
    left.legend(fontsize=8)

    right.imshow(matrix, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            right.text(j, i, matrix[i, j], ha="center", va="center", color=colour)
    right.set_xticks(range(len(labels)), [str(label) for label in labels])
    right.set_yticks(range(len(labels)), [str(label) for label in labels])
    right.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
    fig.suptitle(title)
    return _finish(fig, save_path, show)
