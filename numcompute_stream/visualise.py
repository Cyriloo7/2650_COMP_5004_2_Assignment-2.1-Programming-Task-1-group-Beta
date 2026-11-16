"""Plots for streaming experiments (matplotlib).

The three plots of the specification - ``plot_metric_over_time``,
``compare_models`` and ``plot_predictions_vs_ground_truth`` - plus
``plot_parameter_sweep`` for benchmarks over a setting such as
``buffer_size``. Each function returns the matplotlib Figure. Use
``save_path`` to save it to a file and ``show=True`` to display it in a
script (in Jupyter the figure is shown inline automatically).
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from numcompute.metrics import confusion_matrix  # Assignment 2.1 (reused)


def _style_chunk_axis(ax, title, xlabel, ylabel):
    """Labels, a light grid and whole-number ticks: chunks are counted, so 2.5 is never a tick."""
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)


def _finish(fig, save_path, show):
    """Lay out the figure, then save it and/or show it as asked; return it."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Value", xlabel="Chunk",
                          drift_at=None, save_path=None, show=False):
    """Plot a metric (e.g. accuracy) across chunks.

    Parameters
    ----------
    metric_values : array-like of shape (n_chunks,), or dict of str -> array-like
        One series, or several named series drawn as separate lines (e.g.
        ``trainer.get("rolling_accuracy")``). NaN values, such as chunks
        logged before the first retrain, are left as gaps.
    title, ylabel, xlabel : str
        Plot labels.
    drift_at : int, optional
        Chunk to mark with a dashed vertical line (e.g. a known concept drift).
    save_path : str or Path, optional
        Save the figure there (PNG at 150 dpi, or the format of the extension).
    show : bool, default False
        Call ``plt.show()`` (in Jupyter the figure is shown inline anyway).

    Returns
    -------
    matplotlib.figure.Figure
    """
    series = metric_values if isinstance(metric_values, dict) else {ylabel: metric_values}
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in series.items():
        ax.plot(np.asarray(values, dtype=float), marker="o", markersize=3, label=label)
    if drift_at is not None:
        ax.axvline(drift_at, color="grey", linestyle="--", label="drift")
    _style_chunk_axis(ax, title, xlabel, ylabel)
    if len(series) > 1 or drift_at is not None:
        ax.legend()
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title="Model comparison",
                   ylabel="Accuracy", drift_at=None, save_path=None, show=False):
    """Compare two models on the same streaming metric, chunk by chunk (e.g. two buffer_size values).

    The legend shows each model's mean (ignoring NaN); the shaded area is the
    gap between the two lines.

    Parameters
    ----------
    metric1, metric2 : array-like of shape (n_chunks,)
        The metric of each model, logged on the same chunks.
    labels : (str, str), default ("Model 1", "Model 2")
    title, ylabel : str
        Plot labels.
    drift_at : int, optional
        Chunk to mark with a dashed vertical line.
    save_path : str or Path, optional
        Save the figure there.
    show : bool, default False
        Call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the two series have different lengths.
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
    _style_chunk_axis(ax, title, "Chunk", ylabel)
    ax.legend()
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs ground truth",
                                     save_path=None, show=False):
    """Compare predictions with the true labels of one chunk (e.g. the latest).

    Left: each sample's true class (circle) and predicted class (cross, red
    when wrong), with the chunk accuracy in the title. Right: the confusion
    matrix of the chunk (from the 2.1 ``confusion_matrix``).

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        True and predicted labels, e.g. ``trainer.last_y_true`` and
        ``trainer.last_y_pred``.
    title : str
    save_path : str or Path, optional
        Save the figure there.
    show : bool, default False
        Call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the two inputs have different shapes.
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


def plot_parameter_sweep(parameter_values, metric_values, title="Parameter sweep", xlabel="Parameter",
                         ylabel="Value", baselines=None, save_path=None, show=False):
    """Plot one or more results against the values of a setting (e.g. accuracy against ``buffer_size``).

    Parameters
    ----------
    parameter_values : sequence of shape (n_values,)
        The settings tried, e.g. ``(50, 100, 200, 300)``; they are spaced
        evenly on the x axis and used as its tick labels.
    metric_values : array-like of shape (n_values,), or dict of str -> array-like
        One series, or several named series (e.g. file order and shuffled order).
    title, xlabel, ylabel : str
        Plot labels.
    baselines : dict of str -> float, optional
        Reference levels drawn as dashed horizontal lines (e.g. the accuracy
        of always predicting the most common label).
    save_path : str or Path, optional
        Save the figure there.
    show : bool, default False
        Call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If a series does not have one value per setting.
    """
    positions = np.arange(len(parameter_values))
    series = metric_values if isinstance(metric_values, dict) else {ylabel: metric_values}
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in series.items():
        values = np.asarray(values, dtype=float)
        if values.shape != positions.shape:
            raise ValueError(f"{label!r} has {values.size} values, expected one per setting ({len(positions)}).")
        ax.plot(positions, values, marker="o", label=label)
    for (label, level), style in zip((baselines or {}).items(), itertools.cycle(["--", ":", "-."])):
        ax.axhline(level, color="grey", linestyle=style, label=label)
    ax.set_xticks(positions, [str(value) for value in parameter_values])
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    if len(series) > 1 or baselines:
        ax.legend()
    return _finish(fig, save_path, show)
