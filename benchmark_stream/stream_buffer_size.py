"""Benchmark (b): how buffer_size changes accuracy, retraining time and memory.

Run from the repository root:

    python -m benchmark_stream.stream_buffer_size

The stream is Dataset/Housing.csv (the dataset of Assignment 2.1, 544 houses
with a known air-conditioning label), cut into chunks of 25 rows. The same
pipeline - encode text columns, fill missing values, scale, then a
DecisionTreeClassifier(max_depth=4) - is trained with different buffer sizes,
with test-then-train evaluation.

Two orders of the same data are compared:

* file order: the file is sorted by price from high to low, so the houses
  (and the share with air conditioning) change as the stream goes on;
* shuffled order (seed 0): no change over time, so only buffer_size matters.

Every table also shows the accuracy of always predicting the most common
label ("no air conditioning") on the same chunks, so the scores can be
judged against it. Results go to benchmark_stream/stream_buffer_size.md and
four plots: rolling accuracy over the stream, buffer sizes 50 vs 200, and
accuracy and retraining time against buffer_size.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # also works as: python benchmark_stream/<this file>

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save the plots without opening a window

import numpy as np  # noqa: E402

from numcompute.io import DEFAULT_MISSING_VALUES  # noqa: E402  Assignment 2.1
from numcompute_stream import (  # noqa: E402
    DecisionTreeClassifier, Imputer, OneHotEncoder, Pipeline, StandardScaler, StreamTrainer, iter_chunks, load_csv,
)
from numcompute_stream import visualise as vis  # noqa: E402

DATA_PATH = ROOT / "Dataset" / "Housing.csv"
TARGET = "airconditioning"
TEXT_COLUMNS = ["mainroad", "guestroom", "basement", "hotwaterheating", "prefarea", "furnishingstatus"]
MISSING = (*DEFAULT_MISSING_VALUES, "NAN")  # the file also writes missing cells as NAN
CHUNK_SIZE = 25
WINDOW = 100  # samples in the rolling metrics
BUFFER_SIZES = (50, 100, 200, 300)
HERE = Path(__file__).parent


def make_pipeline(buffer_size, text_columns):
    return Pipeline([
        ("encode", OneHotEncoder(columns=text_columns, max_categories=3)),  # 3 slots: furnishingstatus has 3 values
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size)),
    ])


def run_all(X, y, text_columns):
    """Train one pipeline per buffer size on the stream; return {buffer_size: StreamTrainer}."""
    return {size: StreamTrainer(make_pipeline(size, text_columns), window=WINDOW).run(iter_chunks(X, y, CHUNK_SIZE))
            for size in BUFFER_SIZES}


def majority_accuracy(labels, y_all):
    """Accuracy of always predicting the most common label of the whole stream ``y_all``."""
    values, counts = np.unique(y_all, return_counts=True)
    return np.mean(labels == values[np.argmax(counts)])


def common_accuracy(trainer):
    """Accuracy on the chunks every buffer size can score (weighted by chunk size)."""
    common = max(BUFFER_SIZES) // CHUNK_SIZE  # from this chunk on, every buffer size has a tree
    accuracy, sizes = trainer.get("accuracy")[common:], trainer.get("n_samples")[common:]
    return np.sum(accuracy * sizes) / sizes.sum()


def mean_retrain_time(trainer):
    """Mean seconds per retrain (chunks without a retrain are logged as NaN and skipped)."""
    return np.nanmean(trainer.get("retrain_time"))


def results_table(trainers, y, n_chunks):
    """Markdown table: one row per buffer size, plus the majority-label baseline."""
    common = max(BUFFER_SIZES) // CHUNK_SIZE  # from this chunk on, every buffer size has a tree
    rows = slice(common * CHUNK_SIZE, None)   # the samples of those chunks
    lines = ["| buffer_size | First scored chunk | Retrains | Mean retrain time (ms) "
             f"| Accuracy (chunks {common}-{n_chunks - 1}) | Accuracy (all scored chunks) "
             "| Training time (s) | Peak memory (KB) |", "|---|---|---|---|---|---|---|---|"]
    for size, trainer in trainers.items():
        first_scored = int(np.argmax(~np.isnan(trainer.get("accuracy"))))
        lines.append(f"| {size} | {first_scored} | {int(trainer.get('n_retrains')[-1])} | "
                     f"{1000 * mean_retrain_time(trainer):.1f} | {common_accuracy(trainer):.3f} | "
                     f"{trainer.get('cumulative_accuracy')[-1]:.3f} | {trainer.get('train_time').sum():.3f} | "
                     f"{trainer.get('memory_kb').max():.0f} |")
    lines.append("| always predict the most common label | - | - | - "
                 f"| {majority_accuracy(y[rows], y):.3f} | {majority_accuracy(y, y):.3f} | - | - |")
    return "\n".join(lines)


def main():
    X, y, names = load_csv(DATA_PATH, target_column=TARGET, missing_values=MISSING)
    text_columns = [names.index(name) for name in TEXT_COLUMNS]
    n_chunks = -(-len(y) // CHUNK_SIZE)
    order = np.random.default_rng(0).permutation(len(y))

    file_order = run_all(X, y, text_columns)
    shuffled = run_all(X[order], y[order], text_columns)
    text = (f"Stream: Dataset/Housing.csv, {len(y)} houses, label `{TARGET}`, chunks of {CHUNK_SIZE} rows "
            f"({n_chunks} chunks). A model is scored only once its first buffer has been trained, so the accuracy "
            f"column for chunks {max(BUFFER_SIZES) // CHUNK_SIZE}-{n_chunks - 1} compares every buffer size on the "
            "same samples.\n\n"
            "**File order** (sorted by price, high to low, so the data changes over the stream):\n\n"
            + results_table(file_order, y, n_chunks)
            + "\n\n**Shuffled order** (seed 0, no change over time):\n\n"
            + results_table(shuffled, y[order], n_chunks) + "\n")
    print(text)
    (HERE / "stream_buffer_size.md").write_text(text)

    rolling = {f"buffer_size={size}": trainer.get("rolling_accuracy") for size, trainer in file_order.items()}
    vis.plot_metric_over_time(rolling, title=f"Rolling accuracy (last {WINDOW} samples), file order",
                              ylabel="accuracy", save_path=HERE / "stream_buffer_size_accuracy.png")
    vis.compare_models(file_order[50].get("rolling_accuracy"), file_order[200].get("rolling_accuracy"),
                       labels=("buffer_size=50", "buffer_size=200"), title="Rolling accuracy: buffer_size 50 vs 200",
                       ylabel="accuracy", save_path=HERE / "stream_buffer_size_compare.png")

    common = max(BUFFER_SIZES) // CHUNK_SIZE
    labels, counts = np.unique(y, return_counts=True)
    majority = labels[np.argmax(counts)]
    vis.plot_parameter_sweep(
        BUFFER_SIZES, {"file order": [common_accuracy(t) for t in file_order.values()],
                       "shuffled order": [common_accuracy(t) for t in shuffled.values()]},
        title=f"Accuracy on chunks {common}-{n_chunks - 1} against buffer_size", xlabel="buffer_size",
        ylabel="accuracy", save_path=HERE / "stream_buffer_size_sweep_accuracy.png",
        baselines={f"always '{majority}' (file order)": majority_accuracy(y[common * CHUNK_SIZE:], y),
                   f"always '{majority}' (shuffled)": majority_accuracy(y[order][common * CHUNK_SIZE:], y)})
    vis.plot_parameter_sweep(
        BUFFER_SIZES, {"file order": [1000 * mean_retrain_time(t) for t in file_order.values()],
                       "shuffled order": [1000 * mean_retrain_time(t) for t in shuffled.values()]},
        title="Mean time per retrain against buffer_size", xlabel="buffer_size", ylabel="milliseconds",
        save_path=HERE / "stream_buffer_size_sweep_retrain_time.png")


if __name__ == "__main__":
    main()
