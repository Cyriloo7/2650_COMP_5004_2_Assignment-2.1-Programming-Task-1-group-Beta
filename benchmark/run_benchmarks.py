"""
benchmark/run_benchmarks.py -- vectorised NumCompute vs plain Python loops.

Run from the repo root:

    python -m benchmark.run_benchmarks

What it does
------------
For each case below it times the NumCompute (vectorised) function and a
hand-written pure-Python loop that computes the same thing, using
``numcompute.benchmarking.compare``. ``compare`` also checks that both
give the same answer, so a speedup is only reported for correct code.

The data comes from a fixed random seed, so results are reproducible
(the timings themselves still vary a little between machines and runs).
The environment details (Python, NumPy, OS, CPU) are printed and saved
next to the table so the numbers can be interpreted later.

Output is also written to ``benchmark/results.md``.
"""

import math
import os
import platform
import sys
from datetime import date

import numpy as np

from numcompute.benchmarking import compare, format_results_table
from numcompute.preprocessing import StandardScaler
from numcompute.rank import rank
from numcompute.utils import euclidean_distance


# ----------------------------------------------------------------------
# Loop versions (deliberately plain Python, no NumPy vector tricks)
# ----------------------------------------------------------------------

def standardise_vectorised(X):
    """z-score every column with NumCompute's StandardScaler."""
    return StandardScaler().fit_transform(X)


def standardise_loop(X):
    """z-score every column with explicit Python loops."""
    n_rows, n_cols = X.shape
    out = np.empty((n_rows, n_cols))
    for j in range(n_cols):
        total = 0.0
        for i in range(n_rows):
            total += X[i, j]
        mean = total / n_rows
        sq = 0.0
        for i in range(n_rows):
            sq += (X[i, j] - mean) ** 2
        std = math.sqrt(sq / n_rows)
        for i in range(n_rows):
            out[i, j] = (X[i, j] - mean) / std
    return out


def pairwise_distance_vectorised(A, B):
    """All-pairs Euclidean distance with NumCompute's utils."""
    return euclidean_distance(A, B)


def pairwise_distance_loop(A, B):
    """All-pairs Euclidean distance with three nested loops."""
    out = np.empty((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            s = 0.0
            for k in range(A.shape[1]):
                diff = A[i, k] - B[j, k]
                s += diff * diff
            out[i, j] = math.sqrt(s)
    return out


def rank_vectorised(x):
    """Average ranks (ties share the mean rank) with NumCompute's rank."""
    return rank(x, method="average")


def rank_loop(x):
    """Average ranks by comparing every pair of values (O(n^2))."""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        smaller = 0
        equal = 0
        for j in range(n):
            if x[j] < x[i]:
                smaller += 1
            elif x[j] == x[i]:
                equal += 1
        out[i] = smaller + (equal + 1) / 2.0
    return out


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def run_all(seed=0, repeats=5):
    """Run every benchmark case and return the list of ``compare`` results.

    Parameters
    ----------
    seed : int, default 0
        Seed for the random test data (makes the inputs reproducible).
    repeats : int, default 5
        Timed runs per function (the mean is reported).

    Returns
    -------
    list of dict
        One ``compare()`` result per case, ready for
        ``format_results_table``. The ``name`` is changed to say what was
        measured and on how much data.
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(20_000, 5))
    A = rng.normal(size=(150, 3))
    B = rng.normal(size=(150, 3))
    x = rng.integers(0, 500, size=1_500).astype(float)  # many ties

    cases = [
        ("StandardScaler (20,000 x 5)",
         standardise_vectorised, standardise_loop, (X,)),
        ("euclidean_distance (150 x 150 pairs)",
         pairwise_distance_vectorised, pairwise_distance_loop, (A, B)),
        ("rank, average ties (1,500 values)",
         rank_vectorised, rank_loop, (x,)),
    ]

    results = []
    for label, vec_fn, loop_fn, args in cases:
        result = compare(vec_fn, loop_fn, *args, repeats=repeats)
        result["name"] = label
        results.append(result)
    return results


def environment_notes():
    """Return a short multi-line description of the machine and libraries."""
    return "\n".join([
        f"- Date: {date.today().isoformat()}",
        f"- Python: {sys.version.split()[0]}",
        f"- NumPy: {np.__version__}",
        f"- OS: {platform.platform()}",
        f"- CPU: {platform.processor() or platform.machine()} "
        f"({os.cpu_count()} logical cores)",
        "- Method: mean of 5 timed runs after 1 warm-up run; fixed random seed 0",
    ])


def main():
    results = run_all()
    table = format_results_table(results)
    notes = environment_notes()

    report = (
        "# Benchmark results: vectorised vs Python loops\n\n"
        f"{table}\n\n"
        "## Environment\n\n"
        f"{notes}\n"
    )
    print(report)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()