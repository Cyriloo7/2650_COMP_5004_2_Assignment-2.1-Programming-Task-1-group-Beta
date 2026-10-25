"""Compare vectorised NumCompute functions with plain-Python loop versions.

The input is the Housing dataset in Dataset/Housing.csv (545 houses, 13 columns).
Rows with a missing value are dropped first, because the loop versions are not
NaN-aware. Run from the repository root:

    python -m benchmark.run_benchmarks

`compare` checks that each pair gives the same answer before reporting a
speedup, so a fast but wrong implementation cannot appear in the table.
"""

import bisect
import math
import os
import platform
from pathlib import Path

import numpy as np

from numcompute.benchmarking import compare, format_results_table
from numcompute.io import read_csv
from numcompute.metrics import auc, confusion_matrix, roc_curve
from numcompute.rank import rank
from numcompute.stats import describe, histogram
from numcompute.utils import euclidean_distance, topk_indices

DATA_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "Housing.csv"
REPEATS = 10
NUMERIC = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
YES_NO = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
MISSING_TEXT = ["", "na", "nan"]  # compared case-insensitively, so "NAN" counts as missing


def load_housing(path):
    """Read the CSV; return the numeric columns, standardised features and the air-conditioning label.

    NumCompute's own reader turns blank, NA and NaN cells into NaN (numbers) or "" (text)
    instead of a silent placeholder. Rows with any missing value are then dropped, because
    the plain-Python loop versions are not NaN-aware.
    """
    if not path.is_file():
        raise SystemExit(f"Cannot find {path}. Put Housing.csv in the Dataset folder.")
    numeric, _ = read_csv(path, usecols=NUMERIC)
    text, _ = read_csv(path, usecols=YES_NO + ["furnishingstatus"], dtype=str)

    missing_text = np.isin(np.char.lower(np.char.strip(text)), MISSING_TEXT)
    complete = ~np.isnan(numeric).any(axis=1) & ~missing_text.any(axis=1)
    numeric, text = numeric[complete], text[complete]

    yes_no = (text[:, : len(YES_NO)] == "yes").astype(float)
    furnishing_status = text[:, len(YES_NO)]
    categories = np.unique(furnishing_status)
    furnishing = (furnishing_status[:, None] == categories[None, :]).astype(float)

    # Every column, with text columns encoded as numbers and all columns standardised
    # so that none dominates a distance. No two houses are identical across all columns.
    features = np.hstack([numeric, yes_no, furnishing])
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    has_air_conditioning = yes_no[:, YES_NO.index("airconditioning")].astype(int)
    return numeric, features, has_air_conditioning


def build_cases(numeric, features, has_ac):
    """Return (label, vectorised_fn, loop_fn) triples for the loaded data."""
    n_houses = len(numeric)
    price = numeric[:, 0]
    price_list = price.tolist()

    # column statistics of the six numeric columns
    columns = numeric.T.tolist()

    def describe_vectorised():
        stats = describe(numeric, axis=0)
        return np.array([stats[key] for key in ("mean", "median", "std", "min", "max")])

    def describe_loop():
        rows = []
        for column in columns:
            n = len(column)
            mean = sum(column) / n
            ordered = sorted(column)
            middle = n // 2
            median = ordered[middle] if n % 2 else (ordered[middle - 1] + ordered[middle]) / 2
            std = math.sqrt(sum((v - mean) ** 2 for v in column) / n)
            rows.append([mean, median, std, min(column), max(column)])
        return np.array(rows).T

    # histogram of prices
    edges = np.linspace(price.min(), price.max(), 11).tolist()

    def histogram_vectorised():
        return histogram(price, bins=10)[0]

    def histogram_loop():
        counts = [0] * 10
        for p in price_list:
            counts[min(bisect.bisect_right(edges, p) - 1, 9)] += 1
        return np.array(counts)

    # average ranks of prices; many houses share a price, so there are many ties
    def rank_vectorised():
        return rank(price, method="average")

    def rank_loop():
        out = []
        for p in price_list:
            below = sum(1 for q in price_list if q < p)
            equal = sum(1 for q in price_list if q == p)
            out.append(below + (equal + 1) / 2)
        return np.array(out)

    # distances from 100 query houses to all the others
    queries, database = features[:100], features[100:]
    query_rows, database_rows = queries.tolist(), database.tolist()

    def distance_vectorised():
        return euclidean_distance(queries, database)

    def distance_loop():
        return np.array(
            [
                [math.sqrt(sum((x - y) ** 2 for x, y in zip(q, h))) for h in database_rows]
                for q in query_rows
            ]
        )

    # the 5 most similar houses for each query house
    distances = euclidean_distance(queries, database)
    distance_rows = distances.tolist()

    def topk_vectorised():
        return topk_indices(distances, 5, largest=False)

    def topk_loop():
        return np.array(
            [sorted(range(len(row)), key=row.__getitem__)[:5] for row in distance_rows]
        )

    # a baseline rule: "expensive houses have air conditioning"
    y_pred = (price > np.median(price)).astype(int)
    true_list, pred_list = has_ac.tolist(), y_pred.tolist()

    def confusion_vectorised():
        return confusion_matrix(has_ac, y_pred)[0]

    def confusion_loop():
        counts = [[0, 0], [0, 0]]
        for t, p in zip(true_list, pred_list):
            counts[t][p] += 1
        return np.array(counts)

    # area under the ROC curve when price is the score for "has air conditioning"
    positives = price[has_ac == 1].tolist()
    negatives = price[has_ac == 0].tolist()

    def auc_vectorised():
        return auc(*roc_curve(has_ac, price)[:2])

    def auc_loop():
        wins = ties = 0
        for p in positives:
            for n in negatives:
                if p > n:
                    wins += 1
                elif p == n:
                    ties += 1
        return (wins + 0.5 * ties) / (len(positives) * len(negatives))

    return [
        (f"describe, {n_houses} houses x 6 columns", describe_vectorised, describe_loop),
        ("histogram, price in 10 bins", histogram_vectorised, histogram_loop),
        ("rank (average), price with ties", rank_vectorised, rank_loop),
        (f"euclidean_distance, 100 houses vs {len(database)}", distance_vectorised, distance_loop),
        (f"topk_indices, 5 nearest of {len(database)}, per house", topk_vectorised, topk_loop),
        (f"confusion_matrix, {n_houses} houses", confusion_vectorised, confusion_loop),
        ("roc_curve + auc, price vs air conditioning", auc_vectorised, auc_loop),
    ]


def main():
    numeric, features, has_ac = load_housing(DATA_PATH)

    print(f"Data: {DATA_PATH.name}, {len(numeric)} complete houses (rows with a missing value dropped)")
    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    print(f"{platform.system()} {platform.release()} ({platform.machine()}), CPU cores: {os.cpu_count()}")
    print(f"Each time is the mean of {REPEATS} runs, in seconds.\n")

    results = []
    for label, vectorised, loop in build_cases(numeric, features, has_ac):
        result = compare(vectorised, loop, repeats=REPEATS)
        result["name"] = label
        results.append(result)

    print(format_results_table(results))


if __name__ == "__main__":
    main()