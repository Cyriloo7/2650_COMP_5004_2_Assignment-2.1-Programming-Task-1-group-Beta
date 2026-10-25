"""Timing utilities for comparing vectorized and loop-based implementations.

Functions:
    benchmark: time repeated calls to one function.
    compare: time two implementations and report the speedup.
    format_results_table: render compare() results as a Markdown table.
"""

import time

import numpy as np


def benchmark(func, *args, repeats=10, warmup=1, **kwargs):
    """Time repeated calls to `func(*args, **kwargs)`.

    Untimed warmup calls run first so one-off start-up costs do not skew
    the results.

    Parameters
    ----------
    func : callable
        Function to time.
    *args, **kwargs
        Forwarded to `func` on every call.
    repeats : int, default=10
        Number of timed calls.
    warmup : int, default=1
        Number of untimed calls made first.

    Returns
    -------
    dict
        'mean', 'std', 'min', 'max' : float
            Timing statistics in seconds.
        'times' : ndarray, shape (repeats,)
            Duration of each timed call in seconds.
        'result'
            Return value of the last timed call.

    Raises
    ------
    ValueError
        If `repeats` < 1 or `warmup` < 0.

    Complexity
    ----------
    O(warmup + repeats) calls to `func`.
    """
    if repeats < 1:
        raise ValueError(f"benchmark(): repeats must be a positive integer, got {repeats}")
    if warmup < 0:
        raise ValueError(f"benchmark(): warmup must not be negative, got {warmup}")

    for _ in range(warmup):
        func(*args, **kwargs)

    times = np.empty(repeats, dtype=float)
    result = None
    for i in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times[i] = end - start

    return {
        "mean": float(times.mean()),
        "std": float(times.std()),
        "min": float(times.min()),
        "max": float(times.max()),
        "times": times,
        "result": result,
    }


def compare(vectorized_fn, loop_fn, *args, repeats=10, **kwargs):
    """Time two implementations of one computation and report the speedup.

    Raises instead of reporting a speedup when the two results differ in
    shape or value. Both functions receive the same argument objects on
    every call, so pass copies if either one mutates its input.

    Parameters
    ----------
    vectorized_fn, loop_fn : callable
        Implementations to compare; each is called as `fn(*args, **kwargs)`.
    *args, **kwargs
        Forwarded to both functions.
    repeats : int, default=10
        Timed calls per function.

    Returns
    -------
    dict
        'name' : str
            `vectorized_fn.__name__`.
        'vectorized', 'loop' : dict
            `benchmark()` result for each function.
        'speedup' : float
            Loop mean time divided by vectorized mean time.

    Raises
    ------
    ValueError
        If the results differ in shape, or in value under
        `np.allclose(..., equal_nan=True)`.

    Complexity
    ----------
    Two `benchmark()` runs.
    """
    vec_result = benchmark(vectorized_fn, *args, repeats=repeats, **kwargs)
    loop_result = benchmark(loop_fn, *args, repeats=repeats, **kwargs)

    vec_out = np.asarray(vec_result["result"])
    loop_out = np.asarray(loop_result["result"])

    # np.allclose broadcasts, so check shapes explicitly first.
    if vec_out.shape != loop_out.shape:
        raise ValueError(
            f"compare(): '{vectorized_fn.__name__}' returned a result of shape "
            f"{vec_out.shape} but '{loop_fn.__name__}' returned shape "
            f"{loop_out.shape} -- refusing to report a speedup for two "
            "implementations whose outputs have different shapes"
        )

    # Matching NaNs count as agreement.
    if not np.allclose(vec_out, loop_out, equal_nan=True):
        raise ValueError(
            f"compare(): '{vectorized_fn.__name__}' and '{loop_fn.__name__}' "
            "produced different results on the same input -- refusing to "
            "report a speedup for two implementations that don't agree"
        )

    speedup = loop_result["mean"] / vec_result["mean"]

    return {
        "name": vectorized_fn.__name__,
        "vectorized": vec_result,
        "loop": loop_result,
        "speedup": speedup,
    }


def format_results_table(results):
    """Render `compare()` results as a Markdown table.

    Parameters
    ----------
    results : dict or list[dict]
        A `compare()` result, or a list of them (one row each).

    Returns
    -------
    str
        Table with columns Function, Vectorized (s), Loop (s), Speedup.

    Raises
    ------
    ValueError
        If `results` is empty.

    Complexity
    ----------
    O(k) for k results.
    """
    if isinstance(results, dict):
        results = [results]
    if len(results) == 0:
        raise ValueError("format_results_table(): results is empty -- nothing to format")

    lines = [
        "| Function | Vectorized (s) | Loop (s) | Speedup |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['vectorized']['mean']:.4g} | "
            f"{r['loop']['mean']:.4g} | {r['speedup']:.2f}x |"
        )
    return "\n".join(lines)