"""
benchmarking.py -- A small timing harness for NumCompute.

Provides:
    benchmark(func, *args, repeats=10, warmup=1, **kwargs)
        -> timing stats for one function
    compare(vectorized_fn, loop_fn, *args, repeats=10, **kwargs)
        -> timing stats for both, plus the speedup, after checking they
           actually produce the same result
    format_results_table(results)
        -> a Markdown table of one or more `compare()` results
"""

import time

import numpy as np


def benchmark(func, *args, repeats=10, warmup=1, **kwargs):
    """
    Time `func(*args, **kwargs)`, called `repeats` times, and summarise
    the timings.

    A few "warmup" calls are made first and their timings thrown away --
    the very first call to a function is often slower than the rest
    (memory allocation, disk/OS caching, etc.), and including it would
    make the reported numbers less representative of steady-state speed.

    Parameters
    ----------
    func : callable
        The function to time. Called as `func(*args, **kwargs)`.
    *args, **kwargs
        Passed straight through to `func` on every call.
    repeats : int, default=10
        Number of timed calls to make (after warmup).
    warmup : int, default=1
        Number of untimed calls to make first.

    Returns
    -------
    dict
        Keys:
          'mean', 'std', 'min', 'max' : float, in seconds, over the
              `repeats` timed calls.
          'times' : ndarray, shape (repeats,)
              Every individual timed duration, in seconds.
          'result' : whatever `func` returned on its last timed call --
              useful for `compare()` to check correctness without
              calling `func` a second time.

    Raises
    ------
    ValueError
        If `repeats` is not a positive integer, or `warmup` is negative.

    Complexity
    ----------
    O(warmup + repeats) calls to `func`; the harness itself adds
    negligible overhead on top of whatever `func` costs.
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
    """
    Benchmark `vectorized_fn` and `loop_fn` on identical input, and
    report how much faster the vectorized version is.

    Before reporting a speedup, checks that the two functions actually
    produced the same result (via `np.allclose`) -- a "fast" version
    that's also wrong is a real risk when two different implementations
    of the same computation are being compared, so this refuses to
    report a speedup number for functions that disagree.

    Note: both functions are called with the exact same argument
    objects on every timed run. If either one mutates its input in
    place, pass fresh copies in (e.g. via a small wrapper lambda),
    since the same array would otherwise be reused, and possibly
    already modified, on later calls.

    Parameters
    ----------
    vectorized_fn : callable
    loop_fn : callable
        Both called as `fn(*args, **kwargs)`. Assumed to compute the
        same thing in two different ways.
    *args, **kwargs
        Passed to both functions, unchanged.
    repeats : int, default=10
        Passed through to `benchmark` for both functions.

    Returns
    -------
    dict
        Keys:
          'name' : str -- `vectorized_fn.__name__`, for labelling.
          'vectorized' : dict -- full `benchmark()` result for vectorized_fn.
          'loop' : dict -- full `benchmark()` result for loop_fn.
          'speedup' : float -- loop_mean / vectorized_mean. A value of
              8.0 means the vectorized version was 8x faster.

    Raises
    ------
    ValueError
        If the two functions' results don't agree (checked with
        `np.allclose`).

    Complexity
    ----------
    Same as calling `benchmark()` twice.
    """
    vec_result = benchmark(vectorized_fn, *args, repeats=repeats, **kwargs)
    loop_result = benchmark(loop_fn, *args, repeats=repeats, **kwargs)

    if not np.allclose(vec_result["result"], loop_result["result"]):
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
    """
    Format one or more `compare()` results as a Markdown table.

    Parameters
    ----------
    results : dict or list[dict]
        One `compare()` result, or a list of them (one row per result).

    Returns
    -------
    str
        A Markdown table with columns: Function, Vectorized (s),
        Loop (s), Speedup.

    Raises
    ------
    ValueError
        If `results` is an empty list.

    Complexity
    ----------
    O(k), where k is the number of results being formatted.
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