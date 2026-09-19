"""
Unit tests for numcompute.benchmarking.

Run from the project root with:
    pytest tests/test_benchmarking.py -v
"""

import time

import numpy as np
import pytest

from numcompute.benchmarking import benchmark, compare, format_results_table


# ---------------------------------------------------------------------------
# benchmark()
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_returns_expected_keys(self):
        result = benchmark(lambda: 42, repeats=5, warmup=1)
        assert set(result.keys()) == {"mean", "std", "min", "max", "times", "result"}

    def test_times_array_has_one_entry_per_repeat(self):
        result = benchmark(lambda: None, repeats=6, warmup=0)
        assert result["times"].shape == (6,)

    def test_min_mean_max_are_consistent(self):
        result = benchmark(lambda: sum(range(1000)), repeats=8, warmup=1)
        assert result["min"] <= result["mean"] <= result["max"]

    def test_captures_the_function_return_value(self):
        result = benchmark(lambda: "the answer", repeats=3, warmup=0)
        assert result["result"] == "the answer"

    def test_a_slower_function_reports_a_higher_mean(self):
        def fast():
            return sum(range(100))

        def slow():
            time.sleep(0.005)
            return sum(range(100))

        fast_result = benchmark(fast, repeats=5, warmup=1)
        slow_result = benchmark(slow, repeats=5, warmup=1)
        assert slow_result["mean"] > fast_result["mean"]

    def test_warmup_calls_are_not_included_in_the_timed_count(self):
        calls = {"n": 0}

        def counting_fn():
            calls["n"] += 1

        benchmark(counting_fn, repeats=4, warmup=3)
        assert calls["n"] == 7  # 3 warmup + 4 timed, all real calls

    def test_args_and_kwargs_are_passed_through(self):
        def add(a, b, scale=1):
            return (a + b) * scale

        result = benchmark(add, 2, 3, repeats=3, warmup=0, scale=10)
        assert result["result"] == 50

    @pytest.mark.parametrize("bad_repeats", [0, -1])
    def test_non_positive_repeats_raises(self, bad_repeats):
        with pytest.raises(ValueError):
            benchmark(lambda: None, repeats=bad_repeats)

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError):
            benchmark(lambda: None, warmup=-1)


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

class TestCompare:
    def test_agreeing_functions_report_a_speedup(self):
        def vectorized(x):
            return x * 2

        def loop(x):
            return np.array([v * 2 for v in x])

        result = compare(vectorized, loop, np.arange(2000.0), repeats=5)
        assert result["name"] == "vectorized"
        assert result["speedup"] > 0
        np.testing.assert_allclose(result["vectorized"]["result"], result["loop"]["result"])

    def test_a_genuinely_faster_vectorized_function_shows_speedup_above_one(self):
        def topk_vectorized(values, k):
            idx = np.argpartition(values, -k)[-k:]
            return np.sort(values[idx])[::-1]

        def topk_loop(values, k):
            return np.array(sorted(values, reverse=True)[:k])

        values = np.random.default_rng(1).normal(size=2000)
        result = compare(topk_vectorized, topk_loop, values, 5, repeats=5)
        assert result["speedup"] > 1.0

    def test_mismatched_outputs_raise_instead_of_reporting_a_speedup(self):
        def correct_fn(x):
            return x * 2

        def buggy_fn(x):
            return x * 3  # deliberately wrong

        with pytest.raises(ValueError):
            compare(correct_fn, buggy_fn, np.array([1.0, 2.0, 3.0]), repeats=3)

    def test_result_dict_has_vectorized_and_loop_sub_results(self):
        result = compare(lambda x: x + 1, lambda x: x + 1, 5, repeats=2)
        assert "mean" in result["vectorized"]
        assert "mean" in result["loop"]


# ---------------------------------------------------------------------------
# format_results_table()
# ---------------------------------------------------------------------------

class TestFormatResultsTable:
    def _make_result(self, name="my_fn", speedup=10.0):
        return {
            "name": name,
            "vectorized": {"mean": 0.001},
            "loop": {"mean": 0.01},
            "speedup": speedup,
        }

    def test_single_dict_produces_header_plus_one_row(self):
        table = format_results_table(self._make_result())
        lines = table.split("\n")
        assert len(lines) == 3  # header, separator, one data row
        assert "my_fn" in lines[2]

    def test_list_of_results_produces_one_row_each(self):
        results = [self._make_result("fn_a"), self._make_result("fn_b")]
        table = format_results_table(results)
        lines = table.split("\n")
        assert len(lines) == 4  # header, separator, two data rows
        assert "fn_a" in lines[2]
        assert "fn_b" in lines[3]

    def test_speedup_is_shown_with_an_x_suffix(self):
        table = format_results_table(self._make_result(speedup=51.746))
        assert "51.75x" in table

    def test_output_is_a_valid_markdown_table_header(self):
        table = format_results_table(self._make_result())
        assert table.startswith("| Function |")
        assert "|---|" in table

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            format_results_table([])