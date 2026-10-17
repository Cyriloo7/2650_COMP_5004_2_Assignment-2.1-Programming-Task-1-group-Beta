import numpy as np
import pytest

from numcompute.io import read_csv, read_csv_chunks

@pytest.fixture
def write_csv(tmp_path):
    """Return a helper that writes ``text`` to a temp file and returns its path."""

    def _write(text, name="data.csv", encoding="utf-8"):
        path = tmp_path / name
        path.write_text(text, encoding=encoding)
        return path

    return _write


# ----------------------------------------------------------------------------
# read_csv: basic parsing and missing values
# ----------------------------------------------------------------------------
def test_read_csv_basic_shape_dtype_and_names(write_csv):
    path = write_csv("a,b,c\n1,2,3\n4,5,6\n")
    X, cols = read_csv(path)
    assert cols == ["a", "b", "c"]
    assert X.dtype == np.float64
    np.testing.assert_array_equal(X, [[1, 2, 3], [4, 5, 6]])


def test_blank_cells_become_nan(write_csv):
    path = write_csv("a,b,c\n1,,3\n,5,\n")
    X, _ = read_csv(path)
    expected = np.array([[1, np.nan, 3], [np.nan, 5, np.nan]])
    np.testing.assert_array_equal(np.isnan(X), np.isnan(expected))
    np.testing.assert_array_equal(X[~np.isnan(X)], [1, 3, 5])


def test_na_and_nan_tokens_are_missing(write_csv):
    path = write_csv("a,b\nNA,1\n2,NaN\n 3 , NA \n")
    X, _ = read_csv(path)
    np.testing.assert_array_equal(np.isnan(X), [[True, False], [False, True], [False, True]])
    assert X[2, 0] == 3.0  # surrounding whitespace is stripped


def test_custom_filling_value(write_csv):
    path = write_csv("a,b\n1,\nNA,4\n")
    X, _ = read_csv(path, filling_value=-1.0)
    np.testing.assert_array_equal(X, [[1, -1], [-1, 4]])


def test_custom_missing_values_tokens(write_csv):
    path = write_csv("a,b\n?,1\n2,?\n")
    X, _ = read_csv(path, missing_values=("?",))
    np.testing.assert_array_equal(np.isnan(X), [[True, False], [False, True]])


def test_tab_delimiter(write_csv):
    path = write_csv("x\ty\n1.5\t\n\t-2\n", name="data.tsv")
    X, cols = read_csv(path, delimiter="\t")
    assert cols == ["x", "y"]
    assert X[0, 0] == 1.5 and np.isnan(X[0, 1])
    assert np.isnan(X[1, 0]) and X[1, 1] == -2


def test_header_false_generates_column_names(write_csv):
    path = write_csv("1,2\n3,4\n")
    X, cols = read_csv(path, header=False)
    assert cols == ["col0", "col1"]
    assert X.shape == (2, 2)


def test_single_row_and_single_column_stay_2d(write_csv):
    X, _ = read_csv(write_csv("a,b,c\n1,2,3\n", name="row.csv"))
    assert X.shape == (1, 3)
    X, _ = read_csv(write_csv("a\n1\n2\n3\n", name="col.csv"))
    assert X.shape == (3, 1)


def test_header_only_file_returns_zero_rows(write_csv):
    X, cols = read_csv(write_csv("a,b,c\n"))
    assert X.shape == (0, 3)
    assert cols == ["a", "b", "c"]


def test_blank_lines_crlf_and_bom_are_handled(write_csv):
    path = write_csv("﻿a,b\r\n1,2\r\n\r\n3,4\r\n")
    X, cols = read_csv(path)
    assert cols == ["a", "b"]  # BOM stripped from first name
    np.testing.assert_array_equal(X, [[1, 2], [3, 4]])


# ----------------------------------------------------------------------------
# read_csv: malformed input and error messages
# ----------------------------------------------------------------------------
def test_malformed_row_too_many_fields(write_csv):
    path = write_csv("a,b,c\n1,2,3\n4,5,6,7,8\n")
    with pytest.raises(ValueError, match=r"row 2 has 5 fields, expected 3 \(line 3"):
        read_csv(path)


def test_malformed_row_too_few_fields_reports_first_and_count(write_csv):
    path = write_csv("a,b,c\n1,2\n1,2,3\n4\n")
    with pytest.raises(ValueError, match=r"row 1 has 2 fields, expected 3.*2 malformed rows"):
        read_csv(path)


def test_malformed_row_without_header_uses_first_row_width(write_csv):
    path = write_csv("1,2\n3,4\n5,6,7\n")
    with pytest.raises(ValueError, match="row 3 has 3 fields, expected 2"):
        read_csv(path, header=False)


def test_line_number_accounts_for_blank_lines(write_csv):
    path = write_csv("a,b\n1,2\n\n\n3\n")
    with pytest.raises(ValueError, match=r"row 2 has 1 fields, expected 2 \(line 5"):
        read_csv(path)


@pytest.mark.parametrize("text", ["", "\n", "   \n\n"])
@pytest.mark.parametrize("header", [True, False])
def test_empty_file_raises(write_csv, text, header):
    with pytest.raises(ValueError, match="CSV file is empty"):
        read_csv(write_csv(text), header=header)


def test_blank_header_with_data_raises(write_csv):
    with pytest.raises(ValueError, match="header row"):
        read_csv(write_csv("\n1,2\n"))


def test_duplicate_column_names_raise(write_csv):
    with pytest.raises(ValueError, match=r"duplicate column names \['a'\]"):
        read_csv(write_csv("a,b,a\n1,2,3\n"))


def test_non_numeric_token_raises_instead_of_silent_fill(write_csv):
    # np.genfromtxt alone would turn 'abc' into the filling value without warning.
    path = write_csv("a,b\n1,2\n3,abc\n")
    with pytest.raises(ValueError, match=r"row 2, column 'b': cannot convert 'abc'"):
        read_csv(path)
    with pytest.raises(ValueError, match="cannot convert 'abc'"):
        read_csv(path, filling_value=0.0)


def test_nan_literal_is_data_not_missing(write_csv):
    path = write_csv("a,b\nnan,\n-NAN,1\n")
    X, _ = read_csv(path, filling_value=0.0)
    assert np.isnan(X[0, 0]) and np.isnan(X[1, 0])  # literals kept as NaN
    assert X[0, 1] == 0.0  # blank cell filled


def test_inf_and_scientific_notation_parse(write_csv):
    X, _ = read_csv(write_csv("a,b\ninf,1e-3\n-inf,2.5E2\n"))
    np.testing.assert_array_equal(X, [[np.inf, 1e-3], [-np.inf, 250.0]])


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_csv(tmp_path / "nope.csv")


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"delimiter": ""}, ValueError),
        ({"delimiter": "\n"}, ValueError),
        ({"delimiter": 1}, TypeError),
        ({"missing_values": [1, 2]}, TypeError),
        ({"filling_value": "zero"}, ValueError),
    ],
)
def test_invalid_arguments_raise(write_csv, kwargs, exc):
    with pytest.raises(exc):
        read_csv(write_csv("a\n1\n"), **kwargs)


# ----------------------------------------------------------------------------
# read_csv: categorical columns (dtype=str) and usecols
# ----------------------------------------------------------------------------
MIXED = "age,city,income\n25,Adelaide,50000\n,NA,61000\n40, Perth ,\n"


def test_categorical_column_in_numeric_mode_raises_with_hint(write_csv):
    with pytest.raises(ValueError, match=r"column 'city'.*dtype=str"):
        read_csv(write_csv(MIXED))


def test_dtype_str_returns_stripped_tokens_with_missing_as_empty(write_csv):
    S, cols = read_csv(write_csv(MIXED), dtype=str)
    assert cols == ["age", "city", "income"]
    assert S.dtype.kind == "U"
    np.testing.assert_array_equal(
        S, [["25", "Adelaide", "50000"], ["", "", "61000"], ["40", "Perth", ""]]
    )


def test_mixed_file_split_with_usecols(write_csv):
    path = write_csv(MIXED)
    X, num_cols = read_csv(path, usecols=["income", "age"])  # caller's order kept
    C, cat_cols = read_csv(path, usecols=["city"], dtype=str)
    assert num_cols == ["income", "age"] and cat_cols == ["city"]
    assert X.shape == (3, 2) and C.shape == (3, 1)
    np.testing.assert_array_equal(X[:, 0][:2], [50000, 61000])
    assert np.isnan(X[1, 1]) and np.isnan(X[2, 0])
    np.testing.assert_array_equal(C.ravel(), ["Adelaide", "", "Perth"])


def test_usecols_by_index_and_negative_index(write_csv):
    path = write_csv("a,b,c\n1,2,3\n")
    X, cols = read_csv(path, usecols=[-1, 0])
    assert cols == ["c", "a"]
    np.testing.assert_array_equal(X, [[3, 1]])


def test_usecols_header_only_keeps_selected_width(write_csv):
    X, cols = read_csv(write_csv("a,b,c\n"), usecols=["b"])
    assert X.shape == (0, 1) and cols == ["b"]


def test_float32_dtype(write_csv):
    X, _ = read_csv(write_csv("a,b\n1,\n"), dtype=np.float32)
    assert X.dtype == np.float32 and np.isnan(X[0, 1])


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"dtype": int}, TypeError),  # ints cannot hold NaN
        ({"dtype": "not-a-dtype"}, TypeError),
        ({"usecols": ["missing"]}, ValueError),
        ({"usecols": [5]}, ValueError),
        ({"usecols": [0, 0]}, ValueError),
        ({"usecols": []}, ValueError),
        ({"usecols": [1.5]}, TypeError),
    ],
)
def test_invalid_dtype_and_usecols_raise(write_csv, kwargs, exc):
    with pytest.raises(exc):
        read_csv(write_csv("a,b\n1,2\n"), **kwargs)




# ----------------------------------------------------------------------------
# read_csv_chunks: streaming reader
# ----------------------------------------------------------------------------
SAMPLE = "a,b,c\n1,,3\n4,5,NA\n7,8,9\n,11,12\n13,14,15\n"


@pytest.mark.parametrize("chunksize", [1, 2, 3, 5, 100])
def test_chunks_concatenate_to_full_read(write_csv, chunksize):
    path = write_csv(SAMPLE)
    full, cols = read_csv(path)
    chunks = list(read_csv_chunks(path, chunksize))
    assert all(names == cols for _, names in chunks)
    np.testing.assert_array_equal(np.vstack([X for X, _ in chunks]), full)  # NaN == NaN here


def test_chunk_sizes_last_chunk_smaller(write_csv):
    sizes = [X.shape for X, _ in read_csv_chunks(write_csv(SAMPLE), 2)]
    assert sizes == [(2, 3), (2, 3), (1, 3)]


def test_chunks_returns_a_generator(write_csv):
    import inspect

    gen = read_csv_chunks(write_csv(SAMPLE), 2)
    assert inspect.isgenerator(gen)
    gen.close()


def test_chunks_stream_without_reading_ahead(write_csv):
    # The malformed row is far beyond the first chunk: the first chunk must
    # still be yielded, proving the file is not parsed eagerly.
    rows = "".join(f"{i},{i}\n" for i in range(1, 1000)) + "1,2,3\n"
    gen = read_csv_chunks(write_csv("x,y\n" + rows), chunksize=10)
    X, _ = next(gen)
    np.testing.assert_array_equal(X[:, 0], np.arange(1, 11))
    with pytest.raises(ValueError, match=r"row 1000 has 3 fields, expected 2 \(line 1001"):
        for _ in gen:
            pass


def test_chunks_report_global_row_for_non_numeric_cell(write_csv):
    gen = read_csv_chunks(write_csv("a\n1\n2\n3\nx\n"), chunksize=2)
    next(gen)
    with pytest.raises(ValueError, match=r"row 4, column 'a': cannot convert 'x'"):
        next(gen)


@pytest.mark.parametrize("header", [True, False])
def test_chunks_empty_file_raises_on_first_next(write_csv, header):
    gen = read_csv_chunks(write_csv(""), 3, header=header)  # no error at call time
    with pytest.raises(ValueError, match="CSV file is empty"):
        next(gen)


def test_chunks_header_only_yields_nothing(write_csv):
    assert list(read_csv_chunks(write_csv("a,b\n"), 4)) == []


def test_chunks_header_false_dtype_str_and_usecols(write_csv):
    path = write_csv("1,red\n2,\n3,blue\n")
    chunks = list(read_csv_chunks(path, 2, header=False, usecols=[1], dtype=str))
    assert [names for _, names in chunks] == [["col1"], ["col1"]]
    np.testing.assert_array_equal(np.vstack([C for C, _ in chunks]).ravel(), ["red", "", "blue"])


@pytest.mark.parametrize(
    "chunksize, exc",
    [(0, ValueError), (-5, ValueError), (2.5, TypeError), (True, TypeError), ("3", TypeError)],
)
def test_invalid_chunksize_raises_eagerly(write_csv, chunksize, exc):
    with pytest.raises(exc):
        read_csv_chunks(write_csv(SAMPLE), chunksize)  # no next() needed


def test_chunks_missing_file_raises_eagerly(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_csv_chunks(tmp_path / "nope.csv", 10)
