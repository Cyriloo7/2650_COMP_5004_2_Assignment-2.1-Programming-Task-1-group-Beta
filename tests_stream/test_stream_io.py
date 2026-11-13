"""Tests for io.py: loading CSV files (with the 2.1 reader) and splitting data into chunks."""

from pathlib import Path

import numpy as np
import pytest

from numcompute.io import DEFAULT_MISSING_VALUES
from numcompute_stream.io import iter_chunks, load_csv


def write(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_load_csv_converts_numbers_text_and_missing_values(tmp_path):
    X, y, names = load_csv(write(tmp_path, "temp,site,label\n1.5,north,yes\nNA,south,no\n3.0,,yes\n"))
    assert names == ["temp", "site"] and X.shape == (3, 2) and X.dtype == object
    assert np.isnan(X[1, 0]) and X[2, 1] == ""
    assert y.tolist() == ["yes", "no", "yes"]


def test_numeric_file_gives_a_float_matrix_and_integer_labels(tmp_path):
    X, y, _ = load_csv(write(tmp_path, "a,b,y\n1,2,0\n3,NA,1\n"))
    assert X.dtype == float and np.isnan(X[1, 1])
    assert y.dtype.kind == "i" and y.tolist() == [0, 1]
    X, _, _ = load_csv(write(tmp_path, "a,b,y\n1,?,0\n", "other.csv"), missing_values=["?"])
    assert np.isnan(X[0, 1])  # a custom missing-value token


def test_custom_delimiter_and_target_column_by_name(tmp_path):
    X, y, names = load_csv(write(tmp_path, "y\ta\tb\n1\t2\t3\n0\t4\t5\n", "data.tsv"),
                           delimiter="\t", target_column="y")
    assert names == ["a", "b"] and y.tolist() == [1, 0]
    np.testing.assert_array_equal(X, [[2, 3], [4, 5]])


def test_rows_without_a_label_are_dropped(tmp_path):
    X, y, _ = load_csv(write(tmp_path, "a,y\n1,0\n2,\n3,1\n"))
    assert len(X) == len(y) == 2


def test_load_csv_errors(tmp_path):
    with pytest.raises(ValueError):
        load_csv(write(tmp_path, "a,b\n"))  # header only
    with pytest.raises(ValueError, match="line"):
        load_csv(write(tmp_path, "a,b\n1,2\n3\n", "bad.csv"))  # the 2.1 reader names the bad line
    with pytest.raises(ValueError, match="not found"):
        load_csv(write(tmp_path, "a,b\n1,2\n", "ok.csv"), target_column="label")
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_loads_the_housing_dataset():
    path = Path(__file__).resolve().parents[1] / "Dataset" / "Housing.csv"  # the Assignment 2.1 dataset
    X, y, names = load_csv(path, target_column="airconditioning", missing_values=(*DEFAULT_MISSING_VALUES, "NAN"))
    assert X.shape == (544, 12) and "airconditioning" not in names  # one row has no label
    assert set(y) == {"yes", "no"} and names[0] == "price"
    assert np.isnan(X[:, names.index("area")].astype(float)).sum() == 1      # numeric cell missing -> NaN
    assert set(X[:, names.index("basement")]) == {"yes", "no", ""}           # "NAN" and blank cells -> ""
    assert load_csv(path, target_column="airconditioning")[0][:, names.index("basement")].tolist().count("NAN") == 1


def test_iter_chunks_keeps_the_order_and_sizes():
    X, y = np.arange(10).reshape(-1, 1), np.arange(10)
    chunks = list(iter_chunks(X, y, chunk_size=4))
    assert [len(y_chunk) for _, y_chunk in chunks] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate([y_chunk for _, y_chunk in chunks]), y)
    with pytest.raises(ValueError):
        list(iter_chunks(X, y, chunk_size=0))
    with pytest.raises(ValueError):
        iter_chunks(X, y[:5], chunk_size=2)


def test_text_only_file_and_non_integer_labels(tmp_path):
    X, y, names = load_csv(write(tmp_path, "colour,size,label\nred,big,a\nblue,small,b\n"))
    assert X.dtype == object and X.tolist() == [["red", "big"], ["blue", "small"]]  # no numeric column at all
    assert y.tolist() == ["a", "b"] and names == ["colour", "size"]
    _, y, _ = load_csv(write(tmp_path, "a,target\n1,1.5\n2,2.0\n", "reg.csv"))
    assert y.dtype == float and y.tolist() == [1.5, 2.0]  # labels that are not whole numbers stay floats
