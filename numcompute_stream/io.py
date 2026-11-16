"""Load CSV data and split it into chunks to simulate a data stream.

The file is read with the Assignment 2.1 reader ``numcompute.io.read_csv``
(reused, not changed). It handles the delimiter, the header, missing cells and
rows with the wrong number of fields, with clear error messages. Every column
is first read as text to find out which columns hold numbers:

* numeric columns are then read again by the 2.1 reader as floats, with NaN
  for missing values;
* other columns stay as text, with "" for missing values.
"""

import numpy as np

from numcompute.io import DEFAULT_MISSING_VALUES, read_csv  # Assignment 2.1 (reused)
from numcompute.utils import _is_integer, batch_iterator    # Assignment 2.1 (reused)


def _is_numeric(column):
    """True if every non-missing cell of a text column holds a number."""
    try:
        column[column != ""].astype(float)
    except ValueError:
        return False
    return True


def _target_index(target_column, header):
    """Position of the label column, from its name or its index (negative counts from the end).

    An index outside the header is an error rather than being wrapped around,
    so a typo can never silently pick another column as the label.
    """
    if isinstance(target_column, str):
        if target_column not in header:
            raise ValueError(f"target_column {target_column!r} not found; the columns are {header}.")
        return header.index(target_column)
    if not _is_integer(target_column) or not -len(header) <= target_column < len(header):
        raise ValueError(f"target_column must be a column name or an integer from {-len(header)} to "
                         f"{len(header) - 1} (the file has {len(header)} columns), got {target_column!r}.")
    return target_column % len(header)


def load_csv(path, delimiter=",", target_column=-1, missing_values=DEFAULT_MISSING_VALUES):
    """Load a CSV file that has a header row.

    Parameters
    ----------
    path : str or Path
        File to read.
    delimiter : str, default ","
        Field separator, e.g. "\\t" for tab-separated files.
    target_column : int or str, default -1
        Index or name of the column holding the labels (-1 = last column).
    missing_values : sequence of str
        Cell values that mean "missing" (default: the 2.1 reader's "", "NA", "NaN").

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Float array if every feature is numeric, otherwise an object array
        mixing floats and strings.
    y : ndarray of shape (n_samples,)
        Integer labels if the labels are whole numbers, otherwise strings.
        Rows with a missing label are dropped.
    feature_names : list of str
        Header names of the columns of X (the label column removed).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file has no data rows, a malformed row (the 2.1 reader names
        the line), or ``target_column`` is not a column name or an index
        inside the header.
    """
    table, header = read_csv(path, delimiter=delimiter, missing_values=missing_values, dtype=str)
    if len(table) == 0:
        raise ValueError(f"{path} has a header row but no data rows.")
    target = _target_index(target_column, header)

    columns = [table[:, j].astype(object) for j in range(len(header))]  # text columns as Python strings
    numeric = [j for j in range(len(header)) if _is_numeric(table[:, j])]
    if numeric:
        numbers, _ = read_csv(path, delimiter=delimiter, missing_values=missing_values, usecols=numeric)
        for k, j in enumerate(numeric):
            columns[j] = numbers[:, k]  # parsed by the 2.1 reader: floats, NaN when missing
    y = columns.pop(target)
    feature_names = header[:target] + header[target + 1:]

    has_label = ~np.isnan(y) if y.dtype == float else (y != "")
    y = y[has_label]
    if y.dtype == float and np.all(y == np.round(y)):
        y = y.astype(int)
    elif y.dtype == object:
        y = y.astype(str)

    if all(column.dtype == float for column in columns):
        X = np.column_stack(columns)
    else:
        X = np.empty((len(table), len(columns)), dtype=object)
        for j, column in enumerate(columns):
            X[:, j] = column
    return X[has_label], y, feature_names


def iter_chunks(X, y, chunk_size):
    """Yield (X_chunk, y_chunk) pairs of ``chunk_size`` rows, keeping the original order.

    The last chunk may be smaller. Keeping the order matters for streams:
    shuffling would hide any change (drift) in the data over time. The chunks
    come from the Assignment 2.1 helper ``numcompute.utils.batch_iterator``,
    which also checks ``chunk_size``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    chunk_size : int
        Rows per chunk (>= 1).

    Returns
    -------
    iterator of (ndarray of shape (chunk, n_features), ndarray of shape (chunk,))
        Views of consecutive rows; every row appears exactly once.

    Raises
    ------
    ValueError
        If X and y have different lengths (at once), or ``chunk_size`` is not
        a positive integer (when iteration starts).
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels.")
    return zip(batch_iterator(X, chunk_size), batch_iterator(y, chunk_size))
