"""CSV input for NumCompute.

Reads delimited text files into NumPy arrays with explicit handling of
missing values, either in one go (:func:`read_csv`) or as a stream of
fixed-size chunks (:func:`read_csv_chunks`) for files larger than memory. Parsing is delegated to :func:`numpy.genfromtxt` (vectorised
C-level conversion); the header row is parsed manually so column names are
plain Python strings rather than a structured dtype.

Conventions
-----------
* Returned data is always 2-D with shape ``(n_rows, n_cols)``, even for a
  single row or a single column.
* Missing cells (any token in ``missing_values``; blank cells by default)
  become ``filling_value`` (``NaN`` by default).
* Leading/trailing whitespace around every token is stripped.
* Lines that are completely blank are skipped. A missing value in a
  single-column file must therefore be written explicitly (e.g. ``NA``).
* Quoted fields are not supported: the delimiter is never escaped.

Non-numeric / categorical columns (the contract)
------------------------------------------------
NumPy arrays are homogeneous, so a single call returns *either* numbers
*or* strings, never a mix:

* **Numeric mode** (default, ``dtype=float``): every selected cell must be
  a number or a missing token. A categorical cell such as ``'red'`` raises
  ``ValueError``; it is never coerced to NaN behind the caller's back.
* **Text mode** (``dtype=str``): cells are returned as stripped strings
  (``'<U…'`` dtype). Missing tokens become ``''``, the canonical "missing
  category" for ``OneHotEncoder`` / imputers. ``filling_value`` is ignored.
* For a mixed file, select columns with ``usecols`` (names or indices) and
  read the numeric and categorical parts separately::

      X_num, num_cols = read_csv(path, usecols=["age", "income"])
      X_cat, cat_cols = read_csv(path, usecols=["city"], dtype=str)

Errors
------
* Every row must have the same number of fields as the header (or, with
  ``header=False``, as the first data row); otherwise ``ValueError`` names
  the first offending row, e.g. ``"row 42 has 5 fields, expected 3"``.
* A token that is neither numeric nor listed in ``missing_values`` raises
  ``ValueError`` naming its row and column. ``np.genfromtxt`` on its own
  would silently replace it with ``filling_value``; we treat that silent
  corruption as a bug, not a feature.
* An empty file (no non-blank lines) raises ``ValueError``.
"""

from __future__ import annotations

import os
from typing import Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["read_csv", "read_csv_chunks", "DEFAULT_MISSING_VALUES"]

PathLike = Union[str, "os.PathLike[str]"]

#: Tokens treated as missing unless the caller overrides ``missing_values``.
DEFAULT_MISSING_VALUES: Tuple[str, ...] = ("", "NA", "NaN")

# ``utf-8-sig`` transparently drops the byte-order mark that Excel writes.
_ENCODING = "utf-8-sig"

# Tokens float() parses to NaN; they are legitimate data, not parse errors.
_NAN_LITERALS = ("nan", "+nan", "-nan")


# ----------------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------------
def _validate_delimiter(delimiter: str) -> None:
    if not isinstance(delimiter, str):
        raise TypeError(f"delimiter must be a str, got {type(delimiter).__name__}")
    if delimiter == "" or "\n" in delimiter or "\r" in delimiter:
        raise ValueError(f"delimiter must be a non-empty string without newlines, got {delimiter!r}")


def _normalise_missing(missing_values: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    missing = (missing_values,) if isinstance(missing_values, str) else tuple(missing_values)
    if not all(isinstance(tok, str) for tok in missing):
        raise TypeError("missing_values must be a str or a sequence of str")
    # Tokens are compared after stripping, so strip the sentinels too.
    return tuple(tok.strip() for tok in missing)


def _resolve_dtype(dtype) -> Tuple[np.dtype, bool]:
    """Return ``(numpy_dtype, is_text)`` or raise ``TypeError``.

    Integer dtypes are rejected because they cannot represent NaN; cast
    after imputation instead.
    """
    if dtype is str:
        return np.dtype(str), True
    try:
        dt = np.dtype(dtype)
    except TypeError:
        raise TypeError(f"dtype must be a floating dtype or str, got {dtype!r}") from None
    if dt.kind == "U":
        return dt, True
    if dt.kind != "f":
        raise TypeError(
            f"dtype must be a floating dtype or str, got {dt}; integer and other "
            "dtypes cannot hold NaN for missing values (cast after imputing)"
        )
    return dt, False


def _resolve_filling(filling_value, is_text: bool) -> float:
    if is_text:
        return np.nan  # unused in text mode
    try:
        return float(filling_value)
    except (TypeError, ValueError):
        raise ValueError(f"filling_value must be a number, got {filling_value!r}") from None


def _resolve_usecols(usecols, colnames: List[str]) -> np.ndarray:
    """Map ``usecols`` (names and/or ints, negatives allowed) to column indices.

    Returns an int array preserving the caller's order; ``None`` selects all.

    Raises
    ------
    TypeError, ValueError
        For bad entry types, unknown names, out-of-range indices,
        duplicates, or an empty selection.
    """
    n_cols = len(colnames)
    if usecols is None:
        return np.arange(n_cols)
    if isinstance(usecols, (str, int, np.integer)):
        usecols = [usecols]

    idx = []
    for col in usecols:
        if isinstance(col, str):
            if col not in colnames:
                raise ValueError(f"usecols: unknown column {col!r}; available: {colnames}")
            idx.append(colnames.index(col))
        elif isinstance(col, (int, np.integer)) and not isinstance(col, bool):
            if not -n_cols <= col < n_cols:
                raise ValueError(f"usecols: index {col} out of range for {n_cols} columns")
            idx.append(int(col) % n_cols)
        else:
            raise TypeError(f"usecols entries must be str or int, got {type(col).__name__}")

    if not idx:
        raise ValueError("usecols must select at least one column")
    if len(set(idx)) != len(idx):
        raise ValueError(f"usecols selects a column more than once: {list(usecols)}")
    return np.asarray(idx, dtype=np.intp)


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------
def _read_header(fh, source: str, delimiter: str) -> List[str]:
    """Read and split the first line of ``fh`` into stripped column names.

    Raises
    ------
    ValueError
        If the file is empty, the header line is blank, or names repeat.

    Complexity: O(m) time and space for ``m`` columns.
    """
    line = fh.readline()
    if not line.strip():
        if line == "" or not any(rest.strip() for rest in fh):
            raise ValueError(f"CSV file is empty: {source}")
        raise ValueError(f"header row (line 1) is blank in {source}")

    names = [name.strip() for name in line.rstrip("\r\n").split(delimiter)]
    duplicates = sorted({n for n in names if names.count(n) > 1})  # m is small
    if duplicates:
        raise ValueError(f"duplicate column names {duplicates} in header of {source}")
    return names


def _data_lines(fh, start_lineno: int) -> Iterator[Tuple[int, str]]:
    """Yield ``(physical_line_number, line)`` for every non-blank line.

    Line numbers are 1-based and count every physical line in the file
    (header and blank lines included), so they match what an editor shows.

    Complexity: O(1) memory; streams the file one line at a time.
    """
    for lineno, line in enumerate(fh, start=start_lineno + 1):
        line = line.rstrip("\r\n")
        if line.strip():
            yield lineno, line


def _check_field_counts(
    lines: Sequence[str],
    linenos: Sequence[int],
    delimiter: str,
    n_cols: int,
    first_row: int,
    source: str,
) -> None:
    """Raise ``ValueError`` if any line does not have exactly ``n_cols`` fields.

    ``first_row`` is the 1-based data-row number of ``lines[0]`` so that
    messages stay correct when called on a chunk in the middle of a file.

    Complexity: O(k·L) time for ``k`` lines of average length ``L``
    (``str.count`` runs in C); O(k) extra memory.
    """
    counts = np.fromiter((ln.count(delimiter) for ln in lines), dtype=np.intp, count=len(lines)) + 1
    bad = np.flatnonzero(counts != n_cols)
    if bad.size:
        i = int(bad[0])
        extra = f"; {bad.size} malformed rows in this block" if bad.size > 1 else ""
        raise ValueError(
            f"row {first_row + i} has {counts[i]} fields, expected {n_cols} "
            f"(line {linenos[i]} of {source}{extra})"
        )
