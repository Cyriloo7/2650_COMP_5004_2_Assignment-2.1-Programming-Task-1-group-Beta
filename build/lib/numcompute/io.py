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


class _Spec(NamedTuple):
    """Validated parsing options shared by read_csv and read_csv_chunks."""

    source: str
    delimiter: str
    missing: Tuple[str, ...]
    filling_value: float
    dtype: np.dtype
    is_text: bool
    n_fields: int = 0  # fields per row in the file
    col_idx: np.ndarray = np.empty(0, dtype=np.intp)  # selected columns
    out_names: Tuple[str, ...] = ()  # names of the selected columns


def _make_spec(path, delimiter, missing_values, filling_value, dtype) -> _Spec:
    """Validate the arguments common to both readers (fails fast, before I/O)."""
    _validate_delimiter(delimiter)
    missing = _normalise_missing(missing_values)
    dt, is_text = _resolve_dtype(dtype)
    filling = _resolve_filling(filling_value, is_text)
    return _Spec(os.fspath(path), delimiter, missing, filling, dt, is_text)


def _bind_columns(spec: _Spec, colnames: List[str], usecols) -> _Spec:
    """Attach the file's width and the resolved ``usecols`` selection."""
    idx = _resolve_usecols(usecols, colnames)
    return spec._replace(
        n_fields=len(colnames),
        col_idx=idx,
        out_names=tuple(colnames[i] for i in idx),
    )


def _parse_block(
    lines: Sequence[str],
    linenos: Sequence[int],
    first_row: int,
    spec: _Spec,
) -> np.ndarray:
    """Convert validated lines into an array of shape (len(lines), len(spec.col_idx)).

    Numeric mode strategy: parse once with ``genfromtxt`` filling *every*
    unparseable cell with NaN, then inspect only the NaN cells to tell the
    three cases apart:

    1. token is in ``missing_values``  -> replace with ``filling_value``
    2. token is a NaN literal ('nan')  -> keep NaN (it is real data)
    3. anything else ('abc', '1,2x')   -> raise ValueError

    Text mode returns the stripped tokens with missing tokens mapped to ''.

    Complexity: O(k·m) time and space for ``k`` lines of ``m`` fields, plus
    O(r·m) to re-tokenise the ``r`` rows that contain at least one NaN.
    """
    n_out = len(spec.col_idx)
    if len(lines) == 0:
        empty_dtype = "<U1" if spec.is_text and spec.dtype.itemsize == 0 else spec.dtype
        return np.empty((0, n_out), dtype=empty_dtype)

    common = dict(
        delimiter=spec.delimiter,
        usecols=tuple(spec.col_idx.tolist()),
        autostrip=True,
        comments=None,  # '#' is ordinary data, not a comment marker
        ndmin=2,
    )

    if spec.is_text:
        S = np.genfromtxt(lines, dtype=str, **common)
        S = np.where(np.isin(S, spec.missing), "", S)
        return S if spec.dtype.itemsize == 0 else S.astype(spec.dtype)

    X = np.genfromtxt(
        lines, dtype=np.float64, missing_values=spec.missing, filling_values=np.nan, **common
    )

    nan_r, nan_c = np.nonzero(np.isnan(X))
    if nan_r.size:
        # Re-tokenise only the rows that contain NaN (usually a small subset).
        rows, inverse = np.unique(nan_r, return_inverse=True)
        tokens = np.char.strip(
            np.array([lines[r].split(spec.delimiter) for r in rows], dtype=str)
        )
        cells = tokens[inverse, spec.col_idx[nan_c]]

        is_missing = np.isin(cells, spec.missing)
        is_nan_literal = np.isin(np.char.lower(cells), _NAN_LITERALS)
        bad = np.flatnonzero(~(is_missing | is_nan_literal))
        if bad.size:
            b = int(bad[0])
            r, c = int(nan_r[b]), int(nan_c[b])
            extra = f"; {bad.size} non-numeric cells in this block" if bad.size > 1 else ""
            raise ValueError(
                f"row {first_row + r}, column {spec.out_names[c]!r}: cannot convert "
                f"{str(cells[b])!r} to float (line {linenos[r]} of {spec.source}{extra}). "
                "Add the token to missing_values if it means 'missing', or read "
                "categorical columns with dtype=str (see usecols)."
            )

        if not np.isnan(spec.filling_value):
            X[nan_r[is_missing], nan_c[is_missing]] = spec.filling_value

    return X.astype(spec.dtype, copy=False)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def read_csv(
    path: PathLike,
    delimiter: str = ",",
    header: bool = True,
    missing_values: Sequence[str] = DEFAULT_MISSING_VALUES,
    filling_value: float = np.nan,
    *,
    dtype=float,
    usecols: Optional[Sequence[Union[str, int]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Read an entire delimited text file into a 2-D NumPy array.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the CSV file (UTF-8; a leading byte-order mark is ignored).
    delimiter : str, default ','
        Field separator, e.g. ``','`` or ``'\\t'``.
    header : bool, default True
        If True the first line holds column names. If False, names
        ``'col0', 'col1', ...`` are generated.
    missing_values : str or sequence of str, default ('', 'NA', 'NaN')
        Tokens (compared after whitespace stripping) treated as missing.
    filling_value : float, default nan
        Value written into missing cells (numeric mode only).
    dtype : floating dtype or str, keyword-only, default float
        ``float``/``np.float32``/... for numeric mode, ``str`` for text mode
        (see the module docstring for the categorical-column contract).
    usecols : sequence of str or int, keyword-only, optional
        Columns to return, by name or index (negative indices allowed), in
        the order given. Default: all columns.

    Returns
    -------
    X : numpy.ndarray, shape (n_rows, n_selected_cols)
        Parsed data with the requested ``dtype``. A header-only file gives
        shape ``(0, n_selected_cols)``.
    colnames : list of str, length n_selected_cols
        Names of the returned columns, in the same order as ``X``'s columns.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If ``delimiter``, ``missing_values``, ``dtype`` or ``usecols`` has
        the wrong type (integer dtypes are rejected: they cannot hold NaN).
    ValueError
        If the file is empty, the header is blank or has duplicate names,
        a row has the wrong number of fields (``"row 42 has 5 fields,
        expected 3"``), a numeric-mode cell is neither a number nor a
        missing token, ``usecols`` is invalid, or ``filling_value`` is not
        a number.

    Complexity
    ----------
    Time O(n·m) and memory O(n·m) for ``n`` rows and ``m`` columns: every
    line is held in memory and parsed by a single vectorised
    ``genfromtxt`` call. Use :func:`read_csv_chunks` for O(chunksize·m)
    memory.

    Examples
    --------
    >>> X, cols = read_csv("data.csv")                          # doctest: +SKIP
    >>> X.shape, cols                                            # doctest: +SKIP
    ((100, 3), ['age', 'height', 'weight'])
    >>> C, _ = read_csv("people.csv", usecols=["city"], dtype=str)  # doctest: +SKIP
    """
    spec = _make_spec(path, delimiter, missing_values, filling_value, dtype)

    with open(path, encoding=_ENCODING) as fh:
        colnames: Optional[List[str]] = _read_header(fh, spec.source, delimiter) if header else None
        numbered = list(_data_lines(fh, start_lineno=1 if header else 0))

    linenos = [n for n, _ in numbered]
    lines = [line for _, line in numbered]

    if colnames is None:
        if not lines:
            raise ValueError(f"CSV file is empty: {spec.source}")
        colnames = [f"col{j}" for j in range(lines[0].count(delimiter) + 1)]

    spec = _bind_columns(spec, colnames, usecols)
    _check_field_counts(lines, linenos, delimiter, spec.n_fields, first_row=1, source=spec.source)
    X = _parse_block(lines, linenos, first_row=1, spec=spec)
    return X, list(spec.out_names)

def read_csv_chunks(
    path: PathLike,
    chunksize: int,
    delimiter: str = ",",
    header: bool = True,
    missing_values: Sequence[str] = DEFAULT_MISSING_VALUES,
    filling_value: float = np.nan,
    *,
    dtype=float,
    usecols: Optional[Sequence[Union[str, int]]] = None,
) -> Iterator[Tuple[np.ndarray, List[str]]]:
    """Stream a delimited text file as successive 2-D NumPy arrays.

    The file is read line by line; at most ``chunksize`` data lines are
    buffered before being parsed with the same vectorised path as
    :func:`read_csv`, so concatenating all chunks gives exactly
    ``read_csv(path, ...)[0]``.

    Arguments are validated eagerly (errors raise at call time); the file
    itself is opened lazily on the first ``next()`` and closed when the
    generator is exhausted, closed, or garbage-collected.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the CSV file (UTF-8; a leading byte-order mark is ignored).
    chunksize : int
        Maximum number of data rows per chunk (>= 1). Every chunk has
        exactly ``chunksize`` rows except possibly the last.
    delimiter, header, missing_values, filling_value, dtype, usecols
        Same meaning as in :func:`read_csv`.

    Yields
    ------
    X_chunk : numpy.ndarray, shape (k, n_selected_cols), 1 <= k <= chunksize
        Parsed rows of the chunk with the requested ``dtype``.
    colnames : list of str, length n_selected_cols
        Column names (a fresh list each time, same contents every chunk).

    Raises
    ------
    TypeError
        At call time, if ``chunksize`` is not an int, or for the argument
        type errors listed in :func:`read_csv`.
    ValueError
        At call time, if ``chunksize < 1`` or an argument value is invalid.
        During iteration, for an empty file, a malformed row or a
        non-numeric cell; row numbers in messages count from the start of
        the file, not the chunk, and earlier chunks have already been
        yielded.
    FileNotFoundError
        At call time, if ``path`` does not exist.

    Complexity
    ----------
    Time O(n·m) in total for ``n`` rows and ``m`` columns; memory
    O(chunksize·m) regardless of file size. A header-only file yields no
    chunks.

    Examples
    --------
    >>> total = 0.0                                              # doctest: +SKIP
    >>> for X, cols in read_csv_chunks("big.csv", chunksize=10_000):
    ...     total += np.nansum(X[:, 0])
    """
    if isinstance(chunksize, bool) or not isinstance(chunksize, (int, np.integer)):
        raise TypeError(f"chunksize must be an int, got {type(chunksize).__name__}")
    if chunksize < 1:
        raise ValueError(f"chunksize must be >= 1, got {chunksize}")
    spec = _make_spec(path, delimiter, missing_values, filling_value, dtype)
    if not os.path.isfile(spec.source):
        raise FileNotFoundError(f"No such file: {spec.source!r}")
    return _iter_chunks(int(chunksize), header, usecols, spec)


def _iter_chunks(
    chunksize: int, header: bool, usecols, spec: _Spec
) -> Iterator[Tuple[np.ndarray, List[str]]]:
    """Generator behind :func:`read_csv_chunks` (arguments already validated)."""
    with open(spec.source, encoding=_ENCODING) as fh:
        names: Optional[List[str]] = None
        if header:
            spec = _bind_columns(spec, _read_header(fh, spec.source, spec.delimiter), usecols)
            names = list(spec.out_names)

        lines: List[str] = []
        linenos: List[int] = []
        first_row = 1  # data-row number of lines[0], counted from file start

        for lineno, line in _data_lines(fh, start_lineno=1 if header else 0):
            if names is None:  # header=False: width comes from the first data row
                width = line.count(spec.delimiter) + 1
                spec = _bind_columns(spec, [f"col{j}" for j in range(width)], usecols)
                names = list(spec.out_names)
            lines.append(line)
            linenos.append(lineno)
            if len(lines) == chunksize:
                yield _parse_chunk(lines, linenos, first_row, spec), list(names)
                first_row += len(lines)
                lines, linenos = [], []

        if lines:
            yield _parse_chunk(lines, linenos, first_row, spec), list(names)
        elif names is None:
            raise ValueError(f"CSV file is empty: {spec.source}")


def _parse_chunk(lines: List[str], linenos: List[int], first_row: int, spec: _Spec) -> np.ndarray:
    """Validate field counts for one chunk, then parse it."""
    _check_field_counts(lines, linenos, spec.delimiter, spec.n_fields, first_row, spec.source)
    return _parse_block(lines, linenos, first_row, spec)
