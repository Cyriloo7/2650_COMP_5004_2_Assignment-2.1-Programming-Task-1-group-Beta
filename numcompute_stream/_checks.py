"""Input check shared by the numcompute_stream modules (private).

Every setting that must be a whole number (``bins``, ``window``, ``buffer_size``, ...) is checked here, so the
rule and the error message are the same everywhere.
"""

from numcompute.utils import _is_integer  # Assignment 2.1 (reused)


def check_int_at_least(name, value, smallest=1, allow_none=False):
    """Return ``value`` if it is an integer >= ``smallest`` (or None when ``allow_none``); otherwise raise ValueError.

    Booleans and floats such as 2.5 are rejected. The message names the setting, e.g.
    ``"buffer_size must be an integer >= 1, got 0."``
    """
    if allow_none and value is None:
        return value
    if not _is_integer(value) or value < smallest:
        raise ValueError(f"{name} must be {'None or ' if allow_none else ''}an integer >= {smallest}, got {value!r}.")
    return value
