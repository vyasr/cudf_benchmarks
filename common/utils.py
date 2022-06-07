from numbers import Real

from config import cudf
from config import cupy as cp


def make_gather_map(len_gather_map: Real, len_column: Real, how: str):
    """Create a gather map based on "how" you'd like to gather from input.
    - sequence: gather the first `len_gather_map` rows, the first thread
                collects the first element
    - reverse:  gather the last `len_gather_map` rows, the first thread
                collects the last element
    - random:   create a pseudorandom gather map

    `len_gather_map`, `len_column` gets rounded to integer.
    """
    len_gather_map = round(len_gather_map)
    len_column = round(len_column)

    rstate = cp.random.RandomState(seed=0)
    if how == "sequence":
        return cudf.Series(cp.arange(0, len_gather_map))._column
    elif how == "reverse":
        return cudf.Series(
            cp.arange(len_column - 1, len_column - len_gather_map - 1, -1)
        )._column
    elif how == "random":
        return cudf.Series(rstate.randint(0, len_column, len_gather_map))._column


def make_boolean_mask_column(size):
    rstate = cp.random.RandomState(seed=0)
    return cudf.core.column.as_column(rstate.randint(0, 2, size).astype(bool))
