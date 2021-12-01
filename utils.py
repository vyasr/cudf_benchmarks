import cudf
import cupy as cp

# reproducibility
cp.random.seed(seed=0)


def make_frame(ncols, nkey_cols, nrows, low=0, high=100):
    nval_cols = ncols - nkey_cols
    key_columns = {
        f"key{i}": cp.random.randint(low, high, nrows) for i in range(nkey_cols)
    }
    val_columns = {f"val{i}": cp.random.rand(nrows) for i in range(nval_cols)}
    return cudf.DataFrame({**key_columns, **val_columns})


def make_col(nrows, has_nulls=True):
    c = cudf.core.column.as_column(cp.random.randn(nrows))
    if has_nulls:
        # The choice of null placement is arbitrary.
        c[::2] = None
    return c


def make_gather_map(len_gather_map, len_column, how):
    """Create a gather map based on "how" you'd like to gather from input.
    - sequence: gather the first `len_gather_map` rows, the first thread
                collects the first element
    - reverse:  gather the last `len_gather_map` rows, the first thread
                collects the last element
    - random:   create a pseudorandom gather map
    """
    if how == "sequence":
        return cudf.Series(cp.arange(0, len_gather_map))._column
    elif how == "reverse":
        return cudf.Series(
            cp.arange(len_column - 1, len_column - len_gather_map - 1, -1)
        )._column
    elif how == "random":
        return cudf.Series(cp.random.randint(0, len_column, len_gather_map))._column
