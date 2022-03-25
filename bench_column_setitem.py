from config import cudf
import pytest


@pytest.fixture(params=[10000, 1000000])
def size(request):
    return request.param


@pytest.fixture
def col(size):
    return cudf.core.column.arange(size)


def make_key(mode, col):
    size = len(col)
    if mode == "stride-1-slice":
        return slice(None, None, 1)
    elif mode == "stride-2-slice":
        return slice(None, None, 2)
    elif mode == "boolean_mask":
        return [True, False] * (size // 2)
    elif mode == "int_column":
        return list(range(size))


@pytest.fixture(
    params=[
        ("stride-1-slice", "scalar"),
        ("stride-2-slice", "scalar"),
        ("boolean_mask", "scalar"),
        ("int_column", "scalar"),
        ("stride-1-slice", "align_to_key_size"),
        ("stride-2-slice", "align_to_key_size"),
        ("boolean_mask", "align_to_col_size"),
        ("int_column", "align_to_col_size"),
    ],
    ids=lambda p: f"{p[0]}-{p[1]}",
)
def key_value(request, col):
    key_mode, value_mode = request.param
    if value_mode == "scalar":
        return make_key(key_mode, col), 42
    if value_mode == "align_to_col_size":
        return make_key(key_mode, col), [42] * len(col)
    if value_mode == "align_to_key_size":
        key = make_key(key_mode, col)
        materialized_key_size = len(col[key])
        return key, [42] * materialized_key_size


# Benchmark Grid
# key:  slice == 1  (fill or copy_range shortcut),
#       slice != 1  (scatter),
#       column(bool)    (boolean_mask_scatter),
#       column(int) (scatter)
# value:    scalar,
#           column (len(val) == len(key)),
#           column (len(val) != len(key) & len == num_true)


def test_column_setitem(benchmark, col, key_value):
    benchmark(col.__setitem__, *key_value)
