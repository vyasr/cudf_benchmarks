import cudf
import pytest


@pytest.fixture(params=[10000, 1000000])
def size(request):
    return request.param


@pytest.fixture
def col(size):
    return cudf.core.column.arange(size)


@pytest.fixture(
    params=["stride-1-slice", "stride-2-slice", "boolean_mask", "int_column"]
)
def key(request, col):
    size = len(col)
    if request.param == "stride-1-slice":
        return slice(None, None, 1)
    elif request.param == "stride-2-slice":
        return slice(None, None, 2)
    elif request.param == "boolean_mask":
        return [True, False] * (size // 2)
    elif request.param == "int_column":
        return list(range(size))


@pytest.fixture(params=["scalar", "align_to_key_size", "align_to_col_size"])
def value(request, col, key):
    mode = request.param
    if mode == "scalar":
        return 42
    if mode == "align_to_col_size":
        if isinstance(key, list):
            return [42] * len(col)
        else:
            pytest.skip(
                "Scattering to slice of column requires value size same as key"
                "size. In stride-1 case, it's benchmarked by `align_to_key_size`."
            )
    if mode == "align_to_key_size":
        if isinstance(key, list) and isinstance(key[0], int):
            pytest.skip(
                "Integer scatter map is the same length of column, "
                "which was already benchmarked by `align_to_col_size`."
            )
        key_size = len(col[key])
        return [42] * key_size


# Benchmark Grid
# key:  slice == 1  (fill or copy_range shortcut),
#       slice != 1  (scatter),
#       column(bool)    (boolean_mask_scatter),
#       column(int) (scatter)
# value:    scalar,
#           column (len(val)==len(key)),
#           column (len(val)!=len(key) & len==num_trues)


def test_column_setitem(benchmark, col, key, value):
    benchmark(col.__setitem__, key, value)
