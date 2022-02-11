import cudf
import pytest
from pytest import param

# size = 1000000
size = 10000


def make_value(col_size, key_size, mode):
    if mode == "scalar":
        return 42
    if mode == "align_to_col_size":
        return [42] * col_size
    if mode == "align_to_key_size":
        return [42] * key_size


# Benchmark Grid
# key:  slice == 1 (fill or copy_range shortcut),
#       slice != 1 (scatter),
#       column(bool) (boolean_mask_scatter),
#       column(int) (scatter)
# value:    scalar,
#           column (len(val)==len(key)),
#           column (len(val)!=len(key) & len==num_trues)
@pytest.mark.parametrize(
    "key, value_mode",
    [
        param(slice(None, None, 1), "scalar", id="stride-1_slice_scalar"),
        param(slice(None, None, 1), "align_to_key_size", id="stride-1_slice_col"),
        param(slice(None, None, 2), "scalar", id="stride-2_slice_scalar"),
        param(slice(None, None, 2), "align_to_key_size", id="stride-2_slice_col"),
        param([True, False] * (size // 2), "scalar", id="boolean_mask_scalar"),
        param(
            [True, False] * (size // 2),
            "align_to_key_size",
            id="boolean_mask_col_unaligned",
        ),
        param(
            [True, False] * (size // 2),
            "align_to_col_size",
            id="boolean_mask_col_aligned",
        ),
        param(list(range(0, size)), "scalar", id="integer_scatter_map_scalar"),
        param(list(range(0, size)), "align_to_col_size", id="integer_scatter_map_col"),
    ],
)
def test_column_setitem(benchmark, key, value_mode):
    col = cudf.Series([1] * size)._column
    key_size = len(col[key])
    value = make_value(len(col), key_size, value_mode)
    benchmark(col.__setitem__, key, value)
