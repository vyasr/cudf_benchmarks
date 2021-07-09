import cudf
import operator
import cupy as cp
import pytest


@pytest.mark.parametrize(
    'op', [operator.add, operator.mul, operator.__and__, operator.eq],
)
@pytest.mark.parametrize(
    'cls', [cudf.Series, cudf.Index, cudf.DataFrame]
)
@pytest.mark.parametrize(
    'N', [1_000, 100_000, 10_000_000]
)
@pytest.mark.parametrize(
    'has_nulls', [False, True]
)
def test_binops(benchmark, op, cls, N, has_nulls):
    # Need int array for __and__
    ser = cudf.Series(cp.random.rand(N), dtype=int)
    if has_nulls:
        ser[::2] = None
    frame = cls({'a': ser}) if cls is cudf.DataFrame else cls(ser)
    benchmark(lambda: op(frame, frame))
