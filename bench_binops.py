import cudf
import operator
import cupy as cp
import pytest


@pytest.mark.parametrize(
    'op', [operator.add, operator.mul, operator.__and__, operator.eq],
)
@pytest.mark.parametrize(
    'frame_type', [cudf.Series, cudf.Index]
)
@pytest.mark.parametrize(
    'N', [1_000, 100_000, 10_000_000]
)
def test_binops(benchmark, op, frame_type, N):
    frame = frame_type(cp.random.rand(N))
    benchmark(lambda: op(frame, frame))
