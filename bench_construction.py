import cudf
import cupy as cp
import pytest


@pytest.mark.parametrize(
    'frame_type', [cudf.Series, cudf.Index, cudf.DataFrame]
)
@pytest.mark.parametrize(
    'N', [1_000, 100_000, 10_000_000]
)
def test_binops(benchmark, frame_type, N):
    data = cp.random.rand(N)
    if frame_type is cudf.DataFrame:
        data = {None: data}
    benchmark(lambda: frame_type(data))
