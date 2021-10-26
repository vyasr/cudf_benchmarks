import cudf
import cupy

import pytest

from utils import make_col

n = [100, 10000]

@pytest.fixture(params=n)
def col(request):
    return make_col(request.param)

@pytest.mark.parametrize(
    "dropnan", [True, False]
)
def test_dropna_single_column(benchmark, col, dropnan):
    benchmark(col.dropna, drop_nan=dropnan)