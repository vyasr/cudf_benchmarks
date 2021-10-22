import cudf
from cudf.testing.dataset_generator import rand_dataframe
import pytest

from itertools import product

n = [100, 10000]

@pytest.fixture(params=n)
def df(request):
    return cudf.DataFrame.from_arrow(rand_dataframe(
        [{
            "dtype": "int64",
            "null_frequency": 0.4,
            "cardinality": 1000
        }],
        rows=request.param,
        seed=0,
        use_threads=False
    ))

@pytest.mark.parametrize(
    "dropnan", [True, False]
)
def test_dropna_single_column(benchmark, df, dropnan):
    benchmark(lambda: df['0']._column.dropna(drop_nan=dropnan))