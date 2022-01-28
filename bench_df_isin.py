import pytest

import cudf

from utils import make_frame


@pytest.mark.parametrize("nrows", [1_000, 100_000])
@pytest.mark.parametrize(
    "values",
    [
        range(1000),
        {f"key{i}": range(1000) for i in range(10)},
        cudf.DataFrame({f"key{i}": range(1000) for i in range(10)}),
        cudf.Series(range(1000)),
    ]
)
def test_take_multiple_column(benchmark, nrows, values):
    df = make_frame(10, 10, nrows)
    benchmark(df.isin, values)
