import cudf
import cupy as cp
import numpy as np

import pytest


@pytest.mark.parametrize(
    "size", [10_000, 100_000, 1_000_000]
)
@pytest.mark.parametrize(
    "cardinality", [10, 100, 1000]
)
@pytest.mark.parametrize(
    "dtype", [np.bool_, np.float64]
)
def test_get_dummies(benchmark, size, cardinality, dtype):
    df = cudf.DataFrame({'col': cudf.Series(
        cp.random.randint(low=0, high=cardinality, size=size)
    ).astype("category")})
    benchmark(lambda: cudf.get_dummies(df, columns=['col'], dtype=dtype))

@pytest.mark.parametrize(
    "prefix", [None, 'pre']
)
def test_get_dummies_simple(benchmark, prefix):
    df = cudf.DataFrame(
            {
                'col1': list(range(10)),
                'col2': list('abcdefghij'),
                'col3': cudf.Series(list(range(100, 110)), dtype='category')
            }
        )
    benchmark(lambda: cudf.get_dummies(df, columns=['col1', 'col2', 'col3'], prefix=prefix))

# The below was used to do detail profiling in nvtx
# import nvtx

# df = cudf.DataFrame({'col': cudf.Series(
#     cp.random.randint(low=0, high=1000, size=1_000_000)
# ).astype("category")})

# s = cudf.Series([1])
# s2 = cudf.Series([2])

# with nvtx.annotate("benchmark_all"):
#     res = cudf.concat([s, s2])
