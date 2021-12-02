import cudf
import cupy as cp
import pytest


@pytest.mark.parametrize("size", [10_000, 100_000])
@pytest.mark.parametrize("cardinality", [10, 100, 1000])
@pytest.mark.parametrize("dtype", [cp.bool_, cp.float64])
def test_get_dummies_high_cardinality(benchmark, size, cardinality, dtype):
    """This test is mean to test the performance of get_dummies given the
    cardinality of column to encode is high.
    """
    df = cudf.DataFrame(
        {
            "col": cudf.Series(
                cp.random.randint(low=0, high=cardinality, size=size)
            ).astype("category")
        }
    )
    benchmark(cudf.get_dummies, df, columns=["col"], dtype=dtype)


@pytest.mark.parametrize("prefix", [None, "pre"])
def test_get_dummies_simple(benchmark, prefix):
    """This test provides a small input to get_dummies to test the efficiency
    of the API itself.
    """
    df = cudf.DataFrame(
        {
            "col1": list(range(10)),
            "col2": list("abcdefghij"),
            "col3": cudf.Series(list(range(100, 110)), dtype="category"),
        }
    )
    benchmark(cudf.get_dummies, df, columns=["col1", "col2", "col3"], prefix=prefix)
