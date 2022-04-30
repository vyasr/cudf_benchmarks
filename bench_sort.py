import cudf
import cupy as cp
import pytest


@pytest.mark.parametrize("cls", [cudf.Series, cudf.Index])
@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
def test_series_index_argsort(benchmark, cls, N):
    obj = cls(cp.random.rand(N))
    benchmark(obj.argsort)


@pytest.mark.parametrize("cls", [cudf.Series, cudf.Index])
@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
def test_series_index_sort_values(benchmark, cls, N):
    obj = cls(cp.random.rand(N))
    benchmark(obj.sort_values)


@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
@pytest.mark.parametrize("n", [10])
def test_series_nsmallest(benchmark, N, n):
    ser = cudf.Series(cp.random.rand(N))
    benchmark(ser.nsmallest, n)


@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
@pytest.mark.parametrize("ncol", [5, 10])
# @pytest.mark.parametrize("ncol_sort", [1, 2, 3])
@pytest.mark.parametrize("ncol_sort", [1])
def test_dataframe_argsort(benchmark, N, ncol, ncol_sort):
    df = cudf.DataFrame({i: cp.random.rand(N) for i in range(ncol)})
    benchmark(df.argsort)


@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
@pytest.mark.parametrize("ncol", [5, 10])
# @pytest.mark.parametrize("ncol_sort", [1, 2, 3])
@pytest.mark.parametrize("ncol_sort", [1])
def test_dataframe_sort_values(benchmark, N, ncol, ncol_sort):
    df = cudf.DataFrame({i: cp.random.rand(N) for i in range(ncol)})
    benchmark(df.sort_values, [i for i in range(ncol_sort)])


@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
@pytest.mark.parametrize("ncol", [5, 10])
# @pytest.mark.parametrize("ncol_sort", [1, 2, 3])
@pytest.mark.parametrize("ncol_sort", [1])
@pytest.mark.parametrize("n", [10])
def test_dataframe_nsmallest(benchmark, N, ncol, ncol_sort, n):
    df = cudf.DataFrame({i: cp.random.rand(N) for i in range(ncol)})
    benchmark(df.nsmallest, n, [i for i in range(ncol_sort)])


@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
def test_rangeindex_argsort(benchmark, N):
    obj = cudf.RangeIndex(range(N))
    benchmark(obj.argsort)