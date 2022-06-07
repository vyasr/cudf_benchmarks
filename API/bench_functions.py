"""Benchmarks of free functions that accept cudf objects."""

import pytest
from config import cudf, cupy


# TODO: These cases should be migrated to properly use pytest_cases.
@pytest.mark.parametrize(
    "objs",
    [
        [
            cudf.DataFrame({"a": [1, 2, 3] * 1000000}),
            cudf.DataFrame({"b": [4, 5, 7] * 1000000}),
        ],
        [
            cudf.DataFrame({"a": [1, 2, 3] * 1000000}),
            cudf.DataFrame(
                {"b": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=1000000 * 3, stop=6000000),
            ),
        ],
        [
            cudf.DataFrame({"a": [1, 2, 3] * 1000000, "b": [4, 5, 7] * 1000000}),
            cudf.DataFrame(
                {"c": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=1000000 * 3, stop=6000000),
            ),
        ],
        [
            cudf.DataFrame(
                {"a": [1, 2, 3] * 1000000, "b": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=0, stop=3000000).astype("str"),
            ),
            cudf.DataFrame(
                {"c": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=0, stop=3000000).astype("str"),
            ),
        ],
        [
            cudf.DataFrame(
                {"a": [1, 2, 3] * 1000000, "b": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=0, stop=3000000).astype("str"),
            ),
            cudf.DataFrame(
                {"c": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=3000000, stop=6000000).astype("str"),
            ),
        ],
        [
            cudf.DataFrame(
                {"a": [1, 2, 3] * 1000000, "b": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=0, stop=3000000).astype("str"),
            ),
            cudf.DataFrame(
                {"c": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=3000000, stop=6000000).astype("str"),
            ),
            cudf.DataFrame(
                {"d": [1, 2, 3] * 1000000, "e": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=0, stop=3000000).astype("str"),
            ),
            cudf.DataFrame(
                {"f": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=3000000, stop=6000000).astype("str"),
            ),
            cudf.DataFrame(
                {"g": [1, 2, 3] * 1000000, "h": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=0, stop=3000000).astype("str"),
            ),
            cudf.DataFrame(
                {"i": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=3000000, stop=6000000).astype("str"),
            ),
        ],
        [
            cudf.DataFrame({"a": [1, 2, 3] * 50000}),
            cudf.DataFrame({"b": [4, 5, 7] * 50000}),
            cudf.DataFrame({"c": [1, 2, 3] * 50000}),
            cudf.DataFrame({"d": [4, 5, 7] * 50000}),
            cudf.DataFrame({"e": [1, 2, 3] * 50000}),
            cudf.DataFrame({"f": [4, 5, 7] * 50000}),
            cudf.DataFrame({"g": [1, 2, 3] * 50000}),
            cudf.DataFrame({"h": [4, 5, 7] * 50000}),
            cudf.DataFrame({"i": [1, 2, 3] * 50000}),
            cudf.DataFrame({"j": [4, 5, 7] * 50000}),
        ],
        [
            cudf.DataFrame({"a": [1, 2, 3] * 1000000, "b": [4, 5, 7] * 1000000}),
            cudf.DataFrame(
                {"c": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=1000000 * 3, stop=6000000),
            ),
            cudf.DataFrame({"d": [1, 2, 3] * 1000000, "e": [4, 5, 7] * 1000000}),
            cudf.DataFrame(
                {"f": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=1000000 * 3, stop=6000000),
            ),
            cudf.DataFrame({"g": [1, 2, 3] * 1000000, "h": [4, 5, 7] * 1000000}),
            cudf.DataFrame(
                {"i": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=1000000 * 3, stop=6000000),
            ),
            cudf.DataFrame({"j": [1, 2, 3] * 1000000, "k": [4, 5, 7] * 1000000}),
            cudf.DataFrame(
                {"l": [4, 5, 7] * 1000000},
                index=cudf.RangeIndex(start=1000000 * 3, stop=6000000),
            ),
        ],
    ],
)
@pytest.mark.parametrize(
    "axis",
    [
        1,
    ],
)
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("ignore_index", [True, False])
def bench_concat_axis_1(benchmark, objs, axis, join, ignore_index):
    benchmark(cudf.concat, objs=objs, axis=axis, join=join, ignore_index=ignore_index)


@pytest.mark.parametrize("size", [10_000, 100_000])
@pytest.mark.parametrize("cardinality", [10, 100, 1000])
@pytest.mark.parametrize("dtype", [cupy.bool_, cupy.float64])
def bench_get_dummies_high_cardinality(benchmark, size, cardinality, dtype):
    """This test is mean to test the performance of get_dummies given the
    cardinality of column to encode is high.
    """
    df = cudf.DataFrame(
        {
            "col": cudf.Series(
                cupy.random.randint(low=0, high=cardinality, size=size)
            ).astype("category")
        }
    )
    benchmark(cudf.get_dummies, df, columns=["col"], dtype=dtype)


@pytest.mark.parametrize("prefix", [None, "pre"])
def bench_get_dummies_simple(benchmark, prefix):
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
