import pytest
from config import cudf


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
def test_concat_axis_1(benchmark, objs, axis, join, ignore_index):
    benchmark(cudf.concat, objs=objs, axis=axis, join=join, ignore_index=ignore_index)
