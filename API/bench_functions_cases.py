import pytest_cases
from config import NUM_ROWS, cudf


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_1(nr):
    return [
        cudf.DataFrame({"a": [1, 2, 3] * nr}),
        cudf.DataFrame({"b": [4, 5, 7] * nr}),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_2(nr):
    return [
        cudf.DataFrame({"a": [1, 2, 3] * nr}),
        cudf.DataFrame(
            {"b": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_3(nr):
    return [
        cudf.DataFrame({"a": [1, 2, 3] * nr, "b": [4, 5, 7] * nr}),
        cudf.DataFrame(
            {"c": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_4(nr):
    return [
        cudf.DataFrame(
            {"a": [1, 2, 3] * nr, "b": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"c": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_5(nr):
    return [
        cudf.DataFrame(
            {"a": [1, 2, 3] * nr, "b": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"c": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3).astype("str"),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_6(nr):
    return [
        cudf.DataFrame(
            {"a": [1, 2, 3] * nr, "b": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"c": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"d": [1, 2, 3] * nr, "e": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"f": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"g": [1, 2, 3] * nr, "h": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"i": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3).astype("str"),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_7(nr):
    # To avoid any edge case bugs, always use at least 10 rows per DataFrame.
    nr_actual = max(10, nr // 20)
    return [
        cudf.DataFrame({"a": [1, 2, 3] * nr_actual}),
        cudf.DataFrame({"b": [4, 5, 7] * nr_actual}),
        cudf.DataFrame({"c": [1, 2, 3] * nr_actual}),
        cudf.DataFrame({"d": [4, 5, 7] * nr_actual}),
        cudf.DataFrame({"e": [1, 2, 3] * nr_actual}),
        cudf.DataFrame({"f": [4, 5, 7] * nr_actual}),
        cudf.DataFrame({"g": [1, 2, 3] * nr_actual}),
        cudf.DataFrame({"h": [4, 5, 7] * nr_actual}),
        cudf.DataFrame({"i": [1, 2, 3] * nr_actual}),
        cudf.DataFrame({"j": [4, 5, 7] * nr_actual}),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_8(nr):
    return [
        cudf.DataFrame({"a": [1, 2, 3] * nr, "b": [4, 5, 7] * nr}),
        cudf.DataFrame(
            {"c": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
        cudf.DataFrame({"d": [1, 2, 3] * nr, "e": [4, 5, 7] * nr}),
        cudf.DataFrame(
            {"f": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
        cudf.DataFrame({"g": [1, 2, 3] * nr, "h": [4, 5, 7] * nr}),
        cudf.DataFrame(
            {"i": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
        cudf.DataFrame({"j": [1, 2, 3] * nr, "k": [4, 5, 7] * nr}),
        cudf.DataFrame(
            {"l": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
    ]
