from config import cudf


def concat_case_1():
    return [
        cudf.DataFrame({"a": [1, 2, 3] * 1000000}),
        cudf.DataFrame({"b": [4, 5, 7] * 1000000}),
    ]


def concat_case_2():
    return [
        cudf.DataFrame({"a": [1, 2, 3] * 1000000}),
        cudf.DataFrame(
            {"b": [4, 5, 7] * 1000000},
            index=cudf.RangeIndex(start=1000000 * 3, stop=6000000),
        ),
    ]


def concat_case_3():
    return [
        cudf.DataFrame({"a": [1, 2, 3] * 1000000, "b": [4, 5, 7] * 1000000}),
        cudf.DataFrame(
            {"c": [4, 5, 7] * 1000000},
            index=cudf.RangeIndex(start=1000000 * 3, stop=6000000),
        ),
    ]


def concat_case_4():
    return [
        cudf.DataFrame(
            {"a": [1, 2, 3] * 1000000, "b": [4, 5, 7] * 1000000},
            index=cudf.RangeIndex(start=0, stop=3000000).astype("str"),
        ),
        cudf.DataFrame(
            {"c": [4, 5, 7] * 1000000},
            index=cudf.RangeIndex(start=0, stop=3000000).astype("str"),
        ),
    ]


def concat_case_5():
    return [
        cudf.DataFrame(
            {"a": [1, 2, 3] * 1000000, "b": [4, 5, 7] * 1000000},
            index=cudf.RangeIndex(start=0, stop=3000000).astype("str"),
        ),
        cudf.DataFrame(
            {"c": [4, 5, 7] * 1000000},
            index=cudf.RangeIndex(start=3000000, stop=6000000).astype("str"),
        ),
    ]


def concat_case_6():
    return [
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
    ]


def concat_case_7():
    return [
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
    ]


def concat_case_8():
    return [
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
    ]
