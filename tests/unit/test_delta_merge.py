import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal

import polars as pl
import pytest
from dbt_common.exceptions import DbtRuntimeError
from deltalake import DeltaTable, write_deltalake
from polars.testing import assert_frame_equal

from dbt.adapters.polars.catalogs.formats.delta import DeltaFormat


@pytest.fixture
def table(tmp_path):
    path = str(tmp_path / "t")
    pl.DataFrame({"id": [1, 2], "value": ["a", "b"]}).write_delta(path)
    return path


def merge(path, data, keys=("id",)):
    DeltaFormat.merge(path, data, list(keys), None, None, False)


def last_operation(path):
    return list(DeltaTable(path).history(1))[0]["operation"]


def added_stats(path):
    """The stats of the files the last commit added."""
    dt = DeltaTable(path)
    log = os.path.join(path, "_delta_log", f"{dt.version():020d}.json")
    with open(log) as f:
        actions = [json.loads(line) for line in f]
    return [json.loads(a["add"]["stats"]) for a in actions if "add" in a]


def parquet_files(path):
    return {
        os.path.join(root, name)
        for root, _, names in os.walk(path)
        for name in names
        if name.endswith(".parquet")
    }


def test_all_new_rows_are_appended(table):
    merge(table, pl.LazyFrame({"id": [3, 4], "value": ["c", "d"]}))

    assert_frame_equal(
        pl.read_delta(table).sort("id"),
        pl.DataFrame({"id": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]}),
    )
    assert last_operation(table) == "WRITE"
    [stats] = added_stats(table)
    assert stats["numRecords"] == 2
    assert stats["minValues"] == {"id": 3, "value": "c"}
    assert stats["maxValues"] == {"id": 4, "value": "d"}
    assert stats["nullCount"] == {"id": 0, "value": 0}


def test_stats_match_polars_delta_sink(tmp_path):
    """The append's stats are the ones Polars' own Delta sink writes, but for the
    max of a sub-millisecond timestamp."""
    rows = pl.DataFrame(
        {
            "id": [5, 6, 7],
            "s": ["x", None, "zz"],
            "f": [1.5, float("inf"), -2.0],
            "d": [date(2026, 1, 1), date(2026, 3, 1), None],
            # Whole milliseconds: a sub-millisecond max is rounded up here, where
            # Polars' sink truncates it (test_timestamp_stats_keep_every_row_readable).
            "ts": [datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=timezone.utc)] * 3,
            "ntz": [datetime(2026, 1, 2, 3, 4, 5)] * 3,
            "dec": [Decimal("1.50"), Decimal("-0.05"), None],
            "b": [True, False, True],
            "st": [{"a": 1, "c": "p"}, {"a": 2, "c": "q"}, None],
        },
        schema_overrides={"dec": pl.Decimal(10, 2), "ts": pl.Datetime("us", "UTC")},
    )
    seed = rows.head(1).with_columns(pl.col("id") - 100)
    ours, polars_sink = str(tmp_path / "ours"), str(tmp_path / "polars")
    for path in (ours, polars_sink):
        seed.write_delta(path)
    rows.lazy().sink_delta(polars_sink, mode="append")

    merge(ours, rows.lazy())

    assert added_stats(ours) == added_stats(polars_sink)


def test_a_matching_key_is_merged(table):
    merge(table, pl.LazyFrame({"id": [2, 3], "value": ["B", "c"]}))

    assert_frame_equal(
        pl.read_delta(table).sort("id"),
        pl.DataFrame({"id": [1, 2, 3], "value": ["a", "B", "c"]}),
    )
    assert last_operation(table) == "MERGE"


def two_key_table(tmp_path, rows):
    path = str(tmp_path / "t2")
    schema = {"file": pl.String, "id": pl.Int64}
    pl.DataFrame(rows, schema=schema, orient="row").write_delta(path)
    return path


def test_a_disjoint_key_column_appends(tmp_path, monkeypatch):
    """Target rows inside the source's key range, but no source `file` in the target:
    the probe on that one column proves the keys new, and the semi-join never runs."""
    path = two_key_table(tmp_path, [("b", 1), ("b", 2)])
    semi_joins = []
    join = pl.LazyFrame.join
    monkeypatch.setattr(
        pl.LazyFrame,
        "join",
        lambda self, *a, **k: semi_joins.append(k.get("how")) or join(self, *a, **k),
    )

    source = pl.LazyFrame({"file": ["a", "c", "a"], "id": [1, 2, 3]})
    merge(path, source, ("file", "id"))

    assert semi_joins == []
    assert last_operation(path) == "WRITE"
    assert pl.read_delta(path).height == 5


def test_overlapping_columns_without_a_matching_key_append(tmp_path):
    """Every key column has values on both sides but no key matches: the semi-join
    finds none, and the rows are appended."""
    path = two_key_table(tmp_path, [("a", 1), ("b", 2)])

    merge(path, pl.LazyFrame({"file": ["a", "b"], "id": [2, 1]}), ("file", "id"))

    assert last_operation(path) == "WRITE"
    assert pl.read_delta(path).sort("file", "id").rows() == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]


def test_empty_source_commits_nothing(table):
    version = DeltaTable(table).version()

    merge(table, pl.LazyFrame({"id": [1], "value": ["A"]}).filter(pl.lit(False)))

    assert DeltaTable(table).version() == version


def test_timestamp_stats_keep_every_row_readable(tmp_path):
    path = str(tmp_path / "ts")
    ts = pl.Datetime("us", "UTC")
    first = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pl.DataFrame(
        {"at": [first], "v": [1]}, schema={"at": ts, "v": pl.Int64}
    ).write_delta(path)
    late = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=timezone.utc)

    merge(
        path,
        pl.LazyFrame({"at": [late], "v": [2]}, schema={"at": ts, "v": pl.Int64}),
        keys=("at",),
    )

    [stats] = added_stats(path)
    assert stats["minValues"]["at"] == "2026-01-02T03:04:05.678Z"
    # A max rounded up: Polars' reader does not widen a truncated one.
    assert stats["maxValues"]["at"] == "2026-01-02T03:04:05.679Z"
    found = pl.scan_delta(path).filter(pl.col("at") == late).collect()
    assert found["v"].to_list() == [2]


def test_a_partitioned_table_is_merged(tmp_path):
    path = str(tmp_path / "p")
    pl.DataFrame({"id": [1], "part": ["x"]}).write_delta(
        path, delta_write_options={"partition_by": ["part"]}
    )

    merge(path, pl.LazyFrame({"id": [2], "part": ["y"]}))

    assert pl.read_delta(path).sort("id")["id"].to_list() == [1, 2]
    assert last_operation(path) == "MERGE"


def test_append_refused_after_a_concurrent_commit(table):
    dt = DeltaTable(table)
    write_deltalake(table, pl.DataFrame({"id": [3], "value": ["c"]}), mode="append")
    files = parquet_files(table)

    with pytest.raises(DbtRuntimeError, match="changed since version"):
        DeltaFormat._append_if_all_new(
            dt, table, pl.LazyFrame({"id": [3], "value": ["c"]}), ["id"], None
        )

    assert pl.read_delta(table)["id"].sort().to_list() == [1, 2, 3]
    assert parquet_files(table) == files
