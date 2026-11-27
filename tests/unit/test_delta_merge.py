import polars as pl
import pytest
from dbt_common.exceptions import DbtRuntimeError
from deltalake import DeltaTable, write_deltalake
from polars.testing import assert_frame_equal

from dbt.adapters.polars.catalogs.formats.delta import DeltaFormat

TARGET = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})


@pytest.fixture
def table(tmp_path):
    path = str(tmp_path / "t")
    TARGET.write_delta(path)
    return path


def merge(path, source):
    DeltaFormat.merge(path, source, ["id"], None, None, False)


def read(path):
    return pl.read_delta(path).sort("id")


def test_unmatched_source_is_appended(table):
    merge(table, pl.LazyFrame({"id": [3, 4], "value": ["c", "d"]}))

    assert_frame_equal(
        read(table), pl.DataFrame({"id": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]})
    )
    assert list(DeltaTable(table).history(1))[0]["operation"] == "WRITE"


def test_matched_source_is_merged(table):
    merge(table, pl.LazyFrame({"id": [2, 3], "value": ["B", "c"]}))

    assert_frame_equal(
        read(table), pl.DataFrame({"id": [1, 2, 3], "value": ["a", "B", "c"]})
    )
    assert list(DeltaTable(table).history(1))[0]["operation"] == "MERGE"


def test_empty_source_commits_nothing(table):
    version = DeltaTable(table).version()

    merge(table, pl.LazyFrame({"id": [1], "value": ["A"]}).filter(pl.lit(False)))

    assert DeltaTable(table).version() == version


def test_append_refused_after_a_concurrent_commit(table):
    dt = DeltaTable(table)
    write_deltalake(table, pl.DataFrame({"id": [3], "value": ["c"]}), mode="append")

    with pytest.raises(DbtRuntimeError, match="changed since version"):
        DeltaFormat._append_if_unmatched(
            dt, table, pl.LazyFrame({"id": [3], "value": ["c"]}), ["id"], False, None
        )

    assert read(table)["id"].to_list() == [1, 2, 3]
