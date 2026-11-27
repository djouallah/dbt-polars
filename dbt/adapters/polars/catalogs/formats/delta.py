import itertools
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
from deltalake import DeltaTable

import polars as pl
from dbt.adapters.polars.catalogs.baseCatalog import get_write_options
from dbt.adapters.polars.utils import first_day_from


# delta-rs' merge spills only when max_spill_size is set; without it its memory pool is
# unbounded. The defaults: a share of the memory limit in memory, a share of the temp
# disk's free space on disk (DataFusion's own on-disk cap is a flat 100 GB).
MERGE_SPILL_FRACTION = 0.6
MERGE_TEMP_DIR_FRACTION = 0.8


def _memory_limit_bytes() -> int | None:
    """The cgroup memory limit, else physical RAM; None when neither can be read."""
    for limit_file in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            text = Path(limit_file).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 1 << 60:
            return int(text)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def _merge_spill_defaults() -> dict:
    defaults = {}
    memory = _memory_limit_bytes()
    if memory:
        defaults["max_spill_size"] = int(memory * MERGE_SPILL_FRACTION)
    try:
        free = shutil.disk_usage(tempfile.gettempdir()).free
    except OSError:
        free = 0
    if free:
        defaults["max_temp_directory_size"] = int(free * MERGE_TEMP_DIR_FRACTION)
    return defaults


class DeltaFormat:
    @staticmethod
    def _write(
        path: str,
        data: pl.DataFrame | pl.LazyFrame,
        mode: str,
        model_config: dict,
        adapter_delta_opts: dict,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        if (
            isinstance(data, pl.LazyFrame)
            and model_config.get("write_mode", "lazy") == "lazy"
        ):
            kwargs = get_write_options(
                pl.LazyFrame.sink_delta,
                model_config,
                ignore={"mode", "target", "storage_options"},
                merge={"delta_write_options": adapter_delta_opts},
            )
            data.sink_delta(path, mode=mode, storage_options=storage_options, **kwargs)  # type: ignore[call-overload]
        else:
            df = data.collect() if isinstance(data, pl.LazyFrame) else data
            kwargs = get_write_options(
                pl.DataFrame.write_delta,
                model_config,
                ignore={"mode", "target", "storage_options"},
                merge={"delta_write_options": adapter_delta_opts},
            )
            df.write_delta(path, mode=mode, storage_options=storage_options, **kwargs)  # type: ignore[call-overload]

    @staticmethod
    def write(
        path: str | Path,
        data: pl.DataFrame | pl.LazyFrame,
        mode: str,
        model_config: dict,
        partition_by: list[str] | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        adapter_delta_opts: dict = {
            "schema_mode": "overwrite" if mode == "overwrite" else "merge"
        }
        if partition_by:
            adapter_delta_opts["partition_by"] = partition_by
        DeltaFormat._write(
            str(path),
            data,
            mode,
            model_config,
            adapter_delta_opts,
            storage_options=storage_options,
        )

    @staticmethod
    def read(
        path: str | Path,
        storage_options: dict[str, str] | None = None,
    ) -> pl.LazyFrame:
        return pl.scan_delta(str(path), storage_options=storage_options)

    @staticmethod
    def append(
        path: str | Path,
        data: pl.DataFrame | pl.LazyFrame,
        allow_schema_evolution: bool,
        model_config: dict,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        adapter_delta_opts = {"schema_mode": "merge"} if allow_schema_evolution else {}
        DeltaFormat._write(
            str(path),
            data,
            "append",
            model_config,
            adapter_delta_opts,
            storage_options=storage_options,
        )

    @staticmethod
    def replace_window(
        path: str | Path,
        data: pl.DataFrame | pl.LazyFrame,
        event_time: str,
        start: datetime,
        end: datetime,
        allow_schema_evolution: bool,
        model_config: dict,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        """Overwrites only the rows whose event_time lies in `[start, end)`, in one
        commit. Delta rejects a row of `data` outside that range."""
        dtype = data.collect_schema()[event_time]
        if dtype == pl.Date:
            low, high = (first_day_from(moment).isoformat() for moment in (start, end))
        elif isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            low, high = str(start), str(end)
        else:
            low, high = (
                moment.replace(tzinfo=timezone.utc).isoformat()
                for moment in (start, end)
            )
        column = '"' + event_time.replace('"', '""') + '"'
        adapter_delta_opts: dict = {
            "predicate": f"{column} >= '{low}' AND {column} < '{high}'"
        }
        if allow_schema_evolution:
            adapter_delta_opts["schema_mode"] = "merge"
        DeltaFormat._write(
            str(path),
            data,
            "overwrite",
            model_config,
            adapter_delta_opts,
            storage_options=storage_options,
        )

    @staticmethod
    def _merge_source(
        data: pl.DataFrame | pl.LazyFrame,
    ) -> pa.Table | pa.RecordBatchReader:
        """A DataFrame as an Arrow table; a LazyFrame as an Arrow stream of its batches,
        so the model result is never held whole next to delta-rs' merge."""
        if isinstance(data, pl.DataFrame):
            return data.to_arrow()
        batches = (
            batch
            for frame in data.collect_batches(lazy=True)
            for batch in frame.to_arrow().to_batches()
        )
        first = next(batches, None)
        if first is None:
            return pl.DataFrame(schema=data.collect_schema()).to_arrow()
        return pa.RecordBatchReader.from_batches(
            first.schema, itertools.chain([first], batches)
        )

    @staticmethod
    def merge(
        path: str | Path,
        df: pl.DataFrame | pl.LazyFrame,
        keys: list[str],
        except_cols: list[str] | None,
        incremental_predicates: list[str] | None,
        allow_schema_evolution: bool,
        storage_options: dict[str, str] | None = None,
        model_config: dict | None = None,
    ) -> None:
        predicate = " AND ".join(
            f"DBT_INTERNAL_SOURCE.{k} = DBT_INTERNAL_DEST.{k}" for k in keys
        )
        if incremental_predicates:
            predicate += " AND " + " AND ".join(incremental_predicates)
        dt = DeltaTable(str(path), storage_options=storage_options)
        kwargs = get_write_options(
            DeltaTable.merge,
            model_config or {},
            ignore={
                "source",
                "predicate",
                "source_alias",
                "target_alias",
                "merge_schema",
            },
        )
        kwargs = {**_merge_spill_defaults(), **kwargs}
        # streamed_exec=False collects the source, so delta-rs can prune the target's
        # files on the source's key statistics; streamed, it reads the whole target.
        streamed = kwargs.pop("streamed_exec", True)
        if not streamed and isinstance(df, pl.LazyFrame):
            df = df.collect()
        (
            dt.merge(
                DeltaFormat._merge_source(df),
                predicate,
                source_alias="DBT_INTERNAL_SOURCE",
                target_alias="DBT_INTERNAL_DEST",
                merge_schema=allow_schema_evolution,
                streamed_exec=streamed,
                **kwargs,
            )
            .when_matched_update_all(except_cols=except_cols)
            .when_not_matched_insert_all()
            .execute()
        )

    @staticmethod
    def delete_matched(
        path: str | Path,
        df: pl.DataFrame,
        keys: list[str],
        incremental_predicates: list[str] | None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        predicate = " AND ".join(
            f"DBT_INTERNAL_SOURCE.{k} = DBT_INTERNAL_DEST.{k}" for k in keys
        )
        if incremental_predicates:
            predicate += " AND " + " AND ".join(incremental_predicates)
        dt = DeltaTable(str(path), storage_options=storage_options)
        (
            dt.merge(
                df.to_arrow(),
                predicate,
                source_alias="DBT_INTERNAL_SOURCE",
                target_alias="DBT_INTERNAL_DEST",
            )
            .when_matched_delete()
            .execute()
        )

    @staticmethod
    def truncate(
        path: str | Path,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        DeltaTable(str(path), storage_options=storage_options).delete()

    @staticmethod
    def apply_snapshot(
        path: str | Path,
        rows_to_close: pl.DataFrame,
        rows_to_insert: pl.DataFrame,
        scd_id_col: str,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        staging = pl.concat([rows_to_close, rows_to_insert])
        dt = DeltaTable(str(path), storage_options=storage_options)
        (
            dt.merge(
                staging.to_arrow(),
                f"target.{scd_id_col} = source.{scd_id_col}",
                source_alias="source",
                target_alias="target",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )

    @staticmethod
    def get_partition_columns(
        path: str | Path,
        storage_options: dict[str, str] | None = None,
    ) -> list[str]:
        return (
            DeltaTable(str(path), storage_options=storage_options)
            .metadata()
            .partition_columns
        )

    @staticmethod
    def set_relation_comment(
        path: str | Path,
        comment: str,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        DeltaTable(
            str(path), storage_options=storage_options
        ).alter.set_table_description(comment)

    @staticmethod
    def get_relation_comment(
        path: str | Path,
        storage_options: dict[str, str] | None = None,
    ) -> str | None:
        return (
            DeltaTable(str(path), storage_options=storage_options)
            .metadata()
            .description
        )

    @staticmethod
    def set_column_comments(
        path: str | Path,
        comments: dict[str, str],
        storage_options: dict[str, str] | None = None,
    ) -> None:
        dt = DeltaTable(str(path), storage_options=storage_options)
        for column, comment in comments.items():
            dt.alter.set_column_metadata(column, {"comment": comment})

    @staticmethod
    def get_column_comments(
        path: str | Path,
        storage_options: dict[str, str] | None = None,
    ) -> dict[str, str]:
        dt = DeltaTable(str(path), storage_options=storage_options)
        return {
            field.name: field.metadata["comment"]
            for field in dt.schema().fields
            if field.metadata.get("comment")
        }
