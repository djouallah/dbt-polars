import itertools
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
from dbt.adapters.events.logging import AdapterLogger
from dbt_common.exceptions import DbtRuntimeError
from deltalake import CommitProperties, DeltaTable, write_deltalake
from deltalake.exceptions import CommitFailedError

import polars as pl
from dbt.adapters.polars.catalogs.baseCatalog import get_write_options
from dbt.adapters.polars.utils import first_day_from


logger = AdapterLogger("polars")

# delta-rs' merge spills only when max_spill_size is set; without it its memory pool is
# unbounded. The defaults: a share of the memory this process may use when the merge
# starts, and a share of the temp disk's free space (DataFusion's own on-disk cap is a
# flat 100 GB).
MERGE_SPILL_FRACTION = 0.6
MERGE_TEMP_DIR_FRACTION = 0.8


def _read_int(path: str, key: str | None = None, scale: int = 1) -> int | None:
    """An integer from a /proc or /sys file: the whole file, or the line `key:`."""
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    if key is not None:
        line = next((ln for ln in text.splitlines() if ln.startswith(key + ":")), "")
        text = line.split(":", 1)[-1].split()[0] if line else ""
    text = text.strip()
    return int(text) * scale if text.isdigit() else None


def _cgroup_memory_limit() -> int | None:
    """The tightest memory.max from this process's cgroup up to the root (v2), or the
    v1 limit; None when unlimited or not on Linux."""
    try:
        line = next(
            ln
            for ln in Path("/proc/self/cgroup").read_text().splitlines()
            if ln.startswith("0::")
        )
    except (OSError, StopIteration):
        line = None
    limits = []
    if line is not None:
        base = Path("/sys/fs/cgroup")
        current = base / line[3:].lstrip("/")
        while True:
            limit = _read_int(str(current / "memory.max"))
            if limit:
                limits.append(limit)
            if current == base or base not in current.parents:
                break
            current = current.parent
    v1 = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1 and v1 < 1 << 62:
        limits.append(v1)
    return min(limits) if limits else None


def _memory_limit() -> int | None:
    """The memory this process may use now: the tightest of physical RAM, the cgroup
    limit, and the RAM available plus this process's own resident memory (ours to
    reuse). Sampled when the merge starts, after earlier models took their share."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        total = None
    available = _read_int("/proc/meminfo", "MemAvailable", 1024)
    if available is not None:
        available += _read_int("/proc/self/status", "VmRSS", 1024) or 0
    limits = [v for v in (total, _cgroup_memory_limit(), available) if v]
    return min(limits) if limits else None


def _merge_spill_defaults() -> dict:
    defaults = {}
    memory = _memory_limit()
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
        compat_level: pl.CompatLevel | None = None,
    ) -> pa.Table | pa.RecordBatchReader:
        """A DataFrame as an Arrow table; a LazyFrame as an Arrow stream of its batches,
        so the model result is never held whole next to delta-rs' merge."""
        if isinstance(data, pl.DataFrame):
            return data.to_arrow(compat_level=compat_level)
        batches = (
            batch
            for frame in data.collect_batches(lazy=True)
            for batch in frame.to_arrow(compat_level=compat_level).to_batches()
        )
        first = next(batches, None)
        if first is None:
            empty = pl.DataFrame(schema=data.collect_schema())
            return empty.to_arrow(compat_level=compat_level)
        return pa.RecordBatchReader.from_batches(
            first.schema, itertools.chain([first], batches)
        )

    @staticmethod
    def _append_if_unmatched(
        dt: DeltaTable,
        path: str,
        data: pl.DataFrame | pl.LazyFrame,
        keys: list[str],
        allow_schema_evolution: bool,
        storage_options: dict[str, str] | None,
    ) -> bool:
        """When no key of the source is in the target, the merge only inserts: append
        the source instead of running delta-rs' join, which holds the target's side of
        the join in memory. The target is probed at the version `dt` is loaded at, and
        the append is pinned to it (max_commit_retries=0), so a commit since then fails
        the merge instead of letting a duplicate through. True when handled."""
        version = dt.version()
        target = pl.scan_delta(path, version=version, storage_options=storage_options)
        target_schema = target.collect_schema()
        if any(k not in target_schema for k in keys):
            return False
        source = data.lazy()
        try:
            source_keys = (
                source.select(keys)
                .unique()
                .cast({k: target_schema[k] for k in keys})
                .collect(engine="streaming")
            )
        except pl.exceptions.PolarsError:
            return False
        if source_keys.height == 0:
            # Nothing to insert or update: no commit, unless the schema may evolve.
            return not allow_schema_evolution

        # A target row can only match a source key inside the source's key range:
        # the range lets the scan skip files on their statistics.
        bounds = []
        for k in keys:
            column = source_keys.get_column(k)
            if column.dtype.is_integer() or column.dtype.is_temporal() or (
                column.dtype == pl.String
            ):
                low, high = column.min(), column.max()
                if low is not None:
                    bounds.append(pl.col(k).is_between(low, high))
        candidates = target.select(keys)
        if bounds:
            candidates = candidates.filter(pl.all_horizontal(bounds))
        matched = (
            candidates.join(source_keys.lazy(), on=keys, how="semi")
            .limit(1)
            .collect(engine="streaming")
        )
        if matched.height:
            return False

        columns = source.collect_schema()
        casts = {
            c: target_schema[c]
            for c in columns.names()
            if c in target_schema and columns[c] != target_schema[c]
        }
        if casts:
            source = source.cast(casts)  # type: ignore[arg-type]
        logger.info(
            f"delta merge: no source key in the target at version {version}; "
            "appending the source, pinned to that version"
        )
        try:
            write_deltalake(
                dt,
                DeltaFormat._merge_source(source, pl.CompatLevel.oldest()),
                mode="append",
                schema_mode="merge" if allow_schema_evolution else None,
                commit_properties=CommitProperties(max_commit_retries=0),
            )
        except CommitFailedError as e:
            raise DbtRuntimeError(
                f"merge into {path}: the table changed since version {version} (a "
                "concurrent write committed); merge refused, run it again"
            ) from e
        return True

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
        if not incremental_predicates and DeltaFormat._append_if_unmatched(
            dt, str(path), df, keys, allow_schema_evolution, storage_options
        ):
            return
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
        # The source goes to delta-rs as an Arrow stream. With streamed_exec off (the
        # default) delta-rs collects it and can prune the target on its statistics.
        streamed = kwargs.pop("streamed_exec", False)
        logger.info(
            f"delta merge: streamed_exec={streamed}, "
            + ", ".join(
                f"{k}={kwargs[k] / 2**30:.1f} GiB"
                for k in ("max_spill_size", "max_temp_directory_size")
                if kwargs.get(k)
            )
        )
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
