import contextlib
import json
import math
import re
import time
import uuid
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from dbt.adapters.events.logging import AdapterLogger
from dbt_common.exceptions import DbtRuntimeError
from deltalake import CommitProperties, DeltaTable
from deltalake.exceptions import CommitFailedError
from deltalake.fs import DeltaStorageHandler
from deltalake.schema import ArrayType, MapType, StructType
from deltalake.transaction import AddAction
from pyarrow.fs import PyFileSystem

import polars as pl
from dbt.adapters.polars.catalogs.baseCatalog import get_write_options
from dbt.adapters.polars.utils import first_day_from

logger = AdapterLogger("polars")

# The most distinct source values of one key column the disjoint-key probe looks up
# in the target (_append_if_all_new).
DISJOINT_PROBE_VALUES = 10_000

# The size of a file an append writes, as Polars estimates the rows in memory.
APPEND_FILE_BYTES = 512 * 1024 * 1024

# What an append of new files may commit to, as Polars' own Delta sink allows it
# (polars/io/delta/_sink.py): the writer features it honors, and the variant
# features when the table has no variant column.
ALLOWED_WRITER_FEATURES = frozenset(
    {"appendOnly", "invariants", "timestampNtz", "changeDataFeed", "deletionVectors"}
)
VARIANT_WRITER_FEATURES = frozenset({"variantType", "variantType-preview"})
DEFAULT_NUM_INDEXED_COLS = 32

# A name in backticks, an unquoted name, or a separator.
_STATS_COLUMN_TOKEN = re.compile(r"`((?:[^`]|``)*)`|([^`.,]+)|([.,])")


def _fields(fields: list[Any], prefix: tuple[str, ...] = ()) -> Iterator[Any]:
    """Every field of a Delta schema, nested ones included, with its path."""
    for field in fields:
        path = (*prefix, field.name)
        yield path, field
        if isinstance(field.type, StructType):
            yield from _fields(field.type.fields, path)


def _append_blocker(dt: DeltaTable) -> str | None:
    """Why new files cannot be committed to `dt` as they are written, or None: the
    checks of Polars' Delta sink. Such a table keeps delta-rs' merge."""
    protocol = dt.protocol()
    if protocol.min_writer_version > 7:
        return f"writer version {protocol.min_writer_version}"
    fields = list(_fields(dt.schema().fields))
    if protocol.min_writer_version == 7:
        unsupported = set(protocol.writer_features or ()) - ALLOWED_WRITER_FEATURES
        if not any(getattr(f.type, "type", None) == "variant" for _, f in fields):
            unsupported -= VARIANT_WRITER_FEATURES
        if unsupported:
            return f"writer features {sorted(unsupported)}"
    metadata = dt.metadata()
    configuration = metadata.configuration
    if metadata.partition_columns:
        return "a partitioned table"
    if any(key.startswith("delta.constraints.") for key in configuration):
        return "CHECK constraints"
    if (configuration.get("delta.columnMapping.mode") or "none") != "none":
        return "column mapping"
    if (configuration.get("delta.enableRowTracking") or "false").lower() == "true":
        return "row tracking"
    for _, field in fields:
        for key in field.metadata:
            if key in ("delta.generationExpression", "delta.invariants") or (
                key.startswith("delta.identity.")
            ):
                return f"column {field.name!r} has {key}"
    return None


def _stats_column_paths(stats_columns: str) -> list[tuple[str, ...]]:
    """The column paths `delta.dataSkippingStatsColumns` lists: commas separate
    columns, dots the fields of a nested one, backticks quote a name."""
    paths: list[tuple[str, ...]] = []
    path: list[str] = []
    name = ""
    for m in _STATS_COLUMN_TOKEN.finditer(stats_columns + ","):
        quoted, unquoted, separator = m.groups()
        if quoted is not None:
            name += quoted.replace("``", "`")
        elif unquoted is not None:
            name += unquoted.strip()
        else:
            path.append(name)
            name = ""
            if separator == ",":
                if any(path):
                    paths.append(tuple(path))
                path = []
    return paths


def _stats_leaves(dt: DeltaTable) -> set[tuple[str, ...]]:
    """The leaf columns a file's stats cover, as delta-rs and Polars choose them:
    `delta.dataSkippingStatsColumns`, else the first
    `delta.dataSkippingNumIndexedCols` top-level columns (all when negative).
    Leaves in a list or map, and binary leaves, get none."""
    leaves: list[tuple[str, ...]] = []

    def visit(dtype: Any, path: tuple[str, ...]) -> None:
        if isinstance(dtype, StructType):
            for child in dtype.fields:
                visit(child.type, (*path, child.name))
        elif not isinstance(dtype, (ArrayType, MapType)) and dtype.type != "binary":
            leaves.append(path)

    configuration = dt.metadata().configuration
    fields = list(dt.schema().fields)
    if listed := configuration.get("delta.dataSkippingStatsColumns"):
        selected = _stats_column_paths(listed)
        for field in fields:
            visit(field.type, (field.name,))
        return {leaf for leaf in leaves if any(leaf[: len(p)] == p for p in selected)}
    count = int(
        configuration.get("delta.dataSkippingNumIndexedCols")
        or DEFAULT_NUM_INDEXED_COLS
    )
    for field in fields if count < 0 else fields[:count]:
        visit(field.type, (field.name,))
    return set(leaves)


def _non_nullable_leaves(dt: DeltaTable) -> list[tuple[str, ...]]:
    """The leaves whose null count must be 0: non-nullable, as every struct above
    them, and not in a list or map."""
    out: list[tuple[str, ...]] = []

    def visit(field: Any, path: tuple[str, ...]) -> None:
        if field.nullable:
            return
        if isinstance(field.type, StructType):
            for child in field.type.fields:
                visit(child, (*path, child.name))
        elif not isinstance(field.type, (ArrayType, MapType)):
            out.append(path)

    for field in dt.schema().fields:
        visit(field, (field.name,))
    return out


class _Decimal(str):
    """A decimal written as a bare JSON number, all digits of its scale."""


def _timestamp(micros: int, utc: bool, is_max: bool) -> str | None:
    """A timestamp in the millisecond text Delta stats hold, the fraction left out
    when it is zero. A min is truncated; a max is rounded up, so the bound holds for
    every reader: delta-rs truncates both and counts on readers to widen the max by
    1 ms, which Polars' scan_delta does not (a row at .678901 under a max of .678 is
    skipped)."""
    if is_max:
        micros = -(-micros // 1000) * 1000
    value = datetime(1970, 1, 1) + timedelta(microseconds=micros)
    if not 1 <= value.year <= 9999:
        return None
    fraction = f".{value.microsecond // 1000:03d}" if value.microsecond else ""
    if utc:
        return value.strftime("%Y-%m-%dT%H:%M:%S") + fraction + "Z"
    return value.strftime("%Y-%m-%d %H:%M:%S") + fraction


def _decimal(unscaled: int, scale: int) -> tuple[Any, Any]:
    sign, digits = ("-" if unscaled < 0 else ""), str(abs(unscaled))
    if scale:
        digits = digits.rjust(scale + 1, "0")
        digits = f"{digits[:-scale]}.{digits[-scale:]}"
    return unscaled, _Decimal(sign + digits)


def _bound(statistics: Any, is_min: bool) -> tuple[Any, Any] | None:
    """A row group's min or max as (sort key, Delta value), or None when Delta has
    no value for it (NaN, binary data, a year outside 1 to 9999)."""
    raw = statistics.min_raw if is_min else statistics.max_raw
    logical = json.loads(statistics.logical_type.to_json() or "{}")
    kind = logical.get("Type")
    physical = statistics.physical_type
    if physical == "BOOLEAN":
        return raw, bool(raw)
    if physical in ("FLOAT", "DOUBLE"):
        if math.isnan(raw):
            return None
        return raw, raw if math.isfinite(raw) else None
    if kind == "Decimal" and physical in ("INT32", "INT64"):
        return _decimal(int(raw), int(logical["scale"]))
    if kind == "Decimal" and physical in ("BYTE_ARRAY", "FIXED_LEN_BYTE_ARRAY"):
        return _decimal(int.from_bytes(raw, "big", signed=True), int(logical["scale"]))
    if physical in ("INT32", "INT64") and kind in (None, "None", "Int"):
        value = int(raw)
        bits = 32 if physical == "INT32" else 64
        if kind == "Int" and not logical.get("isSigned", True) and value < 0:
            value += 1 << bits
        return value, value
    if physical == "INT32" and kind == "Date":
        day = date(1970, 1, 1) + timedelta(days=int(raw))
        return (int(raw), day.isoformat()) if 1 <= day.year <= 9999 else None
    if physical == "INT64" and kind == "Timestamp":
        unit, raw = logical.get("timeUnit"), int(raw)
        if unit == "milliseconds":
            micros = raw * 1000
        elif unit == "nanoseconds":
            micros = raw // 1000 if is_min else -(-raw // 1000)
        else:
            micros = raw
        text = _timestamp(micros, bool(logical.get("isAdjustedToUTC")), not is_min)
        return (micros, text) if text is not None else None
    if physical == "BYTE_ARRAY" and kind in ("String", "Enum", "JSON"):
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return text, text
    return None


def _file_stats(
    metadata: pq.FileMetaData,
    leaves: set[tuple[str, ...]],
    non_nullable: list[tuple[str, ...]],
    file: str,
) -> str:
    """A data file's Delta stats from its Parquet footer, by the rules of Polars'
    Delta sink (polars-python/src/io/delta_stats.rs): the row count, and the null
    count, min and max of each leaf in `leaves`, nested in its structs."""
    stats: dict[str, Any] = {
        "numRecords": metadata.num_rows,
        "minValues": {},
        "maxValues": {},
        "nullCount": {},
    }
    for j in range(metadata.num_columns if metadata.num_row_groups else 0):
        path = tuple(metadata.schema.column(j).path.split("."))
        null_count: int | None = 0
        low: tuple[Any, Any] | None = None
        high: tuple[Any, Any] | None = None
        usable = True
        for i in range(metadata.num_row_groups):
            chunk = metadata.row_group(i).column(j)
            st = chunk.statistics
            if st is None or not st.has_null_count:
                null_count = None
            elif null_count is not None:
                null_count += st.null_count
            if st is not None and st.has_min_max:
                lo, hi = _bound(st, True), _bound(st, False)
                if lo is None or hi is None:
                    usable = False
                    continue
                low = lo if low is None or lo[0] < low[0] else low
                high = hi if high is None or hi[0] > high[0] else high
            elif st is None or chunk.num_values - (st.null_count or 0) > 0:
                # A row group of only nulls leaves the bounds as they are.
                usable = False
        if path in non_nullable and metadata.num_rows and null_count != 0:
            raise DbtRuntimeError(
                f"non-nullable column '{'.'.join(path)}' has nulls in {file}"
            )
        if path not in leaves:
            continue
        for key, value in (
            ("minValues", low[1] if usable and low else None),
            ("maxValues", high[1] if usable and high else None),
            ("nullCount", null_count),
        ):
            if value is None and not (key != "nullCount" and usable and low):
                continue
            node = stats[key]
            for name in path[:-1]:
                node = node.setdefault(name, {})
            node[path[-1]] = value
    return _json(stats)


def _json(node: Any) -> str:
    """JSON of the stats, with decimals as bare numbers."""
    if isinstance(node, dict):
        return (
            "{" + ",".join(f"{json.dumps(k)}:{_json(v)}" for k, v in node.items()) + "}"
        )
    if isinstance(node, _Decimal):
        return str.__str__(node)
    return json.dumps(node)


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
    def _append_if_all_new(
        dt: DeltaTable,
        path: str,
        data: pl.DataFrame | pl.LazyFrame,
        keys: list[str],
        storage_options: dict[str, str] | None,
    ) -> bool:
        """When no key of `data` is in the table, the merge only inserts: Polars writes
        the rows as new files, streaming, and delta-rs only commits them, instead of
        running delta-rs' merge, whose join holds the target in memory. The table is
        probed at the version `dt` is loaded at and the commit is pinned to it
        (max_commit_retries=0): a commit since then fails it. True if appended."""
        if _append_blocker(dt) is not None:
            return False
        version = dt.version()
        target = pl.scan_delta(path, version=version, storage_options=storage_options)
        schema = target.collect_schema()
        source = data.lazy()
        if set(source.collect_schema().names()) != set(schema.names()):
            return False
        source = source.select(schema.names()).cast(dict(schema))  # type: ignore[arg-type]

        # A target row can only match inside the source's key range, which lets the
        # scan skip files on their statistics. The range is one streaming pass.
        ranged = [
            k
            for k in keys
            if schema[k].is_integer()
            or schema[k].is_temporal()
            or schema[k] == pl.String
        ]
        extremes = source.select(
            pl.len().alias("rows"),
            *(pl.col(k).min().alias(f"min_{i}") for i, k in enumerate(ranged)),
            *(pl.col(k).max().alias(f"max_{i}") for i, k in enumerate(ranged)),
            *(pl.col(k).approx_n_unique().alias(f"n_{i}") for i, k in enumerate(keys)),
        ).collect(engine="streaming")
        rows = extremes.item(0, "rows")
        if rows == 0:
            return True
        bounds = [
            pl.col(k).is_between(
                pl.lit(extremes.item(0, f"min_{i}"), dtype=schema[k]),
                pl.lit(extremes.item(0, f"max_{i}"), dtype=schema[k]),
            )
            for i, k in enumerate(ranged)
            if extremes.item(0, f"min_{i}") is not None
        ]
        candidates = target.select(keys)
        if bounds:
            candidates = candidates.filter(pl.all_horizontal(bounds))
        if not candidates.limit(1).collect(engine="streaming").height:
            path_taken = "no target row in the key range"
        elif disjoint := DeltaFormat._disjoint_key(source, candidates, keys, extremes):
            path_taken = f"no source {disjoint} in the target"
        else:
            # The semi-join holds the target's keys in the range in memory: the last
            # resort, when every key column has values on both sides.
            matched = source.select(keys).join(candidates, on=keys, how="semi")
            if matched.limit(1).collect(engine="streaming").height:
                logger.info(f"merge into {path}: {rows} rows, a key matches: merge")
                return False
            path_taken = "no key matches (semi-join)"
        logger.info(f"merge into {path}: {rows} rows, all new, {path_taken}: append")

        prefix = f"dbt-polars-{uuid.uuid4()}"
        fs = PyFileSystem(DeltaStorageHandler(dt.table_uri, options=storage_options))
        sinked: list[Any] = []
        try:
            source.sink_parquet(
                pl.PartitionBy(
                    f"{path.rstrip('/')}/{prefix}",
                    approximate_bytes_per_file=APPEND_FILE_BYTES,
                ),
                storage_options=storage_options,
                mkdir=True,
                sinked_paths_callback=lambda args: sinked.extend(args.paths),
            )
            leaves = _stats_leaves(dt)
            non_nullable = _non_nullable_leaves(dt)
            now = int(time.time() * 1000)
            actions = []
            for file in sinked:
                # The path relative to the table root, as the log records it.
                relative = prefix + file.path.replace("\\", "/").split(prefix, 1)[1]
                metadata = pq.read_metadata(fs.open_input_file(relative))
                if metadata.num_rows != file.num_rows:
                    raise DbtRuntimeError(
                        f"merge into {path}: {relative} has {metadata.num_rows} rows, "
                        f"the sink wrote {file.num_rows}"
                    )
                actions.append(
                    AddAction(
                        path=relative,
                        size=fs.get_file_info(relative).size,
                        partition_values={},
                        modification_time=now,
                        data_change=True,
                        stats=_file_stats(metadata, leaves, non_nullable, relative),
                    )
                )
            dt.create_write_transaction(
                actions,  # type: ignore[arg-type]
                mode="append",
                schema=dt.schema(),
                commit_properties=CommitProperties(max_commit_retries=0),
            )
        except BaseException as e:
            with contextlib.suppress(Exception):
                fs.delete_dir(prefix)
            if isinstance(e, CommitFailedError):
                raise DbtRuntimeError(
                    f"merge into {path}: the table changed since version {version} "
                    "(a concurrent write committed); merge refused, run it again"
                ) from e
            raise
        return True

    @staticmethod
    def _disjoint_key(
        source: pl.LazyFrame,
        candidates: pl.LazyFrame,
        keys: list[str],
        extremes: pl.DataFrame,
    ) -> str | None:
        """A key column none of whose source values is in the target, which proves that
        no key matches: the column with the fewest distinct source values, if it has at
        most DISJOINT_PROBE_VALUES. Its values are looked up in a streaming scan of
        that one target column, which holds no target rows, unlike the semi-join on
        every key. Its name, or None if it overlaps or every column has too many
        values."""
        counts = {k: extremes.item(0, f"n_{i}") for i, k in enumerate(keys)}
        key = min(keys, key=lambda k: counts[k])
        if counts[key] > DISJOINT_PROBE_VALUES:
            return None
        values = source.select(pl.col(key).unique()).collect(engine="streaming")
        if values.height > DISJOINT_PROBE_VALUES:
            return None
        hit = (
            candidates.select(key)
            .filter(pl.col(key).is_in(values.get_column(key).implode()))
            .limit(1)
            .collect(engine="streaming")
        )
        return None if hit.height else key

    @staticmethod
    def merge(
        path: str | Path,
        df: pl.DataFrame | pl.LazyFrame,
        keys: list[str],
        except_cols: list[str] | None,
        incremental_predicates: list[str] | None,
        allow_schema_evolution: bool,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        predicate = " AND ".join(
            f"DBT_INTERNAL_SOURCE.{k} = DBT_INTERNAL_DEST.{k}" for k in keys
        )
        if incremental_predicates:
            predicate += " AND " + " AND ".join(incremental_predicates)
        dt = DeltaTable(str(path), storage_options=storage_options)
        if not incremental_predicates and DeltaFormat._append_if_all_new(
            dt, str(path), df, keys, storage_options
        ):
            return
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        (
            dt.merge(
                df.to_arrow(),
                predicate,
                source_alias="DBT_INTERNAL_SOURCE",
                target_alias="DBT_INTERNAL_DEST",
                merge_schema=allow_schema_evolution,
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
