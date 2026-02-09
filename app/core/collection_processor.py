"""
Collection Processor Service
==============================
Handles the heavy-lifting for multi-table dataset operations:

  1. CSV Introspection  — detect column types, uniqueness, sample values
  2. Key Validation     — validate join keys match between tables
  3. Relationship Hints — auto-suggest joins by column name matching
  4. Aggregation Engine — GROUP BY + aggregate functions → flattened features
  5. Merge Pipeline     — execute joins in order → single merged CSV
"""

import os
import json
import math
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Map pandas dtypes → UI-friendly display types
DTYPE_MAP = {
    "int8": "INTEGER", "int16": "INTEGER", "int32": "INTEGER", "int64": "INTEGER",
    "uint8": "INTEGER", "uint16": "INTEGER", "uint32": "INTEGER", "uint64": "INTEGER",
    "float16": "FLOAT", "float32": "FLOAT", "float64": "FLOAT",
    "bool": "BOOLEAN",
    "object": "VARCHAR",
    "string": "VARCHAR",
    "category": "CATEGORY",
    "datetime64[ns]": "DATETIME", "datetime64": "DATETIME",
    "timedelta64[ns]": "TIMEDELTA",
}

# Functions that work on numeric columns only
NUMERIC_ONLY_FUNCTIONS = {"sum", "mean", "median", "std", "variance"}

# Functions that work on any column type
UNIVERSAL_FUNCTIONS = {"count", "unique_count", "first", "last", "min", "max", "mode"}

# Pandas aggregation function mapping
AGG_FUNC_MAP = {
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "max": "max",
    "min": "min",
    "count": "count",
    "unique_count": "nunique",
    "nunique": "nunique",           # Frontend sends this directly
    "std": "std",
    "variance": "var",
    "var": "var",                   # Frontend sends this directly
    "first": "first",
    "last": "last",
}


# ============================================================================
# 1. CSV INTROSPECTION
# ============================================================================

def introspect_csv(file_path: str, sample_size: int = 5) -> Dict[str, Any]:
    """
    Analyze a CSV file and return column metadata.

    Returns:
        {
            "row_count": 307511,
            "column_count": 122,
            "file_size_bytes": 167540445,
            "columns": [
                {
                    "name": "SK_ID_CURR",
                    "dtype": "int64",
                    "display_type": "INTEGER",
                    "nullable": false,
                    "unique_count": 307511,
                    "null_count": 0,
                    "null_percentage": 0.0,
                    "sample_values": [100002, 100003, 100004],
                    "is_potential_key": true
                },
                ...
            ]
        }
    """
    logger.info(f"📊 Introspecting CSV: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    file_size = os.path.getsize(file_path)

    # Read the full CSV for accurate stats (use chunking for very large files)
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise ValueError(f"Cannot read CSV file: {e}")

    row_count = len(df)
    column_count = len(df.columns)

    columns = []
    for col_name in df.columns:
        col = df[col_name]
        dtype_str = str(col.dtype)
        display_type = DTYPE_MAP.get(dtype_str, "VARCHAR")

        # Detect date columns stored as object
        if dtype_str == "object":
            sample_vals = col.dropna().head(20)
            if len(sample_vals) > 0:
                try:
                    pd.to_datetime(sample_vals, format="mixed")
                    display_type = "DATE"
                except (ValueError, TypeError):
                    pass

        null_count = int(col.isnull().sum())
        null_pct = round((null_count / row_count) * 100, 2) if row_count > 0 else 0.0

        # Unique count (expensive for large datasets — cap at 1M sample)
        try:
            unique_count = int(col.nunique())
        except Exception:
            unique_count = None

        # Sample values (safe serialization)
        try:
            raw_samples = col.dropna().head(sample_size).tolist()
            sample_values = _safe_serialize_list(raw_samples)
        except Exception:
            sample_values = []

        is_key = (unique_count == row_count) if unique_count is not None else False

        columns.append({
            "name": col_name,
            "dtype": dtype_str,
            "display_type": display_type,
            "nullable": null_count > 0,
            "unique_count": unique_count,
            "null_count": null_count,
            "null_percentage": null_pct,
            "sample_values": sample_values,
            "is_potential_key": is_key,
        })

    logger.info(f"✅ Introspected {column_count} columns, {row_count} rows")

    return {
        "row_count": row_count,
        "column_count": column_count,
        "file_size_bytes": file_size,
        "columns": columns,
    }


# ============================================================================
# 2. RELATIONSHIP VALIDATION
# ============================================================================

def validate_relationship(
        left_path: str,
        right_path: str,
        left_column: str,
        right_column: str
) -> Dict[str, Any]:
    """
    Validate a join relationship between two tables.

    Returns match stats, orphan counts, and detected relationship type.
    """
    logger.info(f"🔗 Validating relationship: {left_column} ↔ {right_column}")

    try:
        left_df = pd.read_csv(left_path, usecols=[left_column], low_memory=False)
        right_df = pd.read_csv(right_path, usecols=[right_column], low_memory=False)
    except ValueError as e:
        return {
            "is_validated": False,
            "warnings": [f"Column not found: {e}"],
        }

    left_values = set(left_df[left_column].dropna().unique())
    right_values = set(right_df[right_column].dropna().unique())

    left_unique = len(left_values)
    right_unique = len(right_values)

    matches = left_values & right_values
    match_count = len(matches)

    # Match percentage relative to the smaller table
    base = min(left_unique, right_unique) if min(left_unique, right_unique) > 0 else 1
    match_pct = round((match_count / base) * 100, 2)

    orphan_left = len(left_values - right_values)
    orphan_right = len(right_values - left_values)

    # Detect relationship type
    left_row_count = len(left_df)
    right_row_count = len(right_df)
    left_is_unique = (left_unique == left_row_count)
    right_is_unique = (right_unique == right_row_count)

    if left_is_unique and right_is_unique:
        rel_type = "one_to_one"
    elif left_is_unique and not right_is_unique:
        rel_type = "one_to_many"
    elif not left_is_unique and right_is_unique:
        rel_type = "many_to_one"
    else:
        rel_type = "many_to_many"

    # Build warnings
    warnings = []
    if match_pct < 50:
        warnings.append(f"Low key match rate ({match_pct}%) — possible wrong join columns")
    if orphan_left > 0:
        warnings.append(f"{orphan_left} keys in left table have no match in right table")
    if orphan_right > 0:
        warnings.append(f"{orphan_right} keys in right table have no match in left table")
    if left_df[left_column].dtype != right_df[right_column].dtype:
        warnings.append(f"Data type mismatch: {left_df[left_column].dtype} vs {right_df[right_column].dtype}")

    result = {
        "is_validated": True,
        "relationship_type": rel_type,
        "left_unique_count": left_unique,
        "right_unique_count": right_unique,
        "match_count": match_count,
        "match_percentage": match_pct,
        "orphan_left_count": orphan_left,
        "orphan_right_count": orphan_right,
        "left_column_dtype": str(left_df[left_column].dtype),
        "right_column_dtype": str(right_df[right_column].dtype),
        "warnings": warnings,
    }

    logger.info(f"✅ Validated: {rel_type}, {match_pct}% match, {len(warnings)} warnings")
    return result


# ============================================================================
# 3. AUTO-SUGGEST RELATIONSHIPS
# ============================================================================

def suggest_relationships(
        tables: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Auto-detect potential join keys by matching column names across tables.

    Strategy (in priority order):
      1. Exact column name match (e.g. "SK_ID_CURR" in both tables)
      2. Primary key pattern (_id suffix) matching
      3. Potential key columns (high uniqueness)

    Args:
        tables: List of {id, table_name, file_path, role, columns_metadata}

    Returns:
        List of suggested relationships with confidence scores
    """
    suggestions = []
    primary = None
    related_tables = []

    for t in tables:
        if t.get("role") == "primary":
            primary = t
        else:
            related_tables.append(t)

    if not primary or not related_tables:
        return suggestions

    primary_cols = _parse_columns_metadata(primary.get("columns_metadata", "[]"))
    primary_col_names = {c["name"] for c in primary_cols}

    for rel_table in related_tables:
        rel_cols = _parse_columns_metadata(rel_table.get("columns_metadata", "[]"))
        rel_col_names = {c["name"] for c in rel_cols}

        # Strategy 1: Exact name match
        common_cols = primary_col_names & rel_col_names
        for col_name in common_cols:
            p_col = next((c for c in primary_cols if c["name"] == col_name), None)
            r_col = next((c for c in rel_cols if c["name"] == col_name), None)

            if p_col and r_col:
                confidence = 0.9
                reason = f"Exact column name match: '{col_name}'"

                # Boost confidence if it's a potential key
                if p_col.get("is_potential_key"):
                    confidence = 0.95
                    reason = f"Primary key match: '{col_name}'"

                # Lower confidence for generic names
                if col_name.lower() in ("id", "index", "name", "type", "status", "date"):
                    confidence = 0.5
                    reason = f"Common column name match: '{col_name}' (may be coincidental)"

                suggestions.append({
                    "left_table_id": primary["id"],
                    "left_table_name": primary["table_name"],
                    "right_table_id": rel_table["id"],
                    "right_table_name": rel_table["table_name"],
                    "left_column": col_name,
                    "right_column": col_name,
                    "confidence": confidence,
                    "reason": reason,
                })

        # Strategy 2: _id suffix pattern
        for p_col in primary_cols:
            if p_col["name"].lower().endswith("_id") or p_col.get("is_potential_key"):
                for r_col in rel_cols:
                    if r_col["name"] == p_col["name"]:
                        continue  # Already covered in exact match
                    # Check if related table has a column like "SK_ID_CURR" matching primary's "SK_ID_CURR"
                    if (r_col["name"].lower().endswith("_id") and
                            _similar_column_name(p_col["name"], r_col["name"])):
                        suggestions.append({
                            "left_table_id": primary["id"],
                            "left_table_name": primary["table_name"],
                            "right_table_id": rel_table["id"],
                            "right_table_name": rel_table["table_name"],
                            "left_column": p_col["name"],
                            "right_column": r_col["name"],
                            "confidence": 0.6,
                            "reason": f"ID pattern match: '{p_col['name']}' ↔ '{r_col['name']}'",
                        })

    # Sort by confidence descending
    suggestions.sort(key=lambda s: s["confidence"], reverse=True)

    # Deduplicate (keep highest confidence per table pair)
    seen = set()
    deduped = []
    for s in suggestions:
        key = (s["left_table_id"], s["right_table_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


# ============================================================================
# 4. AGGREGATION ENGINE
# ============================================================================

def compute_aggregation(
        file_path: str,
        group_by_column: str,
        column_prefix: str,
        features: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate a detail table by grouping and applying functions.

    Args:
        file_path: Path to the detail CSV
        group_by_column: Column to GROUP BY
        column_prefix: Prefix for output columns (e.g. "BUREAU_")
        features: [{column: "amount", functions: ["sum", "mean"]}, ...]

    Returns:
        (aggregated_df, created_column_names)
    """
    logger.info(f"⚙️ Aggregating: {file_path} GROUP BY {group_by_column}")

    df = pd.read_csv(file_path, low_memory=False)

    if group_by_column not in df.columns:
        raise ValueError(f"Group-by column '{group_by_column}' not found in table")

    # Build aggregation dict: {column: [func1, func2, ...]}
    agg_dict = {}
    for feat in features:
        col = feat["column"]
        if col not in df.columns:
            logger.warning(f"⚠️ Column '{col}' not found, skipping")
            continue

        funcs = []
        for fn in feat["functions"]:
            fn_str = fn if isinstance(fn, str) else fn.value
            pandas_fn = AGG_FUNC_MAP.get(fn_str)
            if pandas_fn:
                funcs.append(pandas_fn)
            else:
                logger.warning(f"⚠️ Unknown function '{fn_str}', skipping")

        if funcs:
            agg_dict[col] = funcs

    if not agg_dict:
        raise ValueError("No valid aggregation features configured")

    # Execute aggregation
    grouped = df.groupby(group_by_column).agg(agg_dict)

    # Flatten MultiIndex columns → "BUREAU_amount_sum"
    created_columns = []
    new_col_names = []
    for col, func in grouped.columns:
        new_name = f"{column_prefix}{col}_{func}"
        # Rename nunique → unique_count for consistency
        new_name = new_name.replace("_nunique", "_unique_count").replace("_var", "_variance")
        new_col_names.append(new_name)
        created_columns.append(new_name)

    grouped.columns = new_col_names
    grouped = grouped.reset_index()

    logger.info(f"✅ Aggregated: {len(grouped)} groups, {len(created_columns)} features")
    return grouped, created_columns


def compute_created_columns(
        column_prefix: str,
        features: List[Dict[str, Any]]
) -> List[str]:
    """
    Preview what columns will be created by an aggregation config,
    WITHOUT actually reading the data.
    """
    columns = []
    for feat in features:
        col = feat["column"]
        for fn in feat["functions"]:
            fn_str = fn if isinstance(fn, str) else fn.value
            pandas_fn = AGG_FUNC_MAP.get(fn_str, fn_str)
            name = f"{column_prefix}{col}_{pandas_fn}"
            name = name.replace("_nunique", "_unique_count").replace("_var", "_variance")
            columns.append(name)
    return columns


# ============================================================================
# 5. MERGE PIPELINE
# ============================================================================

def execute_merge_pipeline(
        primary_path: str,
        relationships: List[Dict[str, Any]],
        aggregations: List[Dict[str, Any]],
        tables_lookup: Dict[str, Dict[str, Any]],
        output_path: str,
        sample_rows: Optional[int] = None,
        drop_duplicates: bool = False,
        handle_missing: str = "keep",
) -> Dict[str, Any]:
    """
    Execute the full merge pipeline:
      1. Read primary table
      2. For each related table with aggregation → aggregate
      3. Join aggregated tables to primary using relationships
      4. Save merged CSV

    Args:
        primary_path: Path to primary table CSV
        relationships: List of relationship dicts with table paths + join config
        aggregations: List of aggregation configs with table paths
        tables_lookup: {table_id: {table_name, file_path, ...}}
        output_path: Where to save the merged CSV
        sample_rows: Limit primary table rows for testing
        drop_duplicates: Remove duplicate rows after merge
        handle_missing: "keep" | "drop_rows" | "fill_zero"

    Returns:
        {
            "rows_before": 307511,
            "rows_after": 307511,
            "columns_after": 245,
            "tables_joined": 5,
            "output_path": "data/01_raw/project_id/merged_home_credit.csv"
        }
    """
    started_at = datetime.utcnow()
    logger.info(f"🚀 Starting merge pipeline: {primary_path}")

    # Step 1: Read primary table
    primary_df = pd.read_csv(primary_path, low_memory=False)
    if sample_rows:
        primary_df = primary_df.head(sample_rows)

    rows_before = len(primary_df)
    logger.info(f"📊 Primary table: {rows_before} rows, {len(primary_df.columns)} columns")

    # Step 2: Pre-compute aggregations for each related table
    aggregated_tables = {}  # table_id → aggregated DataFrame
    for agg_config in aggregations:
        table_id = agg_config["source_table_id"]
        table_info = tables_lookup.get(table_id, {})
        table_path = table_info.get("file_path")

        if not table_path or not os.path.exists(table_path):
            logger.warning(f"⚠️ Table file not found: {table_path}, skipping aggregation")
            continue

        try:
            agg_df, created_cols = compute_aggregation(
                file_path=table_path,
                group_by_column=agg_config["group_by_column"],
                column_prefix=agg_config["column_prefix"],
                features=agg_config["features"],
            )
            aggregated_tables[table_id] = agg_df
            logger.info(f"✅ Aggregated {table_info.get('table_name', table_id)}: {len(created_cols)} features")
        except Exception as e:
            logger.error(f"❌ Aggregation failed for {table_id}: {e}")

    # Step 3: Join tables using relationships
    merged_df = primary_df.copy()
    tables_joined = 0

    for rel in relationships:
        right_table_id = rel["right_table_id"]
        right_table_info = tables_lookup.get(right_table_id, {})
        left_column = rel["left_column"]
        right_column = rel["right_column"]
        join_type = rel.get("join_type", "left")

        # Use aggregated version if available, otherwise use raw table
        if right_table_id in aggregated_tables:
            right_df = aggregated_tables[right_table_id]
            logger.info(f"🔗 Joining aggregated: {right_table_info.get('table_name')} on {left_column}={right_column} ({join_type})")
        else:
            right_path = right_table_info.get("file_path")
            if not right_path or not os.path.exists(right_path):
                logger.warning(f"⚠️ Right table file not found: {right_path}, skipping join")
                continue
            right_df = pd.read_csv(right_path, low_memory=False)
            logger.info(f"🔗 Joining raw: {right_table_info.get('table_name')} on {left_column}={right_column} ({join_type})")

        try:
            # Resolve column name conflicts before merge
            overlap_cols = set(merged_df.columns) & set(right_df.columns) - {right_column}
            if overlap_cols:
                right_table_name = right_table_info.get("table_name", "right")
                right_df = right_df.rename(columns={
                    c: f"{right_table_name}_{c}" for c in overlap_cols
                })

            merged_df = merged_df.merge(
                right_df,
                left_on=left_column,
                right_on=right_column,
                how=join_type,
                suffixes=("", f"_{right_table_info.get('table_name', 'right')}")
            )
            tables_joined += 1
        except Exception as e:
            logger.error(f"❌ Join failed: {e}")

    # Step 4: Post-processing
    if drop_duplicates:
        before = len(merged_df)
        merged_df = merged_df.drop_duplicates()
        logger.info(f"🧹 Dropped {before - len(merged_df)} duplicate rows")

    if handle_missing == "drop_rows":
        before = len(merged_df)
        merged_df = merged_df.dropna()
        logger.info(f"🧹 Dropped {before - len(merged_df)} rows with missing values")
    elif handle_missing == "fill_zero":
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
        logger.info("🧹 Filled numeric NaN with 0")

    # Step 5: Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    output_size = os.path.getsize(output_path)

    duration = (datetime.utcnow() - started_at).total_seconds()

    result = {
        "rows_before": rows_before,
        "rows_after": len(merged_df),
        "columns_after": len(merged_df.columns),
        "tables_joined": tables_joined,
        "output_path": output_path,
        "output_size_bytes": output_size,
        "duration_seconds": round(duration, 2),
        "column_names": merged_df.columns.tolist(),
    }

    logger.info(
        f"✅ Merge complete: {result['rows_after']} rows × {result['columns_after']} columns "
        f"({tables_joined} tables joined) in {duration:.1f}s"
    )
    return result


# ============================================================================
# HELPERS
# ============================================================================

def _safe_serialize_list(values: list) -> list:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    result = []
    for v in values:
        if isinstance(v, (np.integer,)):
            result.append(int(v))
        elif isinstance(v, (np.floating,)):
            if math.isnan(v) or math.isinf(v):
                result.append(None)
            else:
                result.append(float(v))
        elif isinstance(v, (np.bool_,)):
            result.append(bool(v))
        elif isinstance(v, (pd.Timestamp, datetime)):
            result.append(v.isoformat())
        elif isinstance(v, (np.ndarray, pd.Series)):
            result.append(v.tolist())
        else:
            result.append(v)
    return result


def _parse_columns_metadata(columns_json: str) -> List[Dict]:
    """Parse columns_metadata JSON string safely."""
    if not columns_json:
        return []
    try:
        return json.loads(columns_json)
    except (json.JSONDecodeError, TypeError):
        return []


def _similar_column_name(name1: str, name2: str) -> bool:
    """Check if two column names are similar enough to suggest a relationship."""
    n1 = name1.lower().replace("_", "").replace("-", "")
    n2 = name2.lower().replace("_", "").replace("-", "")

    # Exact match after normalization
    if n1 == n2:
        return True

    # One contains the other
    if n1 in n2 or n2 in n1:
        return True

    return False