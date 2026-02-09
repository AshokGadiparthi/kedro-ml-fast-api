"""
Derived Dataset Engine
========================
Pure merge engine: multi-table config → ONE clean merged CSV.

SEPARATE from collection_processor.py (which does wizard helpers).
This file ONLY does the merge — no DB, no FastAPI, no imports from other app files.

5 bugs fixed vs old execute_merge_pipeline in collection_processor.py:
  FIX 1 — Row explosion      (307K × 5.5 = 1.7M rows silently)
  FIX 2 — Duplicate key col  (left_on != right_on → two key columns in output)
  FIX 3 — Type mismatch      (int64 vs float64 from NaN CSVs → 0 matches)
  FIX 4 — Double rename      (rename + suffixes= → "bureau_amt_bureau")
  FIX 5 — Orphaned agg       (aggregation computed but never joined)

Location: app/core/derived_dataset_engine.py
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# AGGREGATION FUNCTION MAP
# ============================================================================

AGG_FUNC_MAP = {
    "sum": "sum", "mean": "mean", "median": "median",
    "max": "max", "min": "min", "count": "count",
    "unique_count": "nunique", "nunique": "nunique",
    "std": "std", "variance": "var", "var": "var",
    "first": "first", "last": "last",
}


# ============================================================================
# THE ENGINE
# ============================================================================

class DerivedDatasetEngine:
    """
    Input:  primary CSV path + relationships + aggregations + table file paths
    Output: one merged CSV file + result dict with warnings

    No database. No FastAPI. Pure pandas.
    """

    def __init__(
            self,
            primary_path: str,
            tables_lookup: Dict[str, Dict[str, Any]],
            relationships: List[Dict[str, Any]],
            aggregations: List[Dict[str, Any]],
    ):
        self.primary_path = primary_path
        self.tables_lookup = tables_lookup
        self.relationships = relationships
        self.aggregations = aggregations
        self.warnings: List[str] = []

    # ================================================================
    # PUBLIC — build the derived dataset
    # ================================================================

    def build(
            self,
            output_path: str,
            sample_rows: Optional[int] = None,
            drop_duplicates: bool = False,
            handle_missing: str = "keep",
    ) -> Dict[str, Any]:
        """
        Full pipeline: aggregate → join → post-process → save CSV.

        Returns dict with rows_before, rows_after, columns_after,
        column_names, warnings, output_path, etc.
        """
        started_at = datetime.utcnow()
        self.warnings = []
        logger.info(f"🚀 DerivedDatasetEngine.build() — {self.primary_path}")

        # Step 1: Read primary
        primary_df = self._read_primary(sample_rows)
        rows_before = len(primary_df)
        columns_before = len(primary_df.columns)

        # Step 2: Pre-compute aggregations
        aggregated_tables = self._run_aggregations()

        # Step 3: Detect orphaned aggregations (FIX 5)
        self._check_orphaned_aggregations(aggregated_tables)

        # Step 4: Join everything to primary
        merged_df, tables_joined = self._run_joins(primary_df, aggregated_tables)

        # Step 5: Post-process
        merged_df = self._post_process(merged_df, drop_duplicates, handle_missing)

        # Step 6: Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        output_size = os.path.getsize(output_path)
        duration = (datetime.utcnow() - started_at).total_seconds()

        result = {
            "rows_before": rows_before,
            "rows_after": len(merged_df),
            "columns_before": columns_before,
            "columns_after": len(merged_df.columns),
            "columns_added": len(merged_df.columns) - columns_before,
            "tables_joined": tables_joined,
            "output_path": output_path,
            "output_size_bytes": output_size,
            "duration_seconds": round(duration, 2),
            "column_names": merged_df.columns.tolist(),
            "warnings": self.warnings,
        }

        logger.info(
            f"✅ Merge complete: {result['rows_after']:,} rows × "
            f"{result['columns_after']} cols "
            f"({tables_joined} tables) in {duration:.1f}s"
        )
        if self.warnings:
            for w in self.warnings:
                logger.warning(f"  ⚠️ {w}")

        return result

    # ────────────────────────────────────────────────────────────────
    # STEP 1: Read primary table
    # ────────────────────────────────────────────────────────────────

    def _read_primary(self, sample_rows: Optional[int]) -> pd.DataFrame:
        if not os.path.exists(self.primary_path):
            raise FileNotFoundError(f"Primary table not found: {self.primary_path}")

        df = pd.read_csv(self.primary_path, low_memory=False)
        if sample_rows:
            df = df.head(sample_rows)

        logger.info(f"📊 Primary: {len(df):,} rows × {len(df.columns)} cols")
        return df

    # ────────────────────────────────────────────────────────────────
    # STEP 2: Aggregations
    # ────────────────────────────────────────────────────────────────

    def _run_aggregations(self) -> Dict[str, pd.DataFrame]:
        """Run all configured aggregations. Returns {table_id: agg_df}."""
        aggregated = {}

        for agg_cfg in self.aggregations:
            table_id = agg_cfg["source_table_id"]
            info = self.tables_lookup.get(table_id, {})
            path = info.get("file_path")
            name = info.get("table_name", table_id)

            if not path or not os.path.exists(path):
                self.warnings.append(f"Aggregation skipped: file not found for '{name}'")
                continue

            try:
                agg_df, created = self._aggregate_one_table(
                    file_path=path,
                    group_by=agg_cfg["group_by_column"],
                    prefix=agg_cfg["column_prefix"],
                    features=agg_cfg["features"],
                    table_name=name,
                )
                aggregated[table_id] = agg_df
                logger.info(f"✅ Aggregated '{name}': "
                            f"{len(agg_df):,} groups, {len(created)} features")
            except Exception as e:
                self.warnings.append(f"Aggregation failed for '{name}': {e}")
                logger.error(f"❌ Aggregation failed for '{name}': {e}")

        return aggregated

    def _aggregate_one_table(
            self, file_path: str, group_by: str,
            prefix: str, features: List[Dict[str, Any]],
            table_name: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """GROUP BY one table → flat feature columns."""
        df = pd.read_csv(file_path, low_memory=False)

        if group_by not in df.columns:
            raise ValueError(
                f"Group-by column '{group_by}' not found in '{table_name}'. "
                f"Available: {df.columns.tolist()[:10]}..."
            )

        agg_dict = {}
        for feat in features:
            col = feat["column"]
            if col not in df.columns:
                self.warnings.append(
                    f"Column '{col}' not found in '{table_name}', skipping"
                )
                continue

            funcs = []
            for fn in feat["functions"]:
                fn_str = fn if isinstance(fn, str) else fn.value
                pandas_fn = AGG_FUNC_MAP.get(fn_str)
                if pandas_fn:
                    funcs.append(pandas_fn)
                else:
                    self.warnings.append(f"Unknown agg function '{fn_str}', skipping")
            if funcs:
                agg_dict[col] = funcs

        if not agg_dict:
            raise ValueError(f"No valid aggregation features for '{table_name}'")

        grouped = df.groupby(group_by).agg(agg_dict)

        # Flatten MultiIndex → "BUREAU_AMT_CREDIT_sum"
        created = []
        new_names = []
        for col, func in grouped.columns:
            col_name = f"{prefix}{col}_{func}"
            col_name = col_name.replace("_nunique", "_unique_count")
            col_name = col_name.replace("_var", "_variance")
            new_names.append(col_name)
            created.append(col_name)

        grouped.columns = new_names
        grouped = grouped.reset_index()
        return grouped, created

    # ────────────────────────────────────────────────────────────────
    # FIX 5: Orphan check
    # ────────────────────────────────────────────────────────────────

    def _check_orphaned_aggregations(self, aggregated: Dict[str, pd.DataFrame]):
        """Warn if aggregation computed but no relationship will use it."""
        rel_right_ids = {r["right_table_id"] for r in self.relationships}
        for table_id in aggregated:
            if table_id not in rel_right_ids:
                name = self.tables_lookup.get(table_id, {}).get("table_name", table_id)
                self.warnings.append(
                    f"ORPHANED AGGREGATION: '{name}' was aggregated but no "
                    f"relationship references it — won't be joined. "
                    f"Either add a relationship or remove the aggregation."
                )

    # ────────────────────────────────────────────────────────────────
    # STEP 4: Joins  (FIX 1, 2, 3, 4)
    # ────────────────────────────────────────────────────────────────

    def _run_joins(
            self, primary_df: pd.DataFrame,
            aggregated: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, int]:
        """Join each relationship onto the primary."""
        merged = primary_df.copy()
        count = 0

        for rel in self.relationships:
            right_id = rel["right_table_id"]
            right_info = self.tables_lookup.get(right_id, {})
            left_col = rel["left_column"]
            right_col = rel["right_column"]
            join_type = rel.get("join_type", "left")
            right_name = right_info.get("table_name", "right")

            # Get right DataFrame
            if right_id in aggregated:
                right_df = aggregated[right_id]
                is_agg = True
            else:
                right_path = right_info.get("file_path")
                if not right_path or not os.path.exists(right_path):
                    self.warnings.append(f"Join skipped: file not found for '{right_name}'")
                    continue
                right_df = pd.read_csv(right_path, low_memory=False)
                is_agg = False

            logger.info(
                f"🔗 Joining {'aggregated' if is_agg else 'raw'} "
                f"'{right_name}' on {left_col}={right_col} ({join_type})"
            )

            # FIX 1: Row explosion protection
            right_df = self._guard_row_explosion(right_df, right_col, right_name, is_agg)

            # FIX 3: Type coercion
            merged, right_df = self._coerce_key_types(merged, right_df, left_col, right_col)

            # FIX 4: Clean column naming (NO suffixes param)
            right_df = self._rename_overlapping_columns(merged, right_df, left_col, right_col, right_name)

            # Execute merge
            try:
                merged = merged.merge(
                    right_df,
                    left_on=left_col,
                    right_on=right_col,
                    how=join_type,
                )

                # FIX 2: Drop duplicate join-key column
                if left_col != right_col and right_col in merged.columns:
                    merged = merged.drop(columns=[right_col])

                count += 1
                logger.info(f"   ✅ Joined: {len(merged):,} rows × {len(merged.columns)} cols")

            except Exception as e:
                self.warnings.append(f"Join failed for '{right_name}': {e}")
                logger.error(f"❌ Join failed for '{right_name}': {e}")

        return merged, count

    # ────────────────────────────────────────────────────────────────
    # FIX 1: Row explosion guard
    # ────────────────────────────────────────────────────────────────

    def _guard_row_explosion(
            self, right_df: pd.DataFrame,
            right_col: str, right_name: str, is_aggregated: bool,
    ) -> pd.DataFrame:
        """
        bureau = 1.7M rows, SK_ID_CURR not unique (avg 5.5 per applicant)
        Raw LEFT JOIN with 307K primary → 307K × 5.5 = 1.7M rows!
        Fix: auto-dedup, warn user to add aggregation.
        """
        if is_aggregated:
            return right_df

        if right_col not in right_df.columns:
            return right_df

        n_unique = right_df[right_col].nunique()
        n_rows = len(right_df)

        if n_unique >= n_rows:
            return right_df

        ratio = n_rows / max(n_unique, 1)
        self.warnings.append(
            f"ROW EXPLOSION PREVENTED: '{right_name}' has {n_rows:,} rows "
            f"but only {n_unique:,} unique keys in '{right_col}' "
            f"(avg {ratio:.1f} rows/key). Kept first row per key. "
            f"Add an aggregation (sum, mean, count) for proper features."
        )
        return right_df.drop_duplicates(subset=[right_col], keep="first")

    # ────────────────────────────────────────────────────────────────
    # FIX 3: Type coercion
    # ────────────────────────────────────────────────────────────────

    def _coerce_key_types(
            self, left_df: pd.DataFrame, right_df: pd.DataFrame,
            left_col: str, right_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        NaN in CSV int column → float64. Primary int64 vs bureau float64 → 0 matches.
        Fix: align dtypes before merge.
        """
        if left_col not in left_df.columns or right_col not in right_df.columns:
            return left_df, right_df

        l_dtype = left_df[left_col].dtype
        r_dtype = right_df[right_col].dtype

        if l_dtype == r_dtype:
            return left_df, right_df

        logger.info(f"   Type coercion: {left_col}({l_dtype}) vs {right_col}({r_dtype})")

        if pd.api.types.is_integer_dtype(l_dtype) and pd.api.types.is_float_dtype(r_dtype):
            left_df = left_df.copy()
            left_df[left_col] = left_df[left_col].astype(float)
        elif pd.api.types.is_float_dtype(l_dtype) and pd.api.types.is_integer_dtype(r_dtype):
            right_df = right_df.copy()
            right_df[right_col] = right_df[right_col].astype(float)
        elif pd.api.types.is_numeric_dtype(l_dtype) != pd.api.types.is_numeric_dtype(r_dtype):
            left_df = left_df.copy()
            left_df[left_col] = left_df[left_col].astype(str)
            right_df = right_df.copy()
            right_df[right_col] = right_df[right_col].astype(str)

        return left_df, right_df

    # ────────────────────────────────────────────────────────────────
    # FIX 4: Column naming
    # ────────────────────────────────────────────────────────────────

    def _rename_overlapping_columns(
            self, left_df: pd.DataFrame, right_df: pd.DataFrame,
            left_col: str, right_col: str, right_name: str,
    ) -> pd.DataFrame:
        """
        Old bug: rename overlaps + suffixes=() → "bureau_amt_bureau".
        Fix: explicit rename only, NO suffixes on merge().
        """
        left_cols = set(left_df.columns)
        right_cols = set(right_df.columns)

        if left_col == right_col:
            overlap = left_cols & right_cols - {right_col}
        else:
            overlap = left_cols & right_cols

        if overlap:
            rename_map = {c: f"{right_name}_{c}" for c in overlap}
            right_df = right_df.rename(columns=rename_map)
            logger.info(f"   Renamed {len(overlap)} overlapping cols with '{right_name}_' prefix")

        return right_df

    # ────────────────────────────────────────────────────────────────
    # STEP 5: Post-processing
    # ────────────────────────────────────────────────────────────────

    def _post_process(
            self, df: pd.DataFrame,
            drop_duplicates: bool, handle_missing: str,
    ) -> pd.DataFrame:

        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"🧹 Dropped {dropped:,} duplicate rows")

        if handle_missing == "drop_rows":
            before = len(df)
            df = df.dropna()
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"🧹 Dropped {dropped:,} rows with missing values")
        elif handle_missing == "fill_zero":
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].fillna(0)
            logger.info("🧹 Filled numeric NaN with 0")

        return df