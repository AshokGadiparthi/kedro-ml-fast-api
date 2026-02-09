"""
Derived Datasets API
======================
FastAPI router to create derived (merged) datasets from multi-table collections.

SEPARATE from collections.py (which has the 22 wizard endpoints).
This file has only 1 endpoint: POST /derived-datasets/{collection_id}/build

Flow:
  1. Frontend wizard (collections.py) → user configures tables, relationships, aggregations
  2. User clicks "Create Derived Dataset" → hits THIS endpoint
  3. This endpoint reads config from DB → calls DerivedDatasetEngine → saves CSV + Dataset record
  4. EDA and ML Flow use the resulting Dataset record as-is

Location: app/api/derived_datasets.py

Register in main.py:
    from app.api.derived_datasets import router as derived_datasets_router
    app.include_router(derived_datasets_router, prefix="/api/derived-datasets")
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime
from typing import Optional, List
import os
import re
import json
import math
import logging
import pandas as pd

from app.core.database import get_db
from app.models.models import (
    Dataset, DatasetCollection, CollectionTable,
    TableRelationship, TableAggregation,
)
from app.core.derived_dataset_engine import DerivedDatasetEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Derived Datasets"])

# ============================================================================
# CONFIG — same as datasets.py / collections.py
# ============================================================================
KEDRO_PROJECT_PATH = "/home/ashok/work/latest/full/kedro-ml-engine-integrated"
KEDRO_RAW_DATA_DIR = os.path.join(KEDRO_PROJECT_PATH, "data", "01_raw")
os.makedirs(KEDRO_RAW_DATA_DIR, exist_ok=True)

# Map pandas dtypes → display types (for preview)
DTYPE_MAP = {
    "int8": "INTEGER", "int16": "INTEGER", "int32": "INTEGER", "int64": "INTEGER",
    "uint8": "INTEGER", "uint16": "INTEGER", "uint32": "INTEGER", "uint64": "INTEGER",
    "float16": "FLOAT", "float32": "FLOAT", "float64": "FLOAT",
    "bool": "BOOLEAN", "object": "VARCHAR", "string": "VARCHAR",
    "category": "CATEGORY", "datetime64[ns]": "DATETIME",
}


# ============================================================================
# SCHEMAS
# ============================================================================

class BuildDerivedRequest(BaseModel):
    """Request body for building a derived dataset."""
    output_name: Optional[str] = None           # Custom output file name
    sample_rows: Optional[int] = None           # Limit rows for testing
    drop_duplicates: bool = False               # Remove duplicate rows after merge
    handle_missing: str = "keep"                # "keep" | "drop_rows" | "fill_zero"


# ============================================================================
# HELPERS
# ============================================================================

def _slugify(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s-]+', '_', slug)
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug or "collection"


def _resolve_path(file_path: str) -> str:
    """Convert Kedro-relative path to absolute path."""
    if file_path and not os.path.isabs(file_path):
        return os.path.join(KEDRO_PROJECT_PATH, file_path)
    return file_path


def _get_file_size(file_path: Optional[str]) -> Optional[int]:
    if not file_path:
        return None
    try:
        abs_path = _resolve_path(file_path)
        return os.path.getsize(abs_path) if os.path.exists(abs_path) else None
    except (OSError, TypeError):
        return None


def _safe_json(v):
    """NaN/Inf → None for JSON serialization."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ============================================================================
# ENDPOINT: Build Derived Dataset
# ============================================================================

@router.post("/{collection_id}/build")
async def build_derived_dataset(
        collection_id: str,
        request: Optional[BuildDerivedRequest] = None,
        db: Session = Depends(get_db),
):
    """
    **Create a derived dataset from a configured multi-table collection.**

    Reads all configuration (tables, relationships, aggregations) from the DB,
    runs the DerivedDatasetEngine, saves merged CSV, creates a Dataset record.

    This replaces the old POST /collections/{id}/process endpoint for the
    actual merge. The old endpoint in collections.py can still be used as
    a fallback — both produce the same output.

    Returns:
        {
            "status": "success",
            "collection_id": "...",
            "merged_dataset_id": "ds-xxx",
            "merged_file_path": "data/01_raw/proj/collection/merged.csv",
            "rows_before": 307511,
            "rows_after": 307511,
            "columns_before": 122,
            "columns_after": 245,
            "columns_added": 123,
            "tables_joined": 5,
            "duration_seconds": 12.34,
            "warnings": ["ROW EXPLOSION PREVENTED: ...", ...],
            "output_columns": ["SK_ID_CURR", "TARGET", "BUREAU_AMT_sum", ...],
            "merged_file_size_bytes": 167540445
        }
    """
    if not request:
        request = BuildDerivedRequest()

    # ── Load collection ──
    coll = db.query(DatasetCollection).filter(
        DatasetCollection.id == collection_id
    ).first()
    if not coll:
        raise HTTPException(404, f"Collection '{collection_id}' not found")

    if coll.status == "processing":
        raise HTTPException(409, "Collection is already being processed")

    # ── Load tables ──
    tables = db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).all()

    primary_table = next((t for t in tables if t.id == coll.primary_table_id), None)
    if not primary_table:
        raise HTTPException(400, "No primary table selected for this collection")

    # ── Load relationships ──
    relationships_db = db.query(TableRelationship).filter(
        TableRelationship.collection_id == collection_id
    ).all()

    # ── Load aggregations ──
    aggregations_db = db.query(TableAggregation).filter(
        TableAggregation.collection_id == collection_id
    ).all()

    # ── Mark as processing ──
    coll.status = "processing"
    coll.processing_started_at = datetime.utcnow()
    coll.processing_error = None
    db.commit()

    try:
        # ── Build engine config from DB records ──
        primary_path = _resolve_path(primary_table.file_path)

        tables_lookup = {}
        for t in tables:
            tables_lookup[t.id] = {
                "table_name": t.table_name,
                "file_path": _resolve_path(t.file_path),
            }

        rel_configs = [
            {
                "left_table_id": r.left_table_id,
                "right_table_id": r.right_table_id,
                "left_column": r.left_column,
                "right_column": r.right_column,
                "join_type": r.join_type,
            }
            for r in relationships_db
        ]

        agg_configs = []
        for a in aggregations_db:
            features = json.loads(a.features) if a.features else []
            agg_configs.append({
                "source_table_id": a.source_table_id,
                "group_by_column": a.group_by_column,
                "column_prefix": a.column_prefix,
                "features": features,
            })

        # ── Output path ──
        collection_slug = _slugify(coll.name)
        output_name = request.output_name or f"merged_{coll.name.replace(' ', '_').lower()}"
        output_filename = f"{output_name}.csv"
        output_dir = os.path.join(KEDRO_RAW_DATA_DIR, coll.project_id, collection_slug)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        kedro_output_path = f"data/01_raw/{coll.project_id}/{collection_slug}/{output_filename}"

        # ── Save config snapshot ──
        config_snapshot = {
            "collection_id": collection_id,
            "engine": "DerivedDatasetEngine",
            "primary_table": primary_table.table_name,
            "target_column": coll.target_column,
            "tables": [{"id": t.id, "name": t.table_name, "role": t.role} for t in tables],
            "relationships": rel_configs,
            "aggregations": agg_configs,
            "options": {
                "sample_rows": request.sample_rows,
                "drop_duplicates": request.drop_duplicates,
                "handle_missing": request.handle_missing,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        coll.config_snapshot = json.dumps(config_snapshot)

        # ══════════════════════════════════════════════════════════
        # CALL THE ENGINE
        # ══════════════════════════════════════════════════════════
        engine = DerivedDatasetEngine(
            primary_path=primary_path,
            tables_lookup=tables_lookup,
            relationships=rel_configs,
            aggregations=agg_configs,
        )

        result = engine.build(
            output_path=output_path,
            sample_rows=request.sample_rows,
            drop_duplicates=request.drop_duplicates,
            handle_missing=request.handle_missing,
        )

        # ── Create Dataset record ──
        merged_dataset_id = str(uuid4())
        merged_dataset = Dataset(
            id=merged_dataset_id,
            name=output_name,
            project_id=coll.project_id,
            description=f"Derived dataset from collection: {coll.name}",
            file_name=output_filename,
            file_size_bytes=result.get("output_size_bytes"),
            file_path=kedro_output_path,
            created_at=datetime.now(),
            source_type="collection_merged",
            collection_id=collection_id,
        )
        db.add(merged_dataset)

        # ── Insert into collection_tables so UI tree shows merged file ──
        # Delete previous "derived" row if re-building
        db.query(CollectionTable).filter(
            CollectionTable.collection_id == collection_id,
            CollectionTable.role == "derived",
            ).delete()

        merged_table_id = str(uuid4())
        merged_ct = CollectionTable(
            id=merged_table_id,
            collection_id=collection_id,
            dataset_id=merged_dataset_id,
            table_name="Merged Dataset",
            file_name=output_filename,
            file_path=kedro_output_path,
            role="derived",
            sort_order=9999,
            row_count=result.get("rows_after"),
            column_count=result.get("columns_after"),
            file_size_bytes=result.get("output_size_bytes"),
            columns_metadata=json.dumps([]),
        )
        db.add(merged_ct)

        # ── Update collection record ──
        coll.status = "processed"
        coll.current_step = 5
        coll.merged_dataset_id = merged_dataset_id
        coll.merged_file_path = kedro_output_path
        coll.rows_before_merge = result.get("rows_before")
        coll.rows_after_merge = result.get("rows_after")
        coll.columns_after_merge = result.get("columns_after")
        coll.processing_completed_at = datetime.utcnow()
        coll.processing_duration_seconds = result.get("duration_seconds")

        db.commit()
        logger.info(f"✅ Derived dataset built: {merged_dataset_id} "
                    f"(collection_table: {merged_table_id})")

        return {
            "status": "success",
            "collection_id": collection_id,
            "merged_dataset_id": merged_dataset_id,
            "merged_table_id": merged_table_id,
            "merged_file_path": kedro_output_path,
            "rows_before": result.get("rows_before"),
            "rows_after": result.get("rows_after"),
            "columns_before": result.get("columns_before"),
            "columns_after": result.get("columns_after"),
            "columns_added": result.get("columns_added"),
            "tables_joined": result.get("tables_joined"),
            "duration_seconds": result.get("duration_seconds"),
            "warnings": result.get("warnings", []),
            "output_columns": result.get("column_names", []),
            "merged_file_size_bytes": _get_file_size(kedro_output_path),
        }

    except Exception as e:
        coll.status = "failed"
        coll.processing_error = str(e)
        coll.processing_completed_at = datetime.utcnow()
        db.commit()

        logger.error(f"❌ Derived dataset build failed: {e}")
        raise HTTPException(500, f"Build failed: {str(e)}")


# ============================================================================
# ENDPOINT: Preview Derived Dataset
# ============================================================================

@router.get("/{collection_id}/preview")
async def preview_derived_dataset(
        collection_id: str,
        rows: int = Query(20, ge=1, le=500),
        db: Session = Depends(get_db),
):
    """
    Preview the derived dataset CSV (first N rows).
    Only works after a successful build.
    """
    coll = db.query(DatasetCollection).filter(
        DatasetCollection.id == collection_id
    ).first()
    if not coll:
        raise HTTPException(404, f"Collection '{collection_id}' not found")

    if coll.status != "processed" or not coll.merged_file_path:
        raise HTTPException(400, "No derived dataset yet. Run POST /build first.")

    abs_path = _resolve_path(coll.merged_file_path)
    if not abs_path or not os.path.exists(abs_path):
        raise HTTPException(404, "Merged file not found on disk")

    try:
        df = pd.read_csv(abs_path, nrows=rows, low_memory=False)
    except Exception as e:
        raise HTTPException(500, f"Failed to read merged file: {e}")

    columns = []
    for col_name in df.columns:
        col = df[col_name]
        dtype_str = str(col.dtype)
        columns.append({
            "name": col_name,
            "dtype": dtype_str,
            "display_type": DTYPE_MAP.get(dtype_str, "VARCHAR"),
            "nullable": bool(col.isnull().any()),
        })

    rows_data = []
    for record in df.to_dict("records"):
        rows_data.append({k: _safe_json(v) for k, v in record.items()})

    return {
        "collection_id": collection_id,
        "merged_dataset_id": coll.merged_dataset_id,
        "total_rows": coll.rows_after_merge or len(df),
        "total_columns": coll.columns_after_merge or len(df.columns),
        "preview_rows": len(rows_data),
        "columns": columns,
        "rows": rows_data,
    }


# ============================================================================
# ENDPOINT: Get Build Status / Warnings
# ============================================================================

@router.get("/{collection_id}/status")
async def get_build_status(
        collection_id: str,
        db: Session = Depends(get_db),
):
    """
    Check the build status and retrieve warnings from the last build.
    """
    coll = db.query(DatasetCollection).filter(
        DatasetCollection.id == collection_id
    ).first()
    if not coll:
        raise HTTPException(404, f"Collection '{collection_id}' not found")

    # Parse config snapshot for warnings (engine saves them in result,
    # but we only store config_snapshot in DB — warnings are in API response)
    config = {}
    if coll.config_snapshot:
        try:
            config = json.loads(coll.config_snapshot)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "collection_id": collection_id,
        "status": coll.status,
        "merged_dataset_id": coll.merged_dataset_id,
        "merged_file_path": coll.merged_file_path,
        "rows_before": coll.rows_before_merge,
        "rows_after": coll.rows_after_merge,
        "columns_after": coll.columns_after_merge,
        "processing_started_at": coll.processing_started_at.isoformat() if coll.processing_started_at else None,
        "processing_completed_at": coll.processing_completed_at.isoformat() if coll.processing_completed_at else None,
        "processing_duration_seconds": coll.processing_duration_seconds,
        "processing_error": coll.processing_error,
        "engine": config.get("engine", "execute_merge_pipeline"),
        "merged_file_size_bytes": _get_file_size(coll.merged_file_path),
    }