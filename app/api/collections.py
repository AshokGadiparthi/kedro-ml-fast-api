"""
Multi-Table Dataset Collections API
=====================================
22 endpoints powering the 5-step Multi-Table Dataset Wizard.

Endpoint Map:
─────────────────────────────────────────────────────────────
  COLLECTION CRUD
    POST   /                              → Create collection + upload files (Step 1)
    GET    /                              → List all collections
    GET    /{id}                          → Full collection detail
    PUT    /{id}                          → Update metadata
    DELETE /{id}                          → Delete collection (cascade)

  TABLES (Step 1 & 2)
    GET    /{id}/tables                   → List tables with columns
    POST   /{id}/tables/upload            → Add more files to collection
    GET    /{id}/tables/{tid}/columns     → Get columns for one table
    DELETE /{id}/tables/{tid}             → Remove table from collection
    PUT    /{id}/primary                  → Set primary table + target column

  RELATIONSHIPS (Step 3)
    GET    /{id}/relationships            → List all relationships
    POST   /{id}/relationships            → Create relationship
    PUT    /{id}/relationships/{rid}      → Update relationship
    DELETE /{id}/relationships/{rid}      → Delete relationship
    GET    /{id}/relationships/suggest    → Auto-suggest join keys
    POST   /{id}/relationships/{rid}/validate → Validate a relationship

  AGGREGATIONS (Step 4)
    GET    /{id}/aggregations             → List all aggregations
    POST   /{id}/aggregations             → Create aggregation
    PUT    /{id}/aggregations/{aid}       → Update aggregation
    DELETE /{id}/aggregations/{aid}       → Delete aggregation

  PROCESSING (Step 5)
    GET    /{id}/review                   → Full review summary
    POST   /{id}/process                  → Start processing
    GET    /{id}/preview                  → Preview merged dataset
─────────────────────────────────────────────────────────────
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from typing import Optional, List
import os
import json
import logging
import pandas as pd

from app.core.database import get_db
from app.models.models import Dataset
from app.models.models import (
    DatasetCollection, CollectionTable, TableRelationship, TableAggregation
)
from app.schemas.collection_schema import (
    CreateCollectionRequest, UpdateCollectionRequest,
    CollectionSummary, CollectionDetail, CollectionCreateResponse,
    TableSummary, TableDetail, ColumnInfo,
    SetPrimaryTableRequest, PrimaryTableResponse,
    CreateRelationshipRequest, UpdateRelationshipRequest,
    RelationshipResponse, RelationshipValidation, SuggestedRelationship,
    CreateAggregationRequest, UpdateAggregationRequest,
    AggregationResponse, AggregationFeature,
    ReviewSummary, ProcessRequest, ProcessStatusResponse, MergedPreviewResponse,
    CollectionStatus, TableRole, JoinType,
)
from app.core.collection_processor import (
    introspect_csv, validate_relationship, suggest_relationships,
    compute_aggregation, compute_created_columns, execute_merge_pipeline,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Dataset Collections"])

# ============================================================================
# CONFIG — same as datasets.py
# ============================================================================
KEDRO_PROJECT_PATH = "/home/ashok/work/latest/full/kedro-ml-engine-integrated"
KEDRO_RAW_DATA_DIR = os.path.join(KEDRO_PROJECT_PATH, "data", "01_raw")
os.makedirs(KEDRO_RAW_DATA_DIR, exist_ok=True)

import math

def _safe_json(v):
    """Make a value JSON-safe (handle NaN/Inf)."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ============================================================================
# HELPERS
# ============================================================================

def _slugify_name(name: str) -> str:
    """Convert collection name to a safe folder name.
    'Home Credit Risk' → 'home_credit_risk'
    """
    import re
    slug = name.strip().lower()
    slug = re.sub(r'[^\w\s-]', '', slug)     # Remove special chars
    slug = re.sub(r'[\s-]+', '_', slug)       # Spaces/dashes → underscores
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug or "collection"


def _get_collection_or_404(db: Session, collection_id: str) -> DatasetCollection:
    coll = db.query(DatasetCollection).filter(DatasetCollection.id == collection_id).first()
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_id}' not found")
    return coll


def _get_table_or_404(db: Session, collection_id: str, table_id: str) -> CollectionTable:
    table = db.query(CollectionTable).filter(
        CollectionTable.id == table_id,
        CollectionTable.collection_id == collection_id
    ).first()
    if not table:
        raise HTTPException(status_code=404, detail=f"Table '{table_id}' not found in collection")
    return table


def _resolve_file_path(file_path: str) -> str:
    """Convert Kedro-relative path to absolute path."""
    if file_path and not os.path.isabs(file_path):
        return os.path.join(KEDRO_PROJECT_PATH, file_path)
    return file_path


def _format_dt(dt) -> Optional[str]:
    if dt:
        return dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
    return None


def _get_file_size(file_path: Optional[str]) -> Optional[int]:
    """Get file size in bytes, returns None if file doesn't exist."""
    if not file_path:
        return None
    try:
        abs_path = _resolve_file_path(file_path)
        return os.path.getsize(abs_path) if os.path.exists(abs_path) else None
    except (OSError, TypeError):
        return None


def _parse_columns(metadata_json: Optional[str]) -> List[dict]:
    if not metadata_json:
        return []
    try:
        return json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        return []


def _build_table_summary(t: CollectionTable) -> dict:
    return {
        "id": t.id,
        "table_name": t.table_name,
        "file_name": t.file_name,
        "role": t.role,
        "row_count": t.row_count,
        "column_count": t.column_count,
        "file_size_bytes": t.file_size_bytes,
        "dataset_id": t.dataset_id,
    }


def _build_table_detail(t: CollectionTable) -> dict:
    result = _build_table_summary(t)
    result["columns"] = _parse_columns(t.columns_metadata)
    result["file_path"] = t.file_path
    result["created_at"] = _format_dt(t.created_at)
    return result


def _build_relationship_response(rel: TableRelationship, db: Session) -> dict:
    left_table = db.query(CollectionTable).filter(CollectionTable.id == rel.left_table_id).first()
    right_table = db.query(CollectionTable).filter(CollectionTable.id == rel.right_table_id).first()

    validation = None
    if rel.is_validated:
        validation = {
            "is_validated": True,
            "relationship_type": rel.relationship_type,
            "left_unique_count": rel.left_unique_count,
            "right_unique_count": rel.right_unique_count,
            "match_count": rel.match_count,
            "match_percentage": rel.match_percentage,
            "orphan_left_count": rel.orphan_left_count,
            "orphan_right_count": rel.orphan_right_count,
            "warnings": [],
        }

    left_name = left_table.table_name if left_table else None
    right_name = right_table.table_name if right_table else None
    preview = None
    if left_name and right_name:
        preview = f"{left_name}.{rel.left_column} = {right_name}.{rel.right_column}"

    return {
        "id": rel.id,
        "collection_id": rel.collection_id,
        "left_table_id": rel.left_table_id,
        "right_table_id": rel.right_table_id,
        "left_table_name": left_name,
        "right_table_name": right_name,
        "left_column": rel.left_column,
        "right_column": rel.right_column,
        "left_column_dtype": rel.left_column_dtype,
        "right_column_dtype": rel.right_column_dtype,
        "join_type": rel.join_type,
        "validation": validation,
        "preview_sql": preview,
        "created_at": _format_dt(rel.created_at),
        "updated_at": _format_dt(rel.updated_at),
    }


def _build_aggregation_response(agg: TableAggregation, db: Session) -> dict:
    source_table = db.query(CollectionTable).filter(CollectionTable.id == agg.source_table_id).first()

    features = []
    try:
        features = json.loads(agg.features) if agg.features else []
    except (json.JSONDecodeError, TypeError):
        features = []

    created_cols = []
    try:
        created_cols = json.loads(agg.created_columns) if agg.created_columns else []
    except (json.JSONDecodeError, TypeError):
        created_cols = []

    return {
        "id": agg.id,
        "collection_id": agg.collection_id,
        "source_table_id": agg.source_table_id,
        "source_table_name": source_table.table_name if source_table else None,
        "group_by_column": agg.group_by_column,
        "column_prefix": agg.column_prefix,
        "features": features,
        "created_columns": created_cols,
        "output_column_count": agg.output_column_count or 0,
        "created_at": _format_dt(agg.created_at),
        "updated_at": _format_dt(agg.updated_at),
    }


def _update_collection_counts(db: Session, collection_id: str):
    """Recalculate total_tables, total_relationships, total_aggregations."""
    coll = db.query(DatasetCollection).filter(DatasetCollection.id == collection_id).first()
    if coll:
        coll.total_tables = db.query(CollectionTable).filter(
            CollectionTable.collection_id == collection_id).count()
        coll.total_relationships = db.query(TableRelationship).filter(
            TableRelationship.collection_id == collection_id).count()
        coll.total_aggregations = db.query(TableAggregation).filter(
            TableAggregation.collection_id == collection_id).count()
        db.commit()


# ============================================================================
# COLLECTION CRUD
# ============================================================================

@router.post("/", status_code=201)
async def create_collection(
        name: str = Form(...),
        project_id: str = Form(...),
        description: Optional[str] = Form(None),
        files: List[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    """
    **Step 1: Create collection and upload CSV files.**

    This is the primary entry point for the Multi-Table Wizard.
    Accepts a collection name, project ID, and multiple CSV files.

    For each file:
      1. Saves to Kedro raw data directory
      2. Creates a Dataset record (reusable independently)
      3. Creates a CollectionTable link
      4. Auto-detects column types and stats

    The first file is auto-marked as "primary" (user can change in Step 2).
    """
    logger.info(f"📦 Creating collection: '{name}' with {len(files)} files")

    if not files:
        raise HTTPException(status_code=400, detail="At least one CSV file is required")

    collection_id = str(uuid4())

    # Create the collection record
    collection = DatasetCollection(
        id=collection_id,
        name=name,
        description=description,
        project_id=project_id,
        status="draft",
        current_step=1,
        total_tables=len(files),
    )
    db.add(collection)

    # Create project/collection directory
    collection_slug = _slugify_name(name)
    project_dir = os.path.join(KEDRO_RAW_DATA_DIR, project_id, collection_slug)
    os.makedirs(project_dir, exist_ok=True)

    tables_response = []

    for idx, file in enumerate(files):
        if not file.filename:
            continue

        file_id = str(uuid4())
        table_id = str(uuid4())
        original_filename = file.filename
        table_name = os.path.splitext(original_filename)[0]  # "application_train.csv" → "application_train"
        role = "primary" if idx == 0 else "related"

        # Save file to disk
        full_path = os.path.join(project_dir, original_filename)
        contents = await file.read()
        with open(full_path, "wb") as f:
            f.write(contents)

        kedro_path = f"data/01_raw/{project_id}/{collection_slug}/{original_filename}"
        logger.info(f"  💾 Saved: {original_filename} ({len(contents)} bytes)")

        # Introspect CSV
        introspection = {"row_count": 0, "column_count": 0, "columns": []}
        try:
            introspection = introspect_csv(full_path)
        except Exception as e:
            logger.warning(f"  ⚠️ Introspection failed for {original_filename}: {e}")

        # Create Dataset record (tagged as collection member — won't show in standalone list)
        dataset = Dataset(
            id=file_id,
            name=table_name,
            project_id=project_id,
            description=f"Part of collection: {name}",
            file_name=original_filename,
            file_size_bytes=len(contents),
            file_path=kedro_path,
            created_at=datetime.now(),
            source_type="collection_member",
            collection_id=collection_id,
        )
        db.add(dataset)

        # Create CollectionTable link
        ct = CollectionTable(
            id=table_id,
            collection_id=collection_id,
            dataset_id=file_id,
            table_name=table_name,
            file_name=original_filename,
            file_path=kedro_path,
            role=role,
            sort_order=idx,
            row_count=introspection.get("row_count"),
            column_count=introspection.get("column_count"),
            file_size_bytes=len(contents),
            columns_metadata=json.dumps(introspection.get("columns", [])),
        )
        db.add(ct)

        # Auto-set primary table
        if idx == 0:
            collection.primary_table_id = table_id

        tables_response.append({
            "id": table_id,
            "table_name": table_name,
            "file_name": original_filename,
            "role": role,
            "row_count": introspection.get("row_count"),
            "column_count": introspection.get("column_count"),
            "file_size_bytes": len(contents),
            "dataset_id": file_id,
        })

    db.commit()
    db.refresh(collection)

    logger.info(f"✅ Collection created: {collection_id} with {len(tables_response)} tables")

    return {
        "id": collection_id,
        "name": name,
        "description": description,
        "project_id": project_id,
        "status": "draft",
        "total_tables": len(tables_response),
        "tables": tables_response,
        "created_at": _format_dt(collection.created_at),
    }


@router.get("/")
async def list_collections(
        project_id: Optional[str] = Query(None, description="Filter by project"),
        status: Optional[str] = Query(None, description="Filter by status"),
        db: Session = Depends(get_db)
):
    """List all collections, optionally filtered by project and/or status."""
    query = db.query(DatasetCollection)
    if project_id:
        query = query.filter(DatasetCollection.project_id == project_id)
    if status:
        query = query.filter(DatasetCollection.status == status)

    query = query.order_by(DatasetCollection.created_at.desc())
    collections = query.all()

    return [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "project_id": c.project_id,
            "status": c.status,
            "current_step": c.current_step,
            "total_tables": c.total_tables,
            "total_relationships": c.total_relationships,
            "total_aggregations": c.total_aggregations,
            "merged_dataset_id": c.merged_dataset_id,
            "rows_after_merge": c.rows_after_merge,
            "columns_after_merge": c.columns_after_merge,
            "created_at": _format_dt(c.created_at),
            "updated_at": _format_dt(c.updated_at),
        }
        for c in collections
    ]


@router.get("/{collection_id}")
async def get_collection(collection_id: str, db: Session = Depends(get_db)):
    """Get complete collection detail with all tables, relationships, and aggregations."""
    coll = _get_collection_or_404(db, collection_id)

    tables = db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).order_by(CollectionTable.sort_order).all()

    relationships = db.query(TableRelationship).filter(
        TableRelationship.collection_id == collection_id
    ).all()

    aggregations = db.query(TableAggregation).filter(
        TableAggregation.collection_id == collection_id
    ).all()

    return {
        "id": coll.id,
        "name": coll.name,
        "description": coll.description,
        "project_id": coll.project_id,
        "status": coll.status,
        "current_step": coll.current_step,
        "primary_table_id": coll.primary_table_id,
        "target_column": coll.target_column,
        "tables": [_build_table_detail(t) for t in tables],
        "relationships": [_build_relationship_response(r, db) for r in relationships],
        "aggregations": [_build_aggregation_response(a, db) for a in aggregations],
        "merged_dataset_id": coll.merged_dataset_id,
        "merged_file_path": coll.merged_file_path,
        "merged_file_size_bytes": _get_file_size(coll.merged_file_path),
        "rows_before_merge": coll.rows_before_merge,
        "rows_after_merge": coll.rows_after_merge,
        "columns_after_merge": coll.columns_after_merge,
        "processing_duration_seconds": coll.processing_duration_seconds,
        "total_tables": coll.total_tables,
        "total_relationships": coll.total_relationships,
        "total_aggregations": coll.total_aggregations,
        "created_at": _format_dt(coll.created_at),
        "updated_at": _format_dt(coll.updated_at),
    }


@router.put("/{collection_id}")
async def update_collection(
        collection_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        current_step: Optional[int] = None,
        db: Session = Depends(get_db)
):
    """Update collection metadata (name, description, wizard step)."""
    coll = _get_collection_or_404(db, collection_id)

    if name is not None:
        coll.name = name
    if description is not None:
        coll.description = description
    if current_step is not None:
        if 1 <= current_step <= 5:
            coll.current_step = current_step

    db.commit()
    db.refresh(coll)

    return {"message": "Collection updated", "id": coll.id, "name": coll.name, "current_step": coll.current_step}


@router.delete("/{collection_id}", status_code=200)
async def delete_collection(collection_id: str, db: Session = Depends(get_db)):
    """
    Delete a collection and all its configuration.

    NOTE: Does NOT delete the underlying CSV files or Dataset records —
    those remain independently accessible.
    """
    coll = _get_collection_or_404(db, collection_id)

    # Cascade deletes handle tables, relationships, aggregations
    db.query(TableAggregation).filter(TableAggregation.collection_id == collection_id).delete()
    db.query(TableRelationship).filter(TableRelationship.collection_id == collection_id).delete()
    db.query(CollectionTable).filter(CollectionTable.collection_id == collection_id).delete()
    db.delete(coll)
    db.commit()

    logger.info(f"🗑️ Deleted collection: {collection_id}")
    return {"message": "Collection deleted", "id": collection_id}


# ============================================================================
# TABLES (Step 1 & 2)
# ============================================================================

@router.get("/{collection_id}/tables")
async def list_tables(collection_id: str, db: Session = Depends(get_db)):
    """List all tables in a collection with full column metadata."""
    _get_collection_or_404(db, collection_id)

    tables = db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).order_by(CollectionTable.sort_order).all()

    return [_build_table_detail(t) for t in tables]


@router.post("/{collection_id}/tables/upload", status_code=201)
async def add_tables(
        collection_id: str,
        files: List[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    """Add more CSV files to an existing collection (the "+ Add Files" button)."""
    coll = _get_collection_or_404(db, collection_id)

    existing_count = db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).count()

    collection_slug = _slugify_name(coll.name)
    project_dir = os.path.join(KEDRO_RAW_DATA_DIR, coll.project_id, collection_slug)
    os.makedirs(project_dir, exist_ok=True)

    added = []
    for idx, file in enumerate(files):
        if not file.filename:
            continue

        file_id = str(uuid4())
        table_id = str(uuid4())
        original_filename = file.filename
        table_name = os.path.splitext(original_filename)[0]

        # Check for duplicate table name
        existing = db.query(CollectionTable).filter(
            CollectionTable.collection_id == collection_id,
            CollectionTable.table_name == table_name
        ).first()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Table '{table_name}' already exists in this collection"
            )

        full_path = os.path.join(project_dir, original_filename)
        contents = await file.read()
        with open(full_path, "wb") as f:
            f.write(contents)

        kedro_path = f"data/01_raw/{coll.project_id}/{collection_slug}/{original_filename}"

        introspection = {"row_count": 0, "column_count": 0, "columns": []}
        try:
            introspection = introspect_csv(full_path)
        except Exception as e:
            logger.warning(f"  ⚠️ Introspection failed for {original_filename}: {e}")

        dataset = Dataset(
            id=file_id,
            name=table_name,
            project_id=coll.project_id,
            description=f"Part of collection: {coll.name}",
            file_name=original_filename,
            file_size_bytes=len(contents),
            file_path=kedro_path,
            created_at=datetime.now(),
            source_type="collection_member",
            collection_id=collection_id,
        )
        db.add(dataset)

        ct = CollectionTable(
            id=table_id,
            collection_id=collection_id,
            dataset_id=file_id,
            table_name=table_name,
            file_name=original_filename,
            file_path=kedro_path,
            role="related",
            sort_order=existing_count + idx,
            row_count=introspection.get("row_count"),
            column_count=introspection.get("column_count"),
            file_size_bytes=len(contents),
            columns_metadata=json.dumps(introspection.get("columns", [])),
        )
        db.add(ct)
        added.append(_build_table_summary(ct))

    _update_collection_counts(db, collection_id)
    db.commit()

    return {"message": f"{len(added)} table(s) added", "tables": added}


@router.get("/{collection_id}/tables/{table_id}/columns")
async def get_table_columns(
        collection_id: str, table_id: str,
        db: Session = Depends(get_db)
):
    """Get detailed column metadata for a specific table (powers the relationship dialog dropdowns)."""
    table = _get_table_or_404(db, collection_id, table_id)

    columns = _parse_columns(table.columns_metadata)

    # If no cached metadata, introspect live
    if not columns and table.file_path:
        try:
            abs_path = _resolve_file_path(table.file_path)
            result = introspect_csv(abs_path)
            columns = result.get("columns", [])
            # Cache for future use
            table.columns_metadata = json.dumps(columns)
            table.row_count = result.get("row_count")
            table.column_count = result.get("column_count")
            db.commit()
        except Exception as e:
            logger.warning(f"Live introspection failed: {e}")

    return {
        "table_id": table_id,
        "table_name": table.table_name,
        "row_count": table.row_count,
        "column_count": table.column_count,
        "columns": columns,
    }


@router.delete("/{collection_id}/tables/{table_id}")
async def remove_table(
        collection_id: str, table_id: str,
        db: Session = Depends(get_db)
):
    """Remove a table from the collection (does NOT delete the underlying CSV)."""
    table = _get_table_or_404(db, collection_id, table_id)
    coll = _get_collection_or_404(db, collection_id)

    # Clean up related relationships
    db.query(TableRelationship).filter(
        TableRelationship.collection_id == collection_id,
        (TableRelationship.left_table_id == table_id) | (TableRelationship.right_table_id == table_id)
    ).delete(synchronize_session=False)

    # Clean up related aggregations
    db.query(TableAggregation).filter(
        TableAggregation.collection_id == collection_id,
        TableAggregation.source_table_id == table_id
    ).delete(synchronize_session=False)

    # If this was the primary table, clear it
    if coll.primary_table_id == table_id:
        coll.primary_table_id = None

    db.delete(table)
    _update_collection_counts(db, collection_id)
    db.commit()

    return {"message": f"Table '{table.table_name}' removed from collection"}


@router.put("/{collection_id}/primary")
async def set_primary_table(
        collection_id: str,
        request: SetPrimaryTableRequest,
        db: Session = Depends(get_db)
):
    """
    **Step 2: Set the primary table and optionally the target column.**

    Marks the selected table as "primary" and all others as "related".
    """
    coll = _get_collection_or_404(db, collection_id)

    # Validate the table belongs to this collection
    primary_table = _get_table_or_404(db, collection_id, request.primary_table_id)

    # Validate target column exists in the table
    if request.target_column:
        columns = _parse_columns(primary_table.columns_metadata)
        col_names = {c["name"] for c in columns}
        if request.target_column not in col_names:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in table '{primary_table.table_name}'. Available: {sorted(col_names)}"
            )

    # Reset all tables to "related"
    db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).update({"role": "related"})

    # Set the selected table as primary
    primary_table.role = "primary"
    coll.primary_table_id = request.primary_table_id
    coll.target_column = request.target_column
    coll.current_step = max(coll.current_step or 1, 2)

    db.commit()

    tables = db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).order_by(CollectionTable.sort_order).all()

    return {
        "collection_id": collection_id,
        "primary_table_id": primary_table.id,
        "primary_table_name": primary_table.table_name,
        "target_column": coll.target_column,
        "tables": [_build_table_summary(t) for t in tables],
    }


# ============================================================================
# RELATIONSHIPS (Step 3)
# ============================================================================

@router.get("/{collection_id}/relationships")
async def list_relationships(collection_id: str, db: Session = Depends(get_db)):
    """List all relationships in a collection."""
    _get_collection_or_404(db, collection_id)

    rels = db.query(TableRelationship).filter(
        TableRelationship.collection_id == collection_id
    ).all()

    return [_build_relationship_response(r, db) for r in rels]


@router.post("/{collection_id}/relationships", status_code=201)
async def create_relationship(
        collection_id: str,
        request: CreateRelationshipRequest,
        db: Session = Depends(get_db)
):
    """
    **Step 3: Create a join relationship between two tables.**

    Automatically validates the join keys and detects the relationship type.
    """
    coll = _get_collection_or_404(db, collection_id)

    # Validate both tables belong to this collection
    left_table = _get_table_or_404(db, collection_id, request.left_table_id)
    right_table = _get_table_or_404(db, collection_id, request.right_table_id)

    if request.left_table_id == request.right_table_id:
        raise HTTPException(status_code=400, detail="Cannot create self-join")

    # Check duplicate
    existing = db.query(TableRelationship).filter(
        TableRelationship.collection_id == collection_id,
        TableRelationship.left_table_id == request.left_table_id,
        TableRelationship.right_table_id == request.right_table_id,
        TableRelationship.left_column == request.left_column,
        TableRelationship.right_column == request.right_column,
        ).first()
    if existing:
        raise HTTPException(status_code=409, detail="This relationship already exists")

    # Validate columns exist
    left_cols = _parse_columns(left_table.columns_metadata)
    right_cols = _parse_columns(right_table.columns_metadata)
    left_col_names = {c["name"] for c in left_cols}
    right_col_names = {c["name"] for c in right_cols}

    if request.left_column not in left_col_names:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{request.left_column}' not found in '{left_table.table_name}'"
        )
    if request.right_column not in right_col_names:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{request.right_column}' not found in '{right_table.table_name}'"
        )

    # Get column types
    left_dtype = next((c.get("display_type") for c in left_cols if c["name"] == request.left_column), None)
    right_dtype = next((c.get("display_type") for c in right_cols if c["name"] == request.right_column), None)

    rel_id = str(uuid4())
    rel = TableRelationship(
        id=rel_id,
        collection_id=collection_id,
        left_table_id=request.left_table_id,
        right_table_id=request.right_table_id,
        left_column=request.left_column,
        right_column=request.right_column,
        join_type=request.join_type.value,
        left_column_dtype=left_dtype,
        right_column_dtype=right_dtype,
    )

    # Auto-validate if files are accessible
    try:
        left_path = _resolve_file_path(left_table.file_path)
        right_path = _resolve_file_path(right_table.file_path)
        if left_path and right_path and os.path.exists(left_path) and os.path.exists(right_path):
            validation = validate_relationship(left_path, right_path, request.left_column, request.right_column)
            rel.is_validated = validation.get("is_validated", False)
            rel.relationship_type = validation.get("relationship_type")
            rel.left_unique_count = validation.get("left_unique_count")
            rel.right_unique_count = validation.get("right_unique_count")
            rel.match_count = validation.get("match_count")
            rel.match_percentage = validation.get("match_percentage")
            rel.orphan_left_count = validation.get("orphan_left_count")
            rel.orphan_right_count = validation.get("orphan_right_count")
    except Exception as e:
        logger.warning(f"Auto-validation failed: {e}")

    db.add(rel)
    coll.current_step = max(coll.current_step or 1, 3)
    _update_collection_counts(db, collection_id)
    db.commit()
    db.refresh(rel)

    return _build_relationship_response(rel, db)


@router.put("/{collection_id}/relationships/{relationship_id}")
async def update_relationship(
        collection_id: str,
        relationship_id: str,
        request: UpdateRelationshipRequest,
        db: Session = Depends(get_db)
):
    """Update a relationship's join columns or join type."""
    _get_collection_or_404(db, collection_id)

    rel = db.query(TableRelationship).filter(
        TableRelationship.id == relationship_id,
        TableRelationship.collection_id == collection_id
    ).first()
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")

    if request.left_column is not None:
        rel.left_column = request.left_column
    if request.right_column is not None:
        rel.right_column = request.right_column
    if request.join_type is not None:
        rel.join_type = request.join_type.value

    # Re-validate after update
    rel.is_validated = False
    try:
        left_table = db.query(CollectionTable).filter(CollectionTable.id == rel.left_table_id).first()
        right_table = db.query(CollectionTable).filter(CollectionTable.id == rel.right_table_id).first()
        if left_table and right_table:
            left_path = _resolve_file_path(left_table.file_path)
            right_path = _resolve_file_path(right_table.file_path)
            if left_path and right_path and os.path.exists(left_path) and os.path.exists(right_path):
                validation = validate_relationship(left_path, right_path, rel.left_column, rel.right_column)
                rel.is_validated = validation.get("is_validated", False)
                rel.relationship_type = validation.get("relationship_type")
                rel.match_count = validation.get("match_count")
                rel.match_percentage = validation.get("match_percentage")
                rel.orphan_left_count = validation.get("orphan_left_count")
                rel.orphan_right_count = validation.get("orphan_right_count")
    except Exception as e:
        logger.warning(f"Re-validation failed: {e}")

    db.commit()
    db.refresh(rel)

    return _build_relationship_response(rel, db)


@router.delete("/{collection_id}/relationships/{relationship_id}")
async def delete_relationship(
        collection_id: str, relationship_id: str,
        db: Session = Depends(get_db)
):
    """Delete a relationship."""
    rel = db.query(TableRelationship).filter(
        TableRelationship.id == relationship_id,
        TableRelationship.collection_id == collection_id
    ).first()
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")

    db.delete(rel)
    _update_collection_counts(db, collection_id)
    db.commit()

    return {"message": "Relationship deleted", "id": relationship_id}


@router.get("/{collection_id}/relationships/suggest")
async def suggest_joins(collection_id: str, db: Session = Depends(get_db)):
    """
    **Auto-suggest join keys based on column name matching.**

    Analyzes all tables in the collection and suggests relationships
    where columns have matching names (e.g. "SK_ID_CURR" in both tables).
    """
    _get_collection_or_404(db, collection_id)

    tables = db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).all()

    table_dicts = [
        {
            "id": t.id,
            "table_name": t.table_name,
            "file_path": t.file_path,
            "role": t.role,
            "columns_metadata": t.columns_metadata,
        }
        for t in tables
    ]

    suggestions = suggest_relationships(table_dicts)
    return suggestions


@router.post("/{collection_id}/relationships/{relationship_id}/validate")
async def validate_relationship_endpoint(
        collection_id: str, relationship_id: str,
        db: Session = Depends(get_db)
):
    """Manually trigger validation for a relationship (recalculate match stats)."""
    rel = db.query(TableRelationship).filter(
        TableRelationship.id == relationship_id,
        TableRelationship.collection_id == collection_id
    ).first()
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")

    left_table = db.query(CollectionTable).filter(CollectionTable.id == rel.left_table_id).first()
    right_table = db.query(CollectionTable).filter(CollectionTable.id == rel.right_table_id).first()

    if not left_table or not right_table:
        raise HTTPException(status_code=404, detail="One or both tables not found")

    left_path = _resolve_file_path(left_table.file_path)
    right_path = _resolve_file_path(right_table.file_path)

    if not left_path or not right_path or not os.path.exists(left_path) or not os.path.exists(right_path):
        raise HTTPException(status_code=400, detail="Table files not accessible on disk")

    validation = validate_relationship(left_path, right_path, rel.left_column, rel.right_column)

    rel.is_validated = validation.get("is_validated", False)
    rel.relationship_type = validation.get("relationship_type")
    rel.left_unique_count = validation.get("left_unique_count")
    rel.right_unique_count = validation.get("right_unique_count")
    rel.match_count = validation.get("match_count")
    rel.match_percentage = validation.get("match_percentage")
    rel.orphan_left_count = validation.get("orphan_left_count")
    rel.orphan_right_count = validation.get("orphan_right_count")
    rel.left_column_dtype = validation.get("left_column_dtype")
    rel.right_column_dtype = validation.get("right_column_dtype")
    db.commit()

    result = _build_relationship_response(rel, db)
    result["validation"]["warnings"] = validation.get("warnings", [])
    return result


# ============================================================================
# AGGREGATIONS (Step 4)
# ============================================================================

@router.get("/{collection_id}/aggregations")
async def list_aggregations(collection_id: str, db: Session = Depends(get_db)):
    """List all aggregation configurations."""
    _get_collection_or_404(db, collection_id)

    aggs = db.query(TableAggregation).filter(
        TableAggregation.collection_id == collection_id
    ).all()

    return [_build_aggregation_response(a, db) for a in aggs]


@router.post("/{collection_id}/aggregations", status_code=201)
async def create_aggregation(
        collection_id: str,
        request: CreateAggregationRequest,
        db: Session = Depends(get_db)
):
    """
    **Step 4: Configure aggregation for a related table.**

    Specifies how to GROUP BY and aggregate columns before joining
    to the primary table.
    """
    coll = _get_collection_or_404(db, collection_id)
    source_table = _get_table_or_404(db, collection_id, request.source_table_id)

    # Validate group_by_column exists
    columns = _parse_columns(source_table.columns_metadata)
    col_names = {c["name"] for c in columns}
    if request.group_by_column not in col_names:
        raise HTTPException(
            status_code=400,
            detail=f"Group-by column '{request.group_by_column}' not found in '{source_table.table_name}'"
        )

    # Validate feature columns exist
    for feat in request.features:
        if feat.column not in col_names:
            raise HTTPException(
                status_code=400,
                detail=f"Feature column '{feat.column}' not found in '{source_table.table_name}'"
            )

    # Check for duplicate (one aggregation per table)
    existing = db.query(TableAggregation).filter(
        TableAggregation.collection_id == collection_id,
        TableAggregation.source_table_id == request.source_table_id
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Aggregation already exists for '{source_table.table_name}'. Use PUT to update."
        )

    # Compute created column names
    features_list = [{"column": f.column, "functions": [fn.value for fn in f.functions]} for f in request.features]
    created_cols = compute_created_columns(request.column_prefix, features_list)

    agg_id = str(uuid4())
    agg = TableAggregation(
        id=agg_id,
        collection_id=collection_id,
        source_table_id=request.source_table_id,
        group_by_column=request.group_by_column,
        column_prefix=request.column_prefix,
        features=json.dumps(features_list),
        created_columns=json.dumps(created_cols),
        output_column_count=len(created_cols),
    )
    db.add(agg)
    coll.current_step = max(coll.current_step or 1, 4)
    _update_collection_counts(db, collection_id)
    db.commit()
    db.refresh(agg)

    return _build_aggregation_response(agg, db)


@router.put("/{collection_id}/aggregations/{aggregation_id}")
async def update_aggregation(
        collection_id: str,
        aggregation_id: str,
        request: UpdateAggregationRequest,
        db: Session = Depends(get_db)
):
    """Update aggregation configuration."""
    _get_collection_or_404(db, collection_id)

    agg = db.query(TableAggregation).filter(
        TableAggregation.id == aggregation_id,
        TableAggregation.collection_id == collection_id
    ).first()
    if not agg:
        raise HTTPException(status_code=404, detail="Aggregation not found")

    if request.group_by_column is not None:
        agg.group_by_column = request.group_by_column
    if request.column_prefix is not None:
        agg.column_prefix = request.column_prefix
    if request.features is not None:
        features_list = [{"column": f.column, "functions": [fn.value for fn in f.functions]} for f in request.features]
        agg.features = json.dumps(features_list)

        prefix = request.column_prefix or agg.column_prefix
        created_cols = compute_created_columns(prefix, features_list)
        agg.created_columns = json.dumps(created_cols)
        agg.output_column_count = len(created_cols)

    db.commit()
    db.refresh(agg)

    return _build_aggregation_response(agg, db)


@router.delete("/{collection_id}/aggregations/{aggregation_id}")
async def delete_aggregation(
        collection_id: str, aggregation_id: str,
        db: Session = Depends(get_db)
):
    """Delete an aggregation configuration."""
    agg = db.query(TableAggregation).filter(
        TableAggregation.id == aggregation_id,
        TableAggregation.collection_id == collection_id
    ).first()
    if not agg:
        raise HTTPException(status_code=404, detail="Aggregation not found")

    db.delete(agg)
    _update_collection_counts(db, collection_id)
    db.commit()

    return {"message": "Aggregation deleted", "id": aggregation_id}


# ============================================================================
# REVIEW + PROCESS (Step 5)
# ============================================================================

@router.get("/{collection_id}/review")
async def get_review_summary(collection_id: str, db: Session = Depends(get_db)):
    """
    **Step 5: Get complete review summary before processing.**

    Returns everything the user configured, plus validation warnings
    and readiness status.
    """
    coll = _get_collection_or_404(db, collection_id)

    tables = db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).order_by(CollectionTable.sort_order).all()

    relationships = db.query(TableRelationship).filter(
        TableRelationship.collection_id == collection_id
    ).all()

    aggregations = db.query(TableAggregation).filter(
        TableAggregation.collection_id == collection_id
    ).all()

    # Build warnings
    warnings = []
    primary_table = None

    if not coll.primary_table_id:
        warnings.append("⚠️ No primary table selected")
    else:
        primary_table = next((t for t in tables if t.id == coll.primary_table_id), None)

    if len(tables) < 2:
        warnings.append("⚠️ Need at least 2 tables for multi-table processing")

    if not relationships:
        warnings.append("⚠️ No relationships defined — tables won't be joined")

    # Check related tables without aggregation
    related_tables = [t for t in tables if t.role == "related"]
    agg_table_ids = {a.source_table_id for a in aggregations}
    tables_with_relationships = set()
    for r in relationships:
        tables_with_relationships.add(r.right_table_id)

    for rt in related_tables:
        if rt.id in tables_with_relationships and rt.id not in agg_table_ids:
            warnings.append(
                f"⚠️ Table '{rt.table_name}' has a relationship but no aggregation configured — "
                f"will be joined as-is (may produce many-to-many rows)"
            )

    # Check low match percentages
    for r in relationships:
        if r.is_validated and r.match_percentage is not None and r.match_percentage < 50:
            left_t = next((t for t in tables if t.id == r.left_table_id), None)
            right_t = next((t for t in tables if t.id == r.right_table_id), None)
            warnings.append(
                f"⚠️ Low join match ({r.match_percentage}%) between "
                f"'{left_t.table_name if left_t else '?'}' and '{right_t.table_name if right_t else '?'}'"
            )

    # Compute estimated columns
    total_input_rows = sum(t.row_count or 0 for t in tables)
    total_input_columns = sum(t.column_count or 0 for t in tables)

    primary_cols = primary_table.column_count if primary_table else 0
    agg_cols = sum(a.output_column_count or 0 for a in aggregations)
    estimated_output_columns = primary_cols + agg_cols

    ready = (
            coll.primary_table_id is not None
            and len(tables) >= 2
            and len(relationships) > 0
    )

    return {
        "collection": {
            "id": coll.id,
            "name": coll.name,
            "description": coll.description,
            "project_id": coll.project_id,
            "status": coll.status,
            "current_step": coll.current_step,
            "total_tables": len(tables),
            "total_relationships": len(relationships),
            "total_aggregations": len(aggregations),
            "created_at": _format_dt(coll.created_at),
            "updated_at": _format_dt(coll.updated_at),
        },
        "primary_table": _build_table_detail(primary_table) if primary_table else None,
        "target_column": coll.target_column,
        "tables": [_build_table_summary(t) for t in tables],
        "relationships": [_build_relationship_response(r, db) for r in relationships],
        "aggregations": [_build_aggregation_response(a, db) for a in aggregations],
        "total_input_rows": total_input_rows,
        "total_input_columns": total_input_columns,
        "estimated_output_columns": estimated_output_columns,
        "warnings": warnings,
        "ready_to_process": ready,
    }


@router.post("/{collection_id}/process")
async def process_collection(
        collection_id: str,
        request: Optional[ProcessRequest] = None,
        db: Session = Depends(get_db)
):
    """
    **Step 5: Execute the merge pipeline — "Create & Process" button.**

    Pipeline:
      1. Read primary table
      2. For each related table with aggregation → GROUP BY + aggregate
      3. Join aggregated tables to primary using defined relationships
      4. Save merged CSV to Kedro data directory
      5. Create a new Dataset record for the merged output

    Returns the processing result with merged dataset info.
    """
    coll = _get_collection_or_404(db, collection_id)

    if coll.status == "processing":
        raise HTTPException(status_code=409, detail="Collection is already being processed")

    if not request:
        request = ProcessRequest()

    # Load all configuration
    tables = db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).all()

    primary_table = next((t for t in tables if t.id == coll.primary_table_id), None)
    if not primary_table:
        raise HTTPException(status_code=400, detail="No primary table selected")

    relationships = db.query(TableRelationship).filter(
        TableRelationship.collection_id == collection_id
    ).all()

    aggregations_db = db.query(TableAggregation).filter(
        TableAggregation.collection_id == collection_id
    ).all()

    # Mark as processing
    coll.status = "processing"
    coll.processing_started_at = datetime.utcnow()
    coll.processing_error = None
    db.commit()

    try:
        # Resolve file paths
        primary_path = _resolve_file_path(primary_table.file_path)

        tables_lookup = {}
        for t in tables:
            tables_lookup[t.id] = {
                "table_name": t.table_name,
                "file_path": _resolve_file_path(t.file_path),
            }

        # Build relationship configs
        rel_configs = [
            {
                "left_table_id": r.left_table_id,
                "right_table_id": r.right_table_id,
                "left_column": r.left_column,
                "right_column": r.right_column,
                "join_type": r.join_type,
            }
            for r in relationships
        ]

        # Build aggregation configs
        agg_configs = []
        for a in aggregations_db:
            features = json.loads(a.features) if a.features else []
            agg_configs.append({
                "source_table_id": a.source_table_id,
                "group_by_column": a.group_by_column,
                "column_prefix": a.column_prefix,
                "features": features,
            })

        # Output path — stored in collection subfolder
        collection_slug = _slugify_name(coll.name)
        output_name = request.output_name or f"merged_{coll.name.replace(' ', '_').lower()}"
        output_filename = f"{output_name}.csv"
        output_dir = os.path.join(KEDRO_RAW_DATA_DIR, coll.project_id, collection_slug)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        kedro_output_path = f"data/01_raw/{coll.project_id}/{collection_slug}/{output_filename}"

        # Save config snapshot
        config_snapshot = {
            "collection_id": collection_id,
            "primary_table": primary_table.table_name,
            "target_column": coll.target_column,
            "tables": [{"id": t.id, "name": t.table_name, "role": t.role} for t in tables],
            "relationships": rel_configs,
            "aggregations": agg_configs,
            "process_options": {
                "sample_rows": request.sample_rows,
                "drop_duplicates": request.drop_duplicates,
                "handle_missing": request.handle_missing,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        coll.config_snapshot = json.dumps(config_snapshot)

        # Execute pipeline
        result = execute_merge_pipeline(
            primary_path=primary_path,
            relationships=rel_configs,
            aggregations=agg_configs,
            tables_lookup=tables_lookup,
            output_path=output_path,
            sample_rows=request.sample_rows,
            drop_duplicates=request.drop_duplicates,
            handle_missing=request.handle_missing or "keep",
        )

        # Create Dataset record for merged output (tagged — won't show in standalone list)
        merged_dataset_id = str(uuid4())
        merged_dataset = Dataset(
            id=merged_dataset_id,
            name=output_name,
            project_id=coll.project_id,
            description=f"Merged dataset from collection: {coll.name}",
            file_name=output_filename,
            file_size_bytes=result.get("output_size_bytes"),
            file_path=kedro_output_path,
            created_at=datetime.now(),
            source_type="collection_merged",
            collection_id=collection_id,
        )
        db.add(merged_dataset)

        # Update collection
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

        logger.info(f"✅ Collection processed: {collection_id}")

        return {
            "status": "processed",
            "collection_id": collection_id,
            "merged_dataset_id": merged_dataset_id,
            "merged_file_path": kedro_output_path,
            "rows_before": result.get("rows_before"),
            "rows_after": result.get("rows_after"),
            "columns_after": result.get("columns_after"),
            "tables_joined": result.get("tables_joined"),
            "duration_seconds": result.get("duration_seconds"),
            "output_columns": result.get("column_names", []),
            "merged_file_size_bytes": _get_file_size(kedro_output_path),
        }

    except Exception as e:
        coll.status = "failed"
        coll.processing_error = str(e)
        coll.processing_completed_at = datetime.utcnow()
        db.commit()

        logger.error(f"❌ Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/{collection_id}/preview")
async def preview_merged(
        collection_id: str,
        rows: int = Query(20, ge=1, le=500, description="Number of rows to preview"),
        db: Session = Depends(get_db)
):
    """
    Preview the merged dataset (only available after processing).
    Returns first N rows with column metadata.
    """
    coll = _get_collection_or_404(db, collection_id)

    if coll.status != "processed" or not coll.merged_file_path:
        raise HTTPException(
            status_code=400,
            detail="Collection has not been processed yet. Run POST /process first."
        )

    abs_path = _resolve_file_path(coll.merged_file_path)
    if not abs_path or not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="Merged file not found on disk")

    try:
        df = pd.read_csv(abs_path, nrows=rows, low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read merged file: {e}")

    # Build column info
    columns = []
    for col_name in df.columns:
        col = df[col_name]
        dtype_str = str(col.dtype)
        from app.services.collection_processor import DTYPE_MAP
        display_type = DTYPE_MAP.get(dtype_str, "VARCHAR")
        columns.append({
            "name": col_name,
            "dtype": dtype_str,
            "display_type": display_type,
            "nullable": bool(col.isnull().any()),
        })

    # Safe serialization of rows
    rows_data = []
    for record in df.to_dict("records"):
        rows_data.append({
            k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v)
            for k, v in record.items()
        })

    return {
        "collection_id": collection_id,
        "total_rows": coll.rows_after_merge or len(df),
        "total_columns": coll.columns_after_merge or len(df.columns),
        "preview_rows": len(rows_data),
        "columns": columns,
        "rows": rows_data,
    }