"""
Kedro Source Code Download API
==============================
Clone the Kedro ML engine from GitHub, patch catalog.yml and parameters.yml
based on single-table or multi-table (collection) mode, return as zip.

Endpoints:
  Single table (no collection_id):
    GET /api/v1/kedro-source/download?project_id=abc&file_path=data/01_raw/abc/data.csv

  Multi table (with collection_id):
    GET /api/v1/kedro-source/download?project_id=abc&file_path=data/01_raw/abc/m1/application_train.csv&collection_id=e273...

Flow:
  1. Clone repo from GitHub
  2. Patch catalog.yml  → raw_data.filepath = file_path
  3. If collection_id:
     - Fetch collection metadata (tables, relationships, aggregations) from DB
     - Generate multi-table data_loading section
     - Replace parameters.yml with multi-table version
  4. Zip & return
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional

import yaml

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import (
    DatasetCollection, CollectionTable, TableRelationship, TableAggregation,
)

logger = logging.getLogger(__name__)
router = APIRouter()

GITHUB_REPO_URL = "https://github.com/AshokGadiparthi/kedro-ml-engine-integrated.git"
CATALOG_REL_PATH = "conf/base/catalog.yml"
PARAMS_REL_PATH = "conf/base/parameters.yml"


# ═══════════════════════════════════════════════════════════════════════════════
# CATALOG PATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_catalog(catalog_path: Path, raw_data_filepath: str) -> None:
    """
    Replace raw_data filepath in catalog.yml.

    BEFORE:  filepath: data/01_raw/data.csv
    AFTER:   filepath: <raw_data_filepath>
    """
    text = catalog_path.read_text(encoding="utf-8")

    pattern = r"(raw_data:\s*\n\s*type:\s*[^\n]+\n\s*filepath:\s*)([^\n]+)"
    patched, count = re.subn(pattern, rf"\g<1>{raw_data_filepath}", text, count=1)

    if count == 0:
        pattern2 = r"(filepath:\s*)data/01_raw/[^\n]+"
        patched, count = re.subn(pattern2, rf"\g<1>{raw_data_filepath}", text, count=1)

    if count == 0:
        raise ValueError(f"Could not find raw_data filepath in {catalog_path}")

    catalog_path.write_text(patched, encoding="utf-8")
    logger.info(f"✅ Patched catalog.yml: raw_data.filepath → {raw_data_filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-TABLE: Collection → data_loading YAML
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_generate_aggregations(
        tables: list,
        primary_table,
        table_join_cols: dict,
) -> list:
    """
    Auto-generate aggregation configs for related tables when none are
    configured in the DB.

    For each related (non-primary) table:
      1. Parse columns_metadata
      2. Skip ID/key columns (columns ending in _ID, or marked is_potential_key)
      3. Numeric columns (int64, float64) → ["sum", "mean", "max"]
      4. Categorical columns (str/object)  → ["nunique"]
      5. group_by = the join column for this table
      6. prefix   = TABLE_NAME_ (uppercased, shortened)

    Limits to top ~8 most useful columns per table to avoid explosion.
    """

    # Columns to always skip (by pattern)
    SKIP_PATTERNS = {"_id", "sk_id", "index"}

    # Dtype → agg functions mapping
    NUMERIC_DTYPES = {"int64", "float64", "int32", "float32", "number"}
    CATEGORICAL_DTYPES = {"str", "object", "string", "category"}

    # Numeric agg functions
    NUMERIC_FUNCS = ["sum", "mean", "max"]
    CATEGORICAL_FUNCS = ["nunique"]

    # Max columns to aggregate per table
    MAX_FEATURES_PER_TABLE = 10

    agg_configs = []

    for t in tables:
        # Skip primary table — we don't aggregate it
        if t.id == primary_table.id:
            continue

        # Parse columns metadata
        cols = []
        try:
            raw = t.columns_metadata
            cols = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (json.JSONDecodeError, TypeError):
            continue

        if not cols:
            continue

        # Determine the group_by column for this table
        group_by = table_join_cols.get(t.id)
        if not group_by:
            continue  # Can't aggregate without a join column

        # Build features dict
        features = {}
        numeric_count = 0
        cat_count = 0

        for c in cols:
            col_name = c.get("name", "")
            dtype = (c.get("dtype") or "").lower()
            is_key = c.get("is_potential_key", False)

            # Skip ID/key columns
            if is_key:
                continue
            if any(pat in col_name.lower() for pat in SKIP_PATTERNS):
                continue

            # Skip columns that are 100% null
            null_pct = c.get("null_percentage", 0) or 0
            if null_pct >= 100:
                continue

            # Skip single-value columns (no variance)
            unique_count = c.get("unique_count", 0) or 0
            if unique_count <= 1:
                continue

            # Determine aggregation functions by dtype
            if dtype in NUMERIC_DTYPES:
                if numeric_count < MAX_FEATURES_PER_TABLE:
                    features[col_name] = NUMERIC_FUNCS.copy()
                    numeric_count += 1
            elif dtype in CATEGORICAL_DTYPES:
                if cat_count < 3:  # Limit categorical to avoid explosion
                    features[col_name] = CATEGORICAL_FUNCS.copy()
                    cat_count += 1

        if not features:
            continue

        # Build prefix from table name
        # e.g. "bureau" → "BUREAU_", "credit_card_balance" → "CC_BAL_"
        prefix = _make_prefix(t.table_name)

        agg_configs.append({
            "table": t.table_name,
            "group_by": group_by,
            "prefix": prefix,
            "features": features,
        })

    return agg_configs


def _make_prefix(table_name: str) -> str:
    """
    Generate a short uppercase prefix from a table name.

    Examples:
      "bureau"                 → "BUREAU_"
      "previous_application"   → "PREV_APP_"
      "credit_card_balance"    → "CC_BAL_"
      "POS_CASH_balance"       → "POS_CASH_"
      "installments_payments"  → "INST_PAY_"
    """
    # Common abbreviation patterns
    ABBREVS = {
        "bureau": "BUREAU",
        "previous_application": "PREV_APP",
        "credit_card_balance": "CC_BAL",
        "pos_cash_balance": "POS_CASH",
        "installments_payments": "INST_PAY",
        "application_train": "APP",
        "application_test": "APP_TEST",
    }

    lower = table_name.lower()
    if lower in ABBREVS:
        return ABBREVS[lower] + "_"

    # Fallback: take first 3 chars of each word, uppercase
    parts = re.split(r"[_\s]+", table_name)
    if len(parts) == 1:
        return table_name.upper()[:8] + "_"
    else:
        short = "_".join(p.upper()[:4] for p in parts[:3])
        return short + "_"


def _build_data_loading_from_collection(
        db: Session,
        collection_id: str,
        file_path: str,
) -> dict:
    """
    Fetch collection metadata from DB and build the data_loading dict
    that will be injected into parameters.yml.

    Maps:
      collection.tables         → data_loading.tables
      collection.relationships  → data_loading.joins
      collection.aggregations   → data_loading.aggregations
      collection.target_column  → data_loading.target_column
    """

    # ── Fetch from DB ───────────────────────────────────────────────────────
    coll = db.query(DatasetCollection).filter(
        DatasetCollection.id == collection_id
    ).first()
    if not coll:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")

    tables = db.query(CollectionTable).filter(
        CollectionTable.collection_id == collection_id
    ).order_by(CollectionTable.sort_order).all()

    relationships = db.query(TableRelationship).filter(
        TableRelationship.collection_id == collection_id
    ).all()

    aggregations = db.query(TableAggregation).filter(
        TableAggregation.collection_id == collection_id
    ).all()

    if not tables:
        raise HTTPException(status_code=400, detail="Collection has no tables")

    # ── Build id→name lookup ────────────────────────────────────────────────
    id_to_name = {t.id: t.table_name for t in tables}

    # ── Find primary table ──────────────────────────────────────────────────
    primary = next((t for t in tables if t.role == "primary"), None)
    if not primary:
        primary = next((t for t in tables if t.id == coll.primary_table_id), tables[0])

    # ── Derive data_directory from file_path ────────────────────────────────
    # e.g. "data/01_raw/proj123/m1/application_train.csv" → "data/01_raw/proj123/m1/"
    data_directory = str(Path(file_path).parent) + "/"

    # ── Build table_id → join_column from relationships ─────────────────────
    table_join_cols = {}
    for rel in relationships:
        if rel.left_table_id not in table_join_cols:
            table_join_cols[rel.left_table_id] = rel.left_column
        if rel.right_table_id not in table_join_cols:
            table_join_cols[rel.right_table_id] = rel.right_column

    # Fallback: scan columns_metadata for potential key columns
    for t in tables:
        if t.id not in table_join_cols and t.columns_metadata:
            try:
                cols = json.loads(t.columns_metadata) if isinstance(t.columns_metadata, str) else t.columns_metadata
                for c in (cols or []):
                    if c.get("is_potential_key"):
                        table_join_cols[t.id] = c["name"]
                        break
            except (json.JSONDecodeError, TypeError):
                pass

    # ── STEP 3: Build tables list ───────────────────────────────────────────
    tables_config = []
    for t in tables:
        id_col = table_join_cols.get(t.id, "ID")
        tables_config.append({
            "name": t.table_name,
            "filepath": t.file_name or (Path(t.file_path).name if t.file_path else "unknown.csv"),
            "id_column": id_col,
        })

    # ── STEP 4: Build aggregations ──────────────────────────────────────────
    agg_config = []

    if aggregations:
        # ── 4a: Use DB-configured aggregations ──────────────────────────
        for agg in aggregations:
            source_name = id_to_name.get(agg.source_table_id, "unknown")

            features_raw = []
            try:
                features_raw = json.loads(agg.features) if isinstance(agg.features, str) else (agg.features or [])
            except (json.JSONDecodeError, TypeError):
                pass

            features_dict = {}
            for f in features_raw:
                col_name = f.get("column", "")
                funcs = f.get("functions", [])
                if col_name and funcs:
                    features_dict[col_name] = funcs

            agg_config.append({
                "table": source_name,
                "group_by": agg.group_by_column,
                "prefix": agg.column_prefix or f"{source_name.upper()}_",
                "features": features_dict,
            })
    else:
        # ── 4b: Auto-generate aggregations from column metadata ─────────
        # When no aggregations are configured in the wizard, generate
        # sensible defaults for every related (non-primary) table:
        #   - numeric columns  → ["sum", "mean", "max"]
        #   - categorical cols → ["nunique"]
        #   - skip ID/key columns
        logger.info("📊 No aggregations in DB — auto-generating from column metadata")
        agg_config = _auto_generate_aggregations(tables, primary, table_join_cols)

    # ── STEP 5: Build joins from relationships ──────────────────────────────
    joins_config = []
    for rel in relationships:
        left_name = id_to_name.get(rel.left_table_id, "unknown")
        right_name = id_to_name.get(rel.right_table_id, "unknown")
        joins_config.append({
            "left_table": left_name,
            "right_table": right_name,
            "left_on": rel.left_column,
            "right_on": rel.right_column,
            "how": rel.join_type or "left",
        })

    # ── Assemble data_loading ───────────────────────────────────────────────
    target = coll.target_column or "TARGET"

    data_loading = {
        "mode": "multi",
        "data_directory": data_directory,
        "main_table": primary.table_name,
        "target_column": target,
        "tables": tables_config,
    }

    if agg_config:
        data_loading["aggregations"] = agg_config

    if joins_config:
        data_loading["joins"] = joins_config

    data_loading["test_size"] = 0.2
    data_loading["random_state"] = 42
    data_loading["stratify"] = True

    logger.info(
        f"✅ Built data_loading from collection: "
        f"tables={len(tables_config)}, joins={len(joins_config)}, "
        f"aggregations={len(agg_config)}, main_table={primary.table_name}"
    )

    return data_loading, target, data_directory


def _patch_parameters_for_multi_table(
        params_path: Path,
        data_loading: dict,
        target_column: str,
        data_directory: str,
) -> None:
    """
    Read existing parameters.yml, replace/inject the data_loading section
    with multi-table config, update related top-level keys.
    """
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    # ── Replace data_loading section ────────────────────────────────────────
    params["data_loading"] = data_loading

    # ── Update top-level keys ───────────────────────────────────────────────
    params["target_column"] = target_column
    params["data_path"] = data_directory

    # ── Adjust feature_engineering for multi-table ──────────────────────────
    fe = params.get("feature_engineering", {})
    fe["max_features_allowed"] = 500
    id_kw = fe.get("id_keywords", [])
    if "sk_id" not in id_kw:
        id_kw.append("sk_id")
    fe["id_keywords"] = id_kw
    params["feature_engineering"] = fe

    # ── Adjust feature_selection ────────────────────────────────────────────
    fs = params.get("feature_selection", {})
    fs["n_features"] = 30
    params["feature_selection"] = fs

    # ── Write back ──────────────────────────────────────────────────────────
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"✅ Patched parameters.yml → mode=multi, target={target_column}")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-TABLE: Patch parameters.yml
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_target_column(
        target_column: Optional[str],
        model_name: Optional[str],
) -> str:
    """
    Determine the target column name.

    Priority:
      1. Explicit target_column param
      2. Parsed from model_name (e.g. "sample_data__loan_approved" → "loan_approved")
      3. Fallback: "TARGET"
    """
    if target_column:
        return target_column.strip()

    if model_name and "__" in model_name:
        # "sample_data__loan_approved" → "loan_approved"
        parts = model_name.split("__", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()

    return "TARGET"


def _patch_parameters_for_single_table(
        params_path: Path,
        file_path: str,
        target: str,
) -> None:
    """
    Patch parameters.yml for single-table mode:
      - data_loading.filepath  → file_path
      - data_loading.target_column → target
      - data_path → file_path
      - target_column → target
    """
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    # Patch data_loading section
    dl = params.get("data_loading", {})
    dl["mode"] = "single"
    dl["filepath"] = file_path
    dl["target_column"] = target
    params["data_loading"] = dl

    # Patch top-level keys
    params["data_path"] = file_path
    params["target_column"] = target

    with open(params_path, "w", encoding="utf-8") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"✅ Patched parameters.yml: mode=single, filepath={file_path}, target={target}")


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/download")
async def download_kedro_source(
        background_tasks: BackgroundTasks,
        project_id: str = Query(
            ...,
            description="Project ID for naming and directory structure",
            example="my_project",
        ),
        file_path: str = Query(
            "data/01_raw/data.csv",
            description="Raw data filepath for catalog.yml (e.g. data/01_raw/proj/data.csv)",
            example="data/01_raw/my_project/data.csv",
        ),
        collection_id: Optional[str] = Query(
            None,
            description="Collection ID for multi-table mode. When provided, fetches collection "
                        "metadata and generates multi-table parameters.yml with tables, joins, "
                        "and aggregations.",
            example="e273e5e5-0f17-4e87-84ae-1f19add9156a",
        ),
        target_column: Optional[str] = Query(
            None,
            description="Target column name for prediction. If not provided, "
                        "auto-detected from model name (e.g. 'sample_data__loan_approved' → 'loan_approved').",
            example="loan_approved",
        ),
        model_name: Optional[str] = Query(
            None,
            description="Model name (used to derive target_column if not provided). "
                        "Format: dataset__target_column",
            example="sample_data__loan_approved",
        ),
        branch: str = Query(
            "main",
            description="Git branch to clone",
            example="main",
        ),
        db: Session = Depends(get_db),
):
    """
    Download Kedro ML engine source code with project-specific configuration.

    **Always patches both `catalog.yml` AND `parameters.yml`.**

    **Single table** (no collection_id):
      - `catalog.yml`: raw_data.filepath → file_path
      - `parameters.yml`: data_loading.filepath → file_path, target_column → resolved target

    **Multi table** (collection_id provided):
      - `catalog.yml`: raw_data.filepath → file_path
      - `parameters.yml`: full multi-table config from collection metadata

    Returns the repo as a `.zip` download.
    """
    # ── Sanitize inputs ─────────────────────────────────────────────────────
    safe_project_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", project_id.strip())
    if not safe_project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

    safe_file_path = file_path.strip()

    # ── If multi-table, fetch collection data early (fail fast) ─────────────
    multi_table_data = None
    if collection_id:
        multi_table_data = _build_data_loading_from_collection(
            db, collection_id.strip(), safe_file_path
        )
        logger.info(f"📊 Multi-table mode: collection={collection_id}")

    tmp_dir = tempfile.mkdtemp(prefix="kedro_source_")

    try:
        # ── STEP 1: Clone the repo ──────────────────────────────────────
        clone_dir = os.path.join(tmp_dir, "kedro-ml-engine-integrated")
        logger.info(f"📥 Cloning {GITHUB_REPO_URL} (branch={branch})...")

        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch,
             GITHUB_REPO_URL, clone_dir],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to clone repository: {result.stderr.strip()}",
            )
        logger.info(f"✅ Cloned to {clone_dir}")

        # ── STEP 2: Patch catalog.yml ───────────────────────────────────
        catalog_path = Path(clone_dir) / CATALOG_REL_PATH
        if not catalog_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"catalog.yml not found at {CATALOG_REL_PATH}",
            )

        _patch_catalog(catalog_path, safe_file_path)

        # ── STEP 3: Patch parameters.yml ────────────────────────────────
        params_path = Path(clone_dir) / PARAMS_REL_PATH
        if not params_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"parameters.yml not found at {PARAMS_REL_PATH}",
            )

        if multi_table_data:
            # Multi-table: full replacement with collection metadata
            data_loading, resolved_target, data_directory = multi_table_data
            _patch_parameters_for_multi_table(
                params_path, data_loading, resolved_target, data_directory
            )
        else:
            # Single-table: patch filepath + target_column
            resolved_target = _resolve_target_column(target_column, model_name)
            _patch_parameters_for_single_table(
                params_path, safe_file_path, resolved_target
            )

        # ── STEP 4: Create raw data directories ─────────────────────────
        raw_dir = Path(clone_dir) / Path(safe_file_path).parent
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {raw_dir}")

        # ── STEP 5: Remove .git folder ──────────────────────────────────
        git_dir = Path(clone_dir) / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)

        # ── STEP 6: Create zip ──────────────────────────────────────────
        zip_name = f"kedro-ml-engine-{safe_project_id}"
        zip_path = os.path.join(tmp_dir, zip_name)
        shutil.make_archive(
            zip_path, "zip",
            root_dir=tmp_dir,
            base_dir="kedro-ml-engine-integrated",
        )

        zip_file = f"{zip_path}.zip"
        logger.info(f"📦 Created zip: {zip_file} ({os.path.getsize(zip_file)} bytes)")

        # ── STEP 7: Return as download ──────────────────────────────────
        background_tasks.add_task(shutil.rmtree, tmp_dir, True)

        return FileResponse(
            path=zip_file,
            media_type="application/zip",
            filename=f"{zip_name}.zip",
            headers={
                "Content-Disposition": f'attachment; filename="{zip_name}.zip"'
            },
        )

    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=504, detail="Git clone timed out (120s)")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error(f"❌ Error in kedro source download: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))