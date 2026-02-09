"""Datasets API Routes - Saves to Kedro Project"""
from fastapi import APIRouter, Depends, Path, UploadFile, File, Form
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
import math
import pandas as pd
import numpy as np
import io
import logging
from app.core.database import get_db
from app.models.models import Dataset
from app.schemas import DatasetCreate, DatasetResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Datasets"])

# ============================================================================
# KEDRO PROJECT PATH - Change this to your Kedro project root
# ============================================================================
KEDRO_PROJECT_PATH = "/home/ashok/work/latest/full/kedro-ml-engine-integrated"
KEDRO_RAW_DATA_DIR = os.path.join(KEDRO_PROJECT_PATH, "data", "01_raw")

# Ensure base directory exists
os.makedirs(KEDRO_RAW_DATA_DIR, exist_ok=True)

logger.info(f"✅ Kedro raw data directory: {KEDRO_RAW_DATA_DIR}")

# In-memory storage for dataset content
dataset_cache = {}


def _resolve_dataset_file(dataset) -> str:
    """
    Resolve the actual file path for a dataset.
    Priority: stored file_path (handles subfolders) → flat path fallback.
    """
    # 1. Try the stored file_path (handles collection subfolders correctly)
    if dataset.file_path:
        if os.path.isabs(dataset.file_path):
            if os.path.exists(dataset.file_path):
                return dataset.file_path
        else:
            abs_path = os.path.join(KEDRO_PROJECT_PATH, dataset.file_path)
            if os.path.exists(abs_path):
                return abs_path

    # 2. Fallback: flat path construction (for older datasets without file_path)
    if dataset.file_name and dataset.project_id:
        flat_path = os.path.join(KEDRO_RAW_DATA_DIR, dataset.project_id, dataset.file_name)
        if os.path.exists(flat_path):
            return flat_path

    return None


@router.get("/", response_model=list)
async def list_datasets(db: Session = Depends(get_db)):
    """List all standalone datasets (excludes collection members and merged outputs)."""
    datasets = db.query(Dataset).filter(
        (Dataset.source_type == None) |       # Legacy records without source_type
        (Dataset.source_type == "standalone")  # Normal uploaded datasets
    ).all()
    return [
        {
            "id": d.id,
            "name": d.name,
            "project_id": d.project_id,
            "description": d.description,
            "file_name": d.file_name,
            "file_path": d.file_path,
            "file_size_bytes": d.file_size_bytes,
            "created_at": d.created_at.isoformat() if d.created_at else ""
        }
        for d in datasets
    ]


@router.post("/")
async def create_dataset(
        file: UploadFile = File(...),
        name: str = Form(...),
        project_id: str = Form(...),
        description: Optional[str] = Form(None),
        db: Session = Depends(get_db)
):
    """
    Create dataset and save to Kedro project path

    Flow:
    1. Create project-specific directory: {KEDRO}/data/01_raw/{project_id}/
    2. Save file with original filename (overwrites if exists)
    3. Store Kedro-relative path in database
    4. Return paths for job parameter
    """

    dataset_id = str(uuid4())
    logger.info(f"📥 Uploading dataset: {name} for project: {project_id}")

    try:
        # ✅ Step 1: Create project directory structure
        project_dir = os.path.join(KEDRO_RAW_DATA_DIR, project_id)
        os.makedirs(project_dir, exist_ok=True)
        logger.info(f"📁 Created project directory: {project_dir}")

        # ✅ Step 2: Save file with original filename (allows overwrite)
        original_filename = file.filename
        full_file_path = os.path.join(project_dir, original_filename)

        contents = await file.read()
        with open(full_file_path, "wb") as f:
            f.write(contents)

        # ✅ Step 2.1: Hard validation (prevents “file not found” later)
        if not os.path.exists(full_file_path) or os.path.getsize(full_file_path) == 0:
            raise RuntimeError(f"File write failed or empty: {full_file_path}")

        logger.info(f"✅ File saved: {full_file_path}")

        # ✅ Step 3: Kedro-relative path for parameters (good for Kedro project runs)
        kedro_relative_path = f"data/01_raw/{project_id}/{original_filename}"
        logger.info(f"📍 Kedro path: {kedro_relative_path}")

        # ✅ Step 4: ANALYZE the file (only if CSV)
        row_count = 0
        column_count = 0
        try:
            if original_filename.lower().endswith(".csv"):
                df = pd.read_csv(full_file_path)
                row_count = len(df)
                column_count = len(df.columns)
                logger.info(f"✅ Analysis: {row_count} rows, {column_count} columns")
                dataset_cache[dataset_id] = df
            else:
                logger.info("ℹ️ Skipping CSV analysis (not a CSV file)")
        except Exception as e:
            logger.warning(f"⚠️ Could not analyze CSV: {e}")

        # ✅ Step 5: Create dataset record in database
        new_dataset = Dataset(
            id=dataset_id,
            name=name,
            project_id=project_id,
            description=description or "",
            file_name=original_filename,
            file_size_bytes=len(contents),
            file_path = kedro_relative_path,
            created_at=datetime.now()
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)

        logger.info(f"✅ Dataset created: {dataset_id}")

        # ✅ IMPORTANT: return both abs path and kedro path
        return {
            "id": new_dataset.id,
            "name": new_dataset.name,
            "project_id": new_dataset.project_id,
            "description": new_dataset.description,
            "file_name": new_dataset.file_name,
            "file_size_bytes": new_dataset.file_size_bytes,
            "created_at": new_dataset.created_at.isoformat(),

            # ✅ for Kedro run (relative)
            "kedro_path": kedro_relative_path,

            # ✅ for Celery/Kedro subprocess safety (absolute)
            "abs_path": full_file_path,

            # optional debug
            "row_count": row_count,
            "column_count": column_count,
        }

    except Exception as e:
        logger.error(f"❌ Error uploading dataset: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")


@router.get("/{dataset_id}/columns")
async def get_dataset_columns(
        dataset_id: str,
        db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get all column names from a dataset

    Used for: Dropdown selection in preprocessing, feature engineering, etc.

    Returns:
    {
      "dataset_id": "10f47671-d3c4-4194-ac28-1b321fbbf469",
      "dataset_name": "sample_data",
      "total_columns": 6,
      "columns": [
        {
          "name": "age",
          "dtype": "int64",
          "type": "numeric"
        },
        {
          "name": "income",
          "dtype": "float64",
          "type": "numeric"
        },
        ...
      ],
      "numeric_columns": ["age", "income", ...],
      "categorical_columns": ["category", ...],
      "datetime_columns": []
    }
    """
    try:
        logger.info(f"🔍 Fetching columns for dataset: {dataset_id}")

        # 1. Query dataset from database
        dataset_record = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if not dataset_record:
            logger.error(f"❌ Dataset not found: {dataset_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_id}' not found"
            )

        # 2. Get file path from dataset record
        file_path = None

        # Try multiple attribute names (adapt to your schema)
        for attr_name in ['kedro_path', 'file_path', 'path', 'filepath']:
            if hasattr(dataset_record, attr_name):
                potential_path = getattr(dataset_record, attr_name)
                if potential_path:
                    file_path = potential_path
                    logger.info(f"✅ Found file path in '{attr_name}': {file_path}")
                    break

        if not file_path:
            available_attrs = [k for k in dataset_record.__dict__.keys() if not k.startswith('_')]
            logger.error(f"❌ No file path attribute found")
            logger.error(f"   Available attributes: {available_attrs}")
            raise HTTPException(
                status_code=500,
                detail=f"Cannot determine file path from dataset. Available attributes: {available_attrs}"
            )

        # 3. Build full path
        full_file_path = os.path.join(str(KEDRO_PROJECT_PATH), file_path)


        logger.info(f"📂 Reading from: {full_file_path}")

        # 4. Check file exists
        if not os.path.exists(full_file_path):
            logger.error(f"❌ File not found at: {full_file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Dataset file not found at {full_file_path}"
            )

        # 5. Read CSV to get columns (only header, not full data)
        df = pd.read_csv(full_file_path, nrows=0)  # Read only header

        logger.info(f"✅ Successfully read {len(df.columns)} columns")

        # 6. Categorize columns by data type
        numeric_columns = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()

        # 7. Prepare detailed column info
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)

            if col in numeric_columns:
                col_type = "numeric"
            elif col in categorical_columns:
                col_type = "categorical"
            elif col in datetime_columns:
                col_type = "datetime"
            else:
                col_type = "other"

            columns_info.append({
                "name": col,
                "dtype": dtype,
                "type": col_type
            })

        # 8. Return response
        response = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_record.name if hasattr(dataset_record, 'name') else "Unknown",
            "file_path": str(file_path),
            "total_columns": len(df.columns),
            "columns": columns_info,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "status": "success"
        }

        logger.info(f"✅ Returning {len(df.columns)} columns for dataset {dataset_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching columns: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching dataset columns: {str(e)}"
        )


@router.get("/{dataset_id}/info")
async def get_dataset_info(
        dataset_id: str,
        db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed dataset information (columns + statistics)

    Returns: All column info plus shape, memory, missing values, etc.
    """
    try:
        logger.info(f"📊 Fetching detailed info for dataset: {dataset_id}")

        # Get columns first
        dataset_record = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset_record:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

        # Get file path
        file_path = None
        for attr_name in ['kedro_path', 'file_path', 'path', 'filepath']:
            if hasattr(dataset_record, attr_name):
                potential_path = getattr(dataset_record, attr_name)
                if potential_path:
                    file_path = potential_path
                    break

        if not file_path:
            raise HTTPException(status_code=500, detail="Cannot determine file path")

        full_file_path = os.path.join(str(KEDRO_PROJECT_PATH), file_path)

        # Read full data for statistics
        df = pd.read_csv(full_file_path)

        # Calculate statistics
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "missing_values": int(df.isnull().sum().sum()),
            "missing_percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
            "duplicates": len(df) - len(df.drop_duplicates()),
            "numeric_cols": len(df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns),
            "categorical_cols": len(df.select_dtypes(include=['object', 'string']).columns),
            "datetime_cols": len(df.select_dtypes(include=['datetime64']).columns)
        }

        # Column details
        columns_info = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)

            if dtype_str in ['int64', 'int32', 'float64', 'float32']:
                col_type = "numeric"
            elif dtype_str in ['object', 'string']:
                col_type = "categorical"
            elif 'datetime' in dtype_str:
                col_type = "datetime"
            else:
                col_type = "other"

            columns_info.append({
                "name": col,
                "dtype": dtype_str,
                "type": col_type,
                "missing": int(df[col].isnull().sum()),
                "unique_values": len(df[col].unique()) if col_type != "numeric" else None
            })

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_record.name if hasattr(dataset_record, 'name') else "Unknown",
            "file_path": str(file_path),
            "statistics": stats,
            "columns": columns_info,
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching dataset info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str = Path(...), rows: int = 100, db: Session = Depends(get_db)):
    """Get dataset preview - returns actual data with columns and rows"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"error": "Dataset not found"}

    df = None

    # ✅ Smart path resolution (handles both flat and collection subfolder paths)
    file_path = _resolve_dataset_file(dataset)

    # Load from cache or file
    if dataset_id in dataset_cache:
        df = dataset_cache[dataset_id]
    else:
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, nrows=rows)
                dataset_cache[dataset_id] = df
                logger.info(f"✅ Loaded dataset preview: {dataset_id}")
            except Exception as e:
                logger.error(f"❌ Could not read file: {str(e)}")
                return {"error": f"Could not read file: {str(e)}"}

    if df is None or df.empty:
        return {"error": "No data available"}

    # Format columns
    columns = [
        {
            "name": col,
            "type": str(df[col].dtype),
        }
        for col in df.columns
    ]

    # Format rows — replace NaN/Inf with None for JSON compatibility
    rows_data = []
    for record in df.to_dict('records'):
        rows_data.append({
            k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v)
            for k, v in record.items()
        })

    return {
        "dataset_id": dataset_id,
        "columns": columns,
        "rows": rows_data,
        "total_rows": len(df),
        "preview_rows": len(rows_data),
    }


@router.get("/{dataset_id}/quality")
async def get_dataset_quality(dataset_id: str = Path(...), db: Session = Depends(get_db)):
    """Get REAL data quality analysis with detailed metrics"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"error": "Dataset not found"}

    df = None

    # ✅ Smart path resolution (handles both flat and collection subfolder paths)
    file_path = _resolve_dataset_file(dataset)

    # Load from cache or file
    if dataset_id in dataset_cache:
        df = dataset_cache[dataset_id]
    else:
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                dataset_cache[dataset_id] = df
                logger.info(f"✅ Loaded dataset for quality analysis: {dataset_id}")
            except Exception as e:
                logger.error(f"❌ Could not read file: {str(e)}")
                return {"error": f"Could not read file: {str(e)}"}

    if df is None or df.empty:
        return {"error": "No data available"}

    # Calculate REAL statistics
    total_rows = len(df)
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    # Calculate metrics
    missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    completeness = 100 - missing_percentage
    uniqueness = 100 - (duplicate_rows / total_rows * 100) if total_rows > 0 else 100
    consistency = 100

    # Per-column quality
    column_quality = []
    for col in df.columns:
        col_missing = df[col].isnull().sum()
        col_total = len(df[col])
        col_missing_pct = (col_missing / col_total * 100) if col_total > 0 else 0

        column_quality.append({
            "name": col,
            "data_type": str(df[col].dtype),
            "missing_count": int(col_missing),
            "missing_percentage": float(col_missing_pct),
            "unique_count": int(df[col].nunique()),
        })

    return {
        "dataset_id": dataset_id,
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "duplicate_rows": int(duplicate_rows),
        "missing_percentage": float(missing_percentage),
        "completeness": float(completeness),
        "uniqueness": float(uniqueness),
        "consistency": float(consistency),
        "overall_quality_score": round((completeness + uniqueness + consistency) / 3, 2),
        "column_quality": column_quality
    }


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str = Path(...), db: Session = Depends(get_db)):
    """Delete dataset and its file from Kedro path"""
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return {"error": "Dataset not found"}

        # ✅ Delete file from Kedro path (smart resolution)
        file_path = _resolve_dataset_file(dataset)

        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🗑️ Deleted file: {file_path}")

        # ✅ Delete from cache
        if dataset_id in dataset_cache:
            del dataset_cache[dataset_id]

        # ✅ Delete from database
        db.delete(dataset)
        db.commit()

        logger.info(f"✅ Dataset deleted: {dataset_id}")

        return {"message": "Dataset deleted successfully"}

    except Exception as e:
        logger.error(f"❌ Error deleting dataset: {str(e)}", exc_info=True)
        db.rollback()
        return {"error": f"Failed to delete dataset: {str(e)}"}