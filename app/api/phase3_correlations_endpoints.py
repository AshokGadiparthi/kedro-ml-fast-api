"""
Phase 3: Advanced Correlations API Endpoints
FastAPI router for advanced correlation analysis
FINAL CUSTOMIZED VERSION - Works with YOUR Dataset model!
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
import logging
from datetime import datetime
import pandas as pd
import os

# ✅ Correct imports
from app.core.phase3_advanced_correlations import AdvancedCorrelationAnalysis
from app.core.database import get_db
from app.models.models import Dataset
from sqlalchemy.orm import Session

router = APIRouter(prefix="", tags=["Phase 3 - Correlations"])
logger = logging.getLogger(__name__)

from pathlib import Path
import os

import numpy as np
from typing import Any
import pandas as pd

from typing import Any
import numpy as np
import pandas as pd
import math
from datetime import datetime, date
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

def convert_numpy_types(obj: Any) -> Any:
    # --- numpy scalars ---
    if isinstance(obj, np.generic):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            # JSON does not like NaN/Infinity
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        return obj.item()  # fallback for other numpy scalars

    # --- pandas missing values ---
    if obj is pd.NA:
        return None

    # --- pandas objects ---
    if isinstance(obj, pd.DataFrame):
        return convert_numpy_types(obj.to_dict(orient="records"))

    if isinstance(obj, pd.Series):
        return convert_numpy_types(obj.to_list())

    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()

    # --- numpy arrays ---
    if isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())

    # --- dicts (IMPORTANT: convert KEYS too) ---
    if isinstance(obj, dict):
        return {
            str(convert_numpy_types(k)): convert_numpy_types(v)
            for k, v in obj.items()
        }

    # --- iterables (IMPORTANT: handle sets) ---
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [convert_numpy_types(x) for x in obj]

    return obj


KEDRO_PROJECT_PATH = Path("/home/ashok/work/latest/full/kedro-ml-engine-integrated")


def get_dataset_from_db(dataset_id: str, db: Session) -> Optional[pd.DataFrame]:
    """
    Get dataset from database - FIXED VERSION

    ✅ FIXED: Corrected file path retrieval logic

    Your Dataset model has these attributes:
    - id: Dataset ID
    - project_id: Project reference
    - file_name: Original filename
    - kedro_path: ✅ CORRECT PATH (data/01_raw/{project_id}/{filename})
    - created_at: Creation timestamp
    - description: Description text
    - name: Dataset name
    - file_size_bytes: File size

    Args:
        dataset_id: The dataset ID
        db: Database session (dependency injection)

    Returns:
        pandas DataFrame or None if not found
    """
    try:
        logger.info(f"📂 Retrieving dataset: {dataset_id}")

        # Query the database using the session
        dataset_record = db.query(Dataset).filter(
            Dataset.id == dataset_id
        ).first()

        if not dataset_record:
            logger.warning(f"⚠️ Dataset not found: {dataset_id}")
            return None

        # ✅ FIX: Get the correct path from your Dataset model
        # Your database has: kedro_path = "data/01_raw/{project_id}/{filename}"

        file_path = None

        # Try to get kedro_path (your actual path column)
        if hasattr(dataset_record, 'kedro_path') and dataset_record.kedro_path:
            file_path = dataset_record.kedro_path
            logger.info(f"✅ Using kedro_path: {file_path}")

        # Fallback: Try other common path attributes
        elif hasattr(dataset_record, 'file_path') and dataset_record.file_path:
            file_path = dataset_record.file_path
            logger.info(f"✅ Using file_path: {file_path}")

        elif hasattr(dataset_record, 'path') and dataset_record.path:
            file_path = dataset_record.path
            logger.info(f"✅ Using path: {file_path}")

        # If still no path, log available attributes
        if not file_path:
            available_attrs = [k for k in dataset_record.__dict__.keys() if not k.startswith('_')]
            logger.error(f"❌ No file path found in Dataset attributes")
            logger.error(f"   Available: {available_attrs}")
            logger.error(f"   Expected one of: kedro_path, file_path, path")
            raise HTTPException(
                status_code=500,
                detail=f"Cannot determine file path from dataset. Available attributes: {available_attrs}"
            )

        # ✅ FIX: Build FULL path AFTER we have the relative path
        full_file_path = os.path.join(str(KEDRO_PROJECT_PATH), file_path)
        logger.info(f"📍 Full file path: {full_file_path}")

        # Verify file exists
        if not os.path.exists(full_file_path):
            logger.error(f"❌ File not found at path: {full_file_path}")
            logger.error(f"   Relative path: {file_path}")
            logger.error(f"   KEDRO_PROJECT_PATH: {KEDRO_PROJECT_PATH}")
            raise HTTPException(status_code=500, detail=f"Dataset file not found: {full_file_path}")

        # Load the CSV file
        logger.info(f"📖 Loading CSV from: {full_file_path}")
        df = pd.read_csv(full_file_path)
        logger.info(f"✅ Loaded dataset {dataset_id} with shape {df.shape}")
        return df

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving dataset {dataset_id}: {str(e)}")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception details: {str(e)}", exc_info=True)
        return None


@router.get("/{dataset_id}/phase3/correlations/enhanced")
async def get_enhanced_correlations(
        dataset_id: str,
        threshold: float = Query(0.3, ge=0.0, le=1.0),
        db: Session = Depends(get_db)
) -> dict:
    """
    Get enhanced correlation analysis

    Parameters:
    - dataset_id: Dataset ID
    - threshold: Correlation threshold (0.0-1.0, default: 0.3)

    Returns:
    - Correlation matrix
    - Correlation pairs above threshold
    - High correlation pairs (>0.7)
    - Very high correlation pairs (>0.9)
    - Multicollinearity pairs
    - Strength distribution
    - Statistics

    Response time: ~200-300ms
    """
    try:
        logger.info(f"📊 Enhanced correlations requested for dataset: {dataset_id}")

        # Get dataset
        df = get_dataset_from_db(dataset_id, db)

        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Perform analysis
        analyzer = AdvancedCorrelationAnalysis(df)
        results = analyzer.get_enhanced_correlations(threshold=threshold)

        logger.info(f"✅ Enhanced correlations analysis completed")

        payload = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "analysis": results,
            "threshold": threshold,
            "analysis_type": "Enhanced Correlations",
            "total_features": len(df.select_dtypes(include=['number']).columns)
        }

        safe_payload = convert_numpy_types(payload)
        return JSONResponse(content=jsonable_encoder(safe_payload))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in enhanced correlations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing correlations: {str(e)}")


@router.get("/{dataset_id}/phase3/correlations/vif")
async def get_vif_analysis(
        dataset_id: str,
        db: Session = Depends(get_db)
) -> dict:
    """
    Get Variance Inflation Factor (VIF) analysis

    Returns:
    - VIF scores for each numeric feature
    - Severity levels
    - Problematic features
    - Overall multicollinearity level
    - Interpretation

    VIF Interpretation:
    - VIF < 5: Acceptable
    - VIF 5-10: Moderate multicollinearity (caution)
    - VIF > 10: High multicollinearity (action needed)

    Response time: ~250-350ms
    """
    try:
        logger.info(f"📈 VIF analysis requested for dataset: {dataset_id}")

        df = get_dataset_from_db(dataset_id, db)

        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        analyzer = AdvancedCorrelationAnalysis(df)
        vif_results = analyzer.get_vif_analysis()

        logger.info(f"✅ VIF analysis completed")

        payload = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "analysis": vif_results,
            "analysis_type": "VIF (Variance Inflation Factor)",
            "interpretation_guide": {
                "low": "VIF < 5: Acceptable multicollinearity",
                "moderate": "VIF 5-10: Moderate multicollinearity - caution recommended",
                "high": "VIF > 10: High multicollinearity - action needed"
            }
        }

        safe_payload = convert_numpy_types(payload)
        return JSONResponse(content=jsonable_encoder(safe_payload))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in VIF analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in VIF analysis: {str(e)}")


@router.get("/{dataset_id}/phase3/correlations/heatmap-data")
async def get_heatmap_data(
        dataset_id: str,
        db: Session = Depends(get_db)
) -> dict:
    """
    Get correlation heatmap visualization data

    Returns:
    - Heatmap data in list format
    - Column names
    - Min/max correlation values

    Response time: ~150-250ms
    """
    try:
        logger.info(f"🔥 Heatmap data requested for dataset: {dataset_id}")

        df = get_dataset_from_db(dataset_id, db)

        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        analyzer = AdvancedCorrelationAnalysis(df)
        heatmap_data = analyzer.get_correlation_heatmap_data()

        logger.info(f"✅ Heatmap data generated")

        payload = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "heatmap": heatmap_data,
            "analysis_type": "Correlation Heatmap Data"
        }

        safe_payload = convert_numpy_types(payload)
        return JSONResponse(content=jsonable_encoder(safe_payload))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating heatmap data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating heatmap: {str(e)}")


@router.get("/{dataset_id}/phase3/correlations/clustering")
async def get_correlation_clustering(
        dataset_id: str,
        db: Session = Depends(get_db)
) -> dict:
    """
    Get feature clustering based on correlations

    Returns:
    - Feature clusters
    - Cluster count
    - Cluster interpretation

    Response time: ~300-400ms
    """
    try:
        logger.info(f"🎯 Correlation clustering requested for dataset: {dataset_id}")

        df = get_dataset_from_db(dataset_id, db)

        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        analyzer = AdvancedCorrelationAnalysis(df)
        clustering = analyzer.get_correlation_clustering()

        logger.info(f"✅ Correlation clustering completed")

        payload  = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "clustering": clustering,
            "analysis_type": "Correlation-Based Feature Clustering"
        }

        safe_payload = convert_numpy_types(payload)
        return JSONResponse(content=jsonable_encoder(safe_payload))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in correlation clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in clustering: {str(e)}")


@router.get("/{dataset_id}/phase3/correlations/relationship-insights")
async def get_relationship_insights(
        dataset_id: str,
        db: Session = Depends(get_db)
) -> dict:
    """
    Get relationship insights and patterns

    Returns:
    - Strongest positive relationships
    - Strongest negative relationships
    - Uncorrelated pairs
    - Feature connectivity scores
    - Interesting patterns

    Response time: ~200-300ms
    """
    try:
        logger.info(f"🔗 Relationship insights requested for dataset: {dataset_id}")

        df = get_dataset_from_db(dataset_id, db)

        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        analyzer = AdvancedCorrelationAnalysis(df)
        insights = analyzer.get_relationship_insights()

        logger.info(f"✅ Relationship insights generated")

        payload  = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "insights": insights,
            "analysis_type": "Relationship Insights"
        }

        safe_payload = convert_numpy_types(payload)
        return JSONResponse(content=jsonable_encoder(safe_payload))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating relationship insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")


@router.get("/{dataset_id}/phase3/correlations/warnings")
async def get_multicollinearity_warnings(
        dataset_id: str,
        db: Session = Depends(get_db)
) -> dict:
    """
    Get multicollinearity warnings and recommendations

    Returns:
    - Warning list
    - Warning count
    - Overall assessment
    - Specific recommendations

    Response time: ~250-350ms
    """
    try:
        logger.info(f"⚠️ Multicollinearity warnings requested for dataset: {dataset_id}")

        df = get_dataset_from_db(dataset_id, db)

        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        analyzer = AdvancedCorrelationAnalysis(df)
        warnings = analyzer.get_multicollinearity_warnings()

        logger.info(f"✅ Multicollinearity warnings generated")

        payload = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "warnings": warnings,
            "analysis_type": "Multicollinearity Warnings"
        }

        safe_payload = convert_numpy_types(payload)
        return JSONResponse(content=jsonable_encoder(safe_payload))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating warnings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating warnings: {str(e)}")


@router.get("/{dataset_id}/phase3/correlations/complete")
async def get_complete_correlation_analysis(
        dataset_id: str,
        threshold: float = Query(0.3, ge=0.0, le=1.0),
        db: Session = Depends(get_db)
) -> dict:
    """
    Get complete correlation analysis (all endpoints combined)

    This is the RECOMMENDED endpoint - use this for best performance!

    Parameters:
    - dataset_id: Dataset ID
    - threshold: Correlation threshold (0.0-1.0, default: 0.3)

    Returns:
    - Enhanced correlations
    - VIF analysis
    - Heatmap data
    - Feature clustering
    - Relationship insights
    - Multicollinearity warnings

    Response time: ~1-2 seconds (combined)
    """
    try:
        logger.info(f"📊 Complete correlation analysis requested for dataset: {dataset_id}")

        df = get_dataset_from_db(dataset_id, db)

        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        analyzer = AdvancedCorrelationAnalysis(df)

        # Gather all analyses
        complete_analysis = {
            "enhanced_correlations": analyzer.get_enhanced_correlations(threshold=threshold),
            "vif_analysis": analyzer.get_vif_analysis(),
            "heatmap_data": analyzer.get_correlation_heatmap_data(),
            "clustering": analyzer.get_correlation_clustering(),
            "relationship_insights": analyzer.get_relationship_insights(),
            "multicollinearity_warnings": analyzer.get_multicollinearity_warnings()
        }

        logger.info(f"✅ Complete correlation analysis finished")

        payload  = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "analysis": complete_analysis,
            "analysis_type": "Complete Phase 3 Correlation Analysis",
            "components": 6,
            "total_features": len(df.select_dtypes(include=['number']).columns)
        }

        safe_payload = convert_numpy_types(payload)
        return JSONResponse(content=jsonable_encoder(safe_payload))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in complete correlation analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")