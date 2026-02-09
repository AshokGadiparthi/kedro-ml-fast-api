"""
Model Evaluation API Endpoints
================================
REST API for real-time model evaluation using Kedro pipeline artifacts.

These endpoints feed ALL 5 tabs of the frontend Model Evaluation Dashboard:
  Tab 1 (Overview)           → GET  /complete/{model_id}  → thresholdEvaluation
  Tab 2 (Business Impact)    → GET  /complete/{model_id}  → businessImpact
  Tab 3 (Curves & Threshold) → GET  /complete/{model_id}  → curves, optimalThreshold
  Tab 4 (Advanced Analysis)  → GET  /complete/{model_id}  → learningCurve, featureImportance
  Tab 5 (Production Ready)   → GET  /complete/{model_id}  → productionReadiness

Routes:
  GET  /api/v1/evaluation/trained-models          → List models from Kedro phases
  GET  /api/v1/evaluation/complete/{model_id}     → Complete evaluation (all 5 tabs)
  POST /api/v1/evaluation/complete/{model_id}     → Complete evaluation with custom params
  GET  /api/v1/evaluation/threshold/{model_id}    → Tab 1: Metrics at threshold
  GET  /api/v1/evaluation/business/{model_id}     → Tab 2: Business impact
  GET  /api/v1/evaluation/curves/{model_id}       → Tab 3: ROC + PR curves
  GET  /api/v1/evaluation/advanced/{model_id}     → Tab 4: Learning + features
  GET  /api/v1/evaluation/production/{model_id}   → Tab 5: Production readiness
  POST /api/v1/evaluation/refresh-cache           → Clear cached artifacts
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging
from app.core.model_evaluation_service import get_evaluation_service, _sanitize_for_json

from app.core.model_evaluation_service import get_evaluation_service
from app.schemas.evaluation_schemas import (
    EvaluationRequest,
    CompleteEvaluationResponse,
    ThresholdEvaluationResponse,
    BusinessImpactResponse,
    CurvesResponse,
    LearningCurveResponse,
    FeatureImportanceResponse,
    OptimalThresholdResponse,
    ProductionReadinessResponse,
    TrainedModelsListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Model Evaluation"])

logger.info("✅ Model Evaluation router created")


# ============================================================================
# LIST TRAINED MODELS (for model selector dropdown)
# ============================================================================

@router.get("/trained-models", response_model=TrainedModelsListResponse)
async def list_trained_models():
    """
    List all trained models available from Kedro pipeline outputs.

    Returns models from:
      - Phase 3: Best model (best_model.pkl)
      - Phase 4: All trained algorithms (phase4_report.csv)

    Frontend uses this to populate the model selector dropdown.
    """
    try:
        service = get_evaluation_service()
        result = service.list_trained_models()
        return _sanitize_for_json(result)
    except Exception as e:
        logger.error(f"Error listing trained models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list trained models: {str(e)}")


# ============================================================================
# COMPLETE EVALUATION - ALL 5 TABS IN ONE CALL (GET)
# ============================================================================

@router.get("/complete/{model_id}")
async def get_complete_evaluation(
        model_id: str,
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Classification threshold"),
        cost_fp: float = Query(500, description="Cost per false positive"),
        cost_fn: float = Query(2000, description="Cost per false negative"),
        revenue_tp: float = Query(1000, description="Revenue per true positive"),
        volume: Optional[int] = Query(None, description="Scale to this prediction volume"),
):
    """
    Get complete model evaluation data for all 5 dashboard tabs.

    This is the PRIMARY endpoint the frontend should call.
    Returns a single JSON object that feeds:
      - Tab 1 (Overview): thresholdEvaluation (confusion matrix, metrics, rates)
      - Tab 2 (Business Impact): businessImpact (costs, revenue, profit)
      - Tab 3 (Curves & Threshold): curves (ROC, PR), optimalThreshold
      - Tab 4 (Advanced Analysis): learningCurve, featureImportance
      - Tab 5 (Production Readiness): productionReadiness (criteria checklist)

    Parameters:
      - model_id: ID from /trained-models endpoint (or "best" for best model)
      - threshold: Classification threshold (0.0-1.0, default 0.5)
      - cost_fp: Cost per false positive in dollars
      - cost_fn: Cost per false negative in dollars
      - revenue_tp: Revenue per true positive in dollars
      - volume: Optional volume to scale predictions to
    """
    try:
        service = get_evaluation_service()

        result = service.get_complete_evaluation(
            threshold=threshold,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            revenue_tp=revenue_tp,
            volume=volume,
            model_id=model_id,
        )

        # Validate that core data is available
        if result.get("thresholdEvaluation") is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Evaluation data not available",
                    "reason": f"Could not compute evaluation for model '{model_id}'. "
                              "Ensure the pipeline has been run (Phase 3+) and artifacts exist. "
                              "If this is a specific algorithm, ensure Phase 4 has been run.",
                    "required_files": [
                        "data/06_models/best_model.pkl (or data/06_models/phase4/all_trained_models.pkl)",
                        "data/03_primary/X_test_selected.csv (or X_test_scaled.csv)",
                        "data/03_primary/y_test_raw.pkl",
                    ],
                    "kedro_command": "kedro run --pipeline phase4",
                }
            )

        logger.info(
            f"Complete evaluation for model={model_id}: "
            f"threshold={result.get('thresholdEvaluation') is not None}, "
            f"business={result.get('businessImpact') is not None}, "
            f"curves={result.get('curves') is not None}, "
            f"learning={result.get('learningCurve') is not None}, "
            f"features={result.get('featureImportance') is not None}, "
            f"production={result.get('productionReadiness') is not None}"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing evaluation for model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ============================================================================
# COMPLETE EVALUATION - ALL 5 TABS IN ONE CALL (POST with body)
# ============================================================================

@router.post("/complete/{model_id}")
async def post_complete_evaluation(
        model_id: str,
        request: EvaluationRequest,
):
    """
    Same as GET /complete/{model_id} but accepts parameters in request body.
    Useful when frontend needs to send complex business parameters.
    """
    try:
        service = get_evaluation_service()

        result = service.get_complete_evaluation(
            threshold=request.threshold,
            cost_fp=request.cost_fp,
            cost_fn=request.cost_fn,
            revenue_tp=request.revenue_tp,
            volume=request.volume,
            model_id=model_id,
        )

        if result.get("thresholdEvaluation") is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Evaluation data not available",
                    "reason": "Pipeline artifacts not found. Run Kedro pipeline first.",
                }
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in POST evaluation for model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ============================================================================
# INDIVIDUAL TAB ENDPOINTS (for lazy loading / partial refresh)
# ============================================================================

@router.get("/threshold/{model_id}")
async def get_threshold_evaluation(
        model_id: str,
        threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Tab 1 (Overview): Get confusion matrix and metrics at a specific threshold.
    Lightweight call for threshold slider updates.
    """
    try:
        service = get_evaluation_service()
        result = service.compute_threshold_evaluation(threshold, model_id=model_id)

        if result is None:
            raise HTTPException(status_code=422, detail="Cannot compute metrics: model artifacts missing")

        return _sanitize_for_json(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing threshold evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business/{model_id}")
async def get_business_impact(
        model_id: str,
        threshold: float = Query(0.5, ge=0.0, le=1.0),
        cost_fp: float = Query(500),
        cost_fn: float = Query(2000),
        revenue_tp: float = Query(1000),
        volume: Optional[int] = Query(None),
):
    """
    Tab 2 (Business Impact): Get financial analysis.
    """
    try:
        service = get_evaluation_service()
        result = service.compute_business_impact(threshold, cost_fp, cost_fn, revenue_tp, volume, model_id=model_id)

        if result is None:
            raise HTTPException(status_code=422, detail="Cannot compute business impact: model artifacts missing")

        return _sanitize_for_json(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing business impact: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/curves/{model_id}")
async def get_curves(model_id: str):
    """
    Tab 3 (Curves & Threshold): Get ROC and PR curve arrays.
    """
    try:
        service = get_evaluation_service()
        curves = service.compute_curves(model_id=model_id)
        optimal = service.compute_optimal_threshold(model_id=model_id)

        return _sanitize_for_json({
            "curves": curves,
            "optimalThreshold": optimal,
        })

    except Exception as e:
        logger.error(f"Error computing curves: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced/{model_id}")
async def get_advanced_analysis(model_id: str):
    """
    Tab 4 (Advanced Analysis): Get learning curve and feature importance.
    """
    try:
        service = get_evaluation_service()
        learning = service.compute_learning_curve(model_id=model_id)
        features = service.compute_feature_importance(model_id=model_id)

        return _sanitize_for_json({
            "learningCurve": learning,
            "featureImportance": features,
        })

    except Exception as e:
        logger.error(f"Error computing advanced analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/production/{model_id}")
async def get_production_readiness(
        model_id: str,
        threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Tab 5 (Production Readiness): Get readiness criteria checklist.
    """
    try:
        service = get_evaluation_service()
        result = service.compute_production_readiness(threshold, model_id=model_id)
        return _sanitize_for_json(result)

    except Exception as e:
        logger.error(f"Error computing production readiness: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

@router.post("/refresh-cache")
async def refresh_cache():
    """
    Clear the evaluation service cache.
    Call this after re-running the Kedro pipeline to pick up new artifacts.
    """
    try:
        service = get_evaluation_service()
        service.clear_cache()
        return {"status": "ok", "message": "Evaluation cache cleared. Next request will reload all artifacts."}
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DIAGNOSTICS (development only)
# ============================================================================

@router.get("/diagnostics")
async def get_diagnostics():
    """
    Check which Kedro artifacts are available.
    Useful for debugging missing data in the dashboard.
    """
    import os
    from pathlib import Path

    kedro_path = Path(os.getenv(
        'KEDRO_PROJECT_PATH',
        '/home/ashok/work/latest/full/kedro-ml-engine-integrated'
    ))

    artifacts = {
        "kedro_project_path": str(kedro_path),
        "kedro_project_exists": kedro_path.exists(),
        "artifacts": {}
    }

    # Check each expected file
    expected_files = {
        # Phase 2
        "feature_importance_csv": "data/08_reporting/feature_importance.csv",
        "X_test_selected": "data/03_primary/X_test_selected.csv",
        "X_test_scaled": "data/03_primary/X_test_scaled.csv",
        "y_test_raw": "data/03_primary/y_test_raw.pkl",
        "y_train_raw": "data/03_primary/y_train_raw.pkl",
        "selected_features": "data/03_primary/selected_features.pkl",
        # Phase 3
        "best_model": "data/06_models/best_model.pkl",
        "model_evaluation": "data/06_models/model_evaluation.pkl",
        "phase3_predictions": "data/07_model_output/phase3_predictions.csv",
        "scalers": "data/03_primary/scalers.pkl",
        "problem_type": "data/08_reporting/problem_type.txt",
        "cross_validation": "data/08_reporting/cross_validation_results.pkl",
        # Phase 4
        "phase4_report": "data/07_model_output/phase4_report.csv",
        "phase4_summary": "data/07_model_output/phase4_summary.json",
        # Phase 5
        "phase5_metrics": "data/08_reporting/phase5_metrics.json",
    }

    for name, rel_path in expected_files.items():
        full_path = kedro_path / rel_path
        exists = full_path.exists()
        size = full_path.stat().st_size if exists else 0
        artifacts["artifacts"][name] = {
            "path": rel_path,
            "exists": exists,
            "size_bytes": size,
            "status": "✅" if exists and size > 0 else "❌ missing",
        }

    # Summary
    found = sum(1 for a in artifacts["artifacts"].values() if a["exists"])
    total = len(artifacts["artifacts"])
    artifacts["summary"] = {
        "found": found,
        "total": total,
        "percentage": round(found / total * 100, 1) if total > 0 else 0,
        "ready_for_evaluation": found >= 4,  # Need at minimum: best_model, X_test, y_test, model_evaluation
    }

    # Tab readiness
    artifacts["tab_readiness"] = {
        "tab1_overview": all(
            artifacts["artifacts"][f]["exists"]
            for f in ["best_model", "y_test_raw"]
            if f in artifacts["artifacts"]
        ),
        "tab2_business": all(
            artifacts["artifacts"][f]["exists"]
            for f in ["best_model", "y_test_raw"]
            if f in artifacts["artifacts"]
        ),
        "tab3_curves": all(
            artifacts["artifacts"][f]["exists"]
            for f in ["best_model", "y_test_raw"]
            if f in artifacts["artifacts"]
        ),
        "tab4_advanced": any(
            artifacts["artifacts"][f]["exists"]
            for f in ["model_evaluation", "feature_importance_csv"]
            if f in artifacts["artifacts"]
        ),
        "tab5_production": artifacts["artifacts"]["best_model"]["exists"],
    }

    return artifacts