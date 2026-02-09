"""
Predictions API Endpoints
=========================
REST API for real-time and batch predictions using Kedro pipeline artifacts.
All responses match the exact data shapes expected by the React frontend.

Frontend ↔ Backend mapping:
  PredictionsContainer.tsx  → GET /deployed-models
  PredictionsDashboard.tsx  → All 8 endpoints
  SinglePredictionTab.tsx   → POST /predict
  BatchPredictionTab.tsx    → POST /batch, GET /batch/{job_id}, GET /csv-template
  HistoryTab.tsx            → GET /history
  Monitoring tab            → GET /monitoring/stats

Routes:
  GET  /api/v1/predictions/deployed-models          → Models + input schema
  POST /api/v1/predictions/predict                  → Single prediction
  POST /api/v1/predictions/batch                    → Start batch job
  GET  /api/v1/predictions/batch/{job_id}           → Batch status/results
  GET  /api/v1/predictions/batch/{job_id}/download  → Download results CSV
  GET  /api/v1/predictions/history                  → Paginated history
  GET  /api/v1/predictions/monitoring/stats          → Real-time monitoring
  GET  /api/v1/predictions/csv-template             → Batch template CSV
"""

from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import io
import logging

from app.core.prediction_service import get_prediction_service
from app.schemas.prediction_schemas import PredictionRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])

logger.info("✅ Predictions router created")


# ============================================================================
# ENDPOINT 1: GET /deployed-models
# Frontend: PredictionsContainer.tsx → populates model selector dropdown
# Response shape: { models: [...], totalModels, activeModel }
# CRITICAL: accuracy is 0-1 scale (frontend does * 100)
# ============================================================================
@router.get("/deployed-models")
def get_deployed_models():
    """
    List all available trained models for prediction.

    Source: data/06_models/phase4/all_trained_models.pkl
            data/07_model_output/phase4_report.csv

    Response includes inputFeatures array for dynamic form generation
    and outputSchema for result rendering.
    """
    service = get_prediction_service()
    return service.get_deployed_models()


# ============================================================================
# ENDPOINT 2: POST /predict
# Frontend: SinglePredictionTab.tsx → form submit → render result
# Response shape: { predictionId, output: { prediction, predictionLabel,
#   probability, probabilities, confidence, threshold },
#   explanation: { topFeatures, baselineScore, explanation },
#   metadata: { processingTimeMs, modelVersion } }
# ============================================================================
@router.post("/predict")
def predict_single(request: PredictionRequest):
    """
    Make a single prediction from raw feature input.

    Request body example:
    {
        "modelId": "adaboostclassifier",
        "threshold": 0.5,
        "features": {
            "age": 35,
            "workclass": "Private",
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 5000,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "United-States"
        }
    }
    """
    if not request.features:
        raise HTTPException(status_code=400, detail="Missing 'features' in request body")

    service = get_prediction_service()
    result = service.predict_single(
        raw_features=request.features,
        model_id=request.modelId,
        threshold=request.threshold or 0.5,
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


# ============================================================================
# ENDPOINT 3: POST /batch
# Frontend: BatchPredictionTab.tsx → file upload → poll status
# Response shape: matches MOCK_BATCH_JOB in PredictionsContainer.tsx
# ============================================================================
@router.post("/batch")
async def start_batch(
        file: UploadFile = File(...),
        model_id: Optional[str] = Query(None, description="Model ID (defaults to best)")
):
    """
    Upload CSV and start batch prediction job.

    CSV columns accepted (both dot and underscore formats):
      age, workclass, education, education.num / education_num,
      marital.status / marital_status, occupation, relationship, race, sex,
      capital.gain / capital_gain, capital.loss / capital_loss,
      hours.per.week / hours_per_week, native.country / native_country
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

    service = get_prediction_service()
    result = service.start_batch_prediction(content, file.filename, model_id)

    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error", "Batch job failed"))

    return result


# ============================================================================
# ENDPOINT 4: GET /batch/{job_id}
# Frontend: BatchPredictionTab.tsx → polls every 2s
# Response shape: { jobId, status, progress, summary, errors, ... }
# ============================================================================
@router.get("/batch/{job_id}")
def get_batch_status(job_id: str):
    """
    Check status of a batch prediction job.
    Frontend polls this every 2 seconds until status is "completed" or "failed".
    """
    service = get_prediction_service()
    job = service.get_batch_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch job '{job_id}' not found")
    return job


# ============================================================================
# ENDPOINT 5: GET /batch/{job_id}/download
# Frontend: "Download Results" button in BatchPredictionTab.tsx
# ============================================================================
@router.get("/batch/{job_id}/download")
def download_batch_results(job_id: str):
    """
    Download batch prediction results as CSV.
    CSV includes original columns plus: prediction, probability, confidence
    """
    service = get_prediction_service()
    csv_bytes = service.get_batch_results_csv(job_id)
    if not csv_bytes:
        raise HTTPException(status_code=404, detail="Results not available")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{job_id}.csv"},
    )


# ============================================================================
# ENDPOINT 6: GET /history
# Frontend: HistoryTab.tsx → filterable table + detail modal
# Response shape: { predictions: [...], pagination: { page, pageSize, totalItems, totalPages } }
# ============================================================================
@router.get("/history")
def get_history(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, alias="limit", description="Items per page"),
        type: Optional[str] = Query(None, description="Filter: 'single', 'batch', or 'all'"),
        model_id: Optional[str] = Query(None, description="Filter by model ID"),
):
    """
    Get paginated prediction history.
    Includes both single and batch predictions.
    Filterable by type and model.
    """
    service = get_prediction_service()
    return service.get_prediction_history(page, page_size, type, model_id)


# ============================================================================
# ENDPOINT 7: GET /monitoring/stats
# Frontend: Monitoring tab → 4 stat cards + 3 charts + alerts
# Response shape: matches MOCK_MONITORING_STATS
# CRITICAL: stats is NESTED, confidenceDistribution is ARRAY [{range,count}]
# ============================================================================
@router.get("/monitoring/stats")
def get_monitoring_stats(
        model_id: Optional[str] = Query(None, alias="modelId", description="Filter by model"),
):
    """
    Get real-time monitoring statistics.

    Returns:
      stats.totalPredictions     → "Total Predictions" card
      stats.averageLatencyMs     → "Avg Latency" card
      stats.errorRate            → "Error Rate" card
      stats.throughput           → "Throughput" card
      predictionDistribution     → Pie chart
      confidenceDistribution[]   → Bar chart
      hourlyTrend[]              → Line chart
      alerts[]                   → Alert cards
    """
    service = get_prediction_service()
    return service.get_monitoring_stats(model_id)


# ============================================================================
# ENDPOINT 8: GET /csv-template
# Frontend: "Download CSV Template" button in BatchPredictionTab.tsx
# ============================================================================
@router.get("/csv-template")
def download_csv_template():
    """
    Download a CSV template with correct headers and 2 example rows.
    """
    service = get_prediction_service()
    template = service.get_csv_template()

    return StreamingResponse(
        io.BytesIO(template.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_template.csv"},
    )