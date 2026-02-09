"""
Prediction API Schemas
======================
Pydantic models matching the exact frontend data contracts.

Frontend components served:
  PredictionsDashboard.tsx  → Single Prediction, Batch, History, Monitoring tabs
  PredictionsContainer.tsx  → Model selection, loading states
  SinglePredictionTab.tsx   → Input form, result rendering
  BatchPredictionTab.tsx    → CSV upload, validation, results
  HistoryTab.tsx            → Prediction history with filters
  APITab.tsx                → API usage stats, code examples

All response shapes are derived from MOCK_* constants in PredictionsContainer.tsx
to ensure pixel-perfect frontend compatibility.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# REQUEST MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """
    Request body for POST /predict
    Frontend sends: { modelId, threshold, features: { age: 35, workclass: "Private", ... } }
    """
    modelId: Optional[str] = "best"
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    features: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
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
        }


# ============================================================================
# DEPLOYED MODELS — matches MOCK_DEPLOYED_MODELS in PredictionsContainer.tsx
# ============================================================================

class InputFeatureSchema(BaseModel):
    """
    Single input feature definition.
    Frontend renders:
      - numeric → <Input type="number" min={min} max={max} />
      - categorical → <Select> with options
    """
    name: str
    type: str                       # "numeric" or "categorical"
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    description: Optional[str] = None
    defaultValue: Any = None
    required: bool = True
    displayName: Optional[str] = None
    inputType: Optional[str] = None
    typicalRange: Optional[str] = None


class OutputSchema(BaseModel):
    """Model output schema definition."""
    prediction: str = "binary"
    classes: List[str] = ["<=50K", ">50K"]
    includesProbability: bool = True
    includesExplanation: bool = True


class DeployedModel(BaseModel):
    """
    Single deployed model.
    PredictionsDashboard.tsx line ~170: (selectedModel.accuracy * 100).toFixed(1)
    → accuracy MUST be 0-1 scale (e.g., 0.85 not 85.0)
    """
    id: str
    name: str
    algorithm: str
    version: str = "v1.0"
    deployedAt: str
    accuracy: float                   # 0-1 scale! Frontend does * 100
    status: str                       # "active" or "available"
    endpoint: str = "/api/v1/predictions/predict"
    inputFeatures: List[InputFeatureSchema]
    outputSchema: OutputSchema = OutputSchema()


class DeployedModelsResponse(BaseModel):
    """Response for GET /deployed-models — returned as list but wrapped for consistency."""
    models: List[DeployedModel] = []
    totalModels: int = 0
    activeModel: Optional[DeployedModel] = None


# ============================================================================
# SINGLE PREDICTION — matches MOCK_SINGLE_PREDICTION in PredictionsContainer.tsx
# ============================================================================

class PredictionOutput(BaseModel):
    """
    Nested output block.
    Frontend accesses: predictionResult.output.predictionLabel, .probability,
    .probabilities, .confidence, .threshold
    """
    prediction: str               # ">50K" or "<=50K"
    predictionLabel: str          # "High Income (>$50K)" or "Low Income (≤$50K)"
    predictionValue: int          # 1 or 0
    probability: float            # 0-1, probability of positive class
    probabilities: Dict[str, float]  # { "<=50K": 0.153, ">50K": 0.847 }
    confidence: str               # "high", "moderate", "low"
    threshold: float


class FeatureContribution(BaseModel):
    """
    Single feature's contribution to the prediction.
    PredictionsDashboard.tsx uses: feature.feature, .contribution, .direction, .value
    SinglePredictionTab.tsx  uses: exp.feature, .impact, .direction, .value
    → We include both .contribution and .impact (same value) for compat with both components.
    """
    feature: str
    contribution: float
    impact: Optional[float] = None    # alias of contribution for SinglePredictionTab
    direction: str                    # "positive" or "negative"
    value: Any = None                 # raw input value
    shapValue: Optional[float] = None


class PredictionExplanation(BaseModel):
    """
    Explanation block.
    PredictionsDashboard: explanation.topFeatures[], .baselineScore, .explanation
    SinglePredictionTab:  predictionResult.explanation (as array directly, OR .topFeatures)
    """
    topFeatures: List[FeatureContribution] = []
    baselineScore: Optional[float] = None
    explanation: Optional[str] = None


class PredictionMetadata(BaseModel):
    processingTimeMs: float
    modelVersion: str


class PredictionResponse(BaseModel):
    """
    Response for POST /predict

    PredictionsDashboard.tsx renders:
      predictionResult.output.predictionLabel  → big green/red label
      predictionResult.output.probabilities    → progress bars per class
      predictionResult.explanation.topFeatures → feature contribution bars
      predictionResult.metadata.processingTimeMs → "Processing: 45ms"

    SinglePredictionTab.tsx also reads:
      predictionResult.prediction   → top-level alias
      predictionResult.confidence   → top-level alias (0-1)
      predictionResult.probabilities → top-level alias
      predictionResult.explanation   → can be array OR object
    """
    predictionId: str
    modelId: str
    modelName: str
    timestamp: str
    input: Dict[str, Any]
    # Nested (PredictionsDashboard)
    output: PredictionOutput
    explanation: PredictionExplanation
    metadata: PredictionMetadata
    # Top-level aliases (SinglePredictionTab compatibility)
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


# ============================================================================
# BATCH — matches MOCK_BATCH_JOB in PredictionsContainer.tsx
# ============================================================================

class BatchInputFile(BaseModel):
    name: str
    size: int
    records: int


class BatchOutputFile(BaseModel):
    url: str
    name: str
    size: Optional[int] = None


class BatchSummary(BaseModel):
    """
    PredictionsDashboard uses: summary.predictions (Dict), summary.averageConfidence
    BatchPredictionTab uses: batchResults.approved, .rejected, .total, .avgConfidence
    """
    predictions: Dict[str, int] = {}
    averageConfidence: Optional[float] = None
    highConfidencePredictions: int = 0
    lowConfidencePredictions: int = 0
    # Aliases for BatchPredictionTab.tsx
    approved: Optional[int] = None
    rejected: Optional[int] = None
    total: Optional[int] = None
    avgConfidence: Optional[float] = None
    processingTime: Optional[str] = None


class BatchError(BaseModel):
    row: int
    error: str


class BatchJobStatus(BaseModel):
    """
    Response for POST /batch and GET /batch/{jobId}
    Frontend polls this every 2s and renders progress bar + summary.
    """
    jobId: str
    modelId: str
    modelName: str
    status: str                   # "pending", "processing", "completed", "failed"
    progress: float
    totalRecords: int
    processedRecords: int
    successfulRecords: int
    failedRecords: int
    startedAt: str
    completedAt: Optional[str] = None
    durationSeconds: Optional[int] = None
    inputFile: Optional[BatchInputFile] = None
    outputFile: Optional[BatchOutputFile] = None
    summary: Optional[BatchSummary] = None
    errors: List[BatchError] = []


# ============================================================================
# HISTORY — matches MOCK_PREDICTION_HISTORY in PredictionsContainer.tsx
# ============================================================================

class PredictionHistoryItem(BaseModel):
    """
    PredictionsDashboard: item.id, item.type, item.modelName, item.timestamp,
                          item.prediction, item.confidence, item.status
    HistoryTab:           item.predictedLabel, item.predictedClass, item.confidence,
                          item.model, item.inputs, item.details
    """
    id: str
    type: str                     # "single" or "batch"
    modelName: str
    timestamp: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    recordsProcessed: Optional[int] = None
    status: str
    # HistoryTab compatibility
    predictedLabel: Optional[str] = None
    predictedClass: Optional[str] = None
    model: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None
    timestampLabel: Optional[str] = None


class PaginationInfo(BaseModel):
    page: int
    pageSize: int
    totalItems: int
    totalPages: int


class PredictionHistoryResponse(BaseModel):
    predictions: List[PredictionHistoryItem]
    pagination: PaginationInfo


# ============================================================================
# MONITORING — matches MOCK_MONITORING_STATS in PredictionsContainer.tsx
# ============================================================================

class MonitoringSummaryStats(BaseModel):
    """
    MUST be nested under "stats" key.
    Frontend: monitoring.stats.totalPredictions, .averageLatencyMs, .errorRate, .throughput
    """
    totalPredictions: int
    averageLatencyMs: float
    errorRate: float              # 0-1 scale
    throughput: float


class ConfidenceBucket(BaseModel):
    """Frontend bar chart: { range, count }"""
    range: str
    count: int


class MonitoringAlert(BaseModel):
    id: str
    severity: str
    message: str
    timestamp: str


class MonitoringStats(BaseModel):
    """
    Response for GET /monitoring/stats
    Frontend: monitoring.stats.*, monitoring.predictionDistribution,
              monitoring.confidenceDistribution[], monitoring.hourlyTrend[], monitoring.alerts[]
    """
    modelId: Optional[str] = None
    modelName: Optional[str] = None
    timeRange: str = "last_24_hours"
    stats: MonitoringSummaryStats
    predictionDistribution: Dict[str, int]
    confidenceDistribution: List[ConfidenceBucket]
    hourlyTrend: List[Dict[str, Any]]
    alerts: List[MonitoringAlert] = []