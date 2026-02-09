"""
Model Evaluation API Schemas
============================
Pydantic models matching the frontend CompleteEvaluationResponse TypeScript types.
Every field name uses camelCase to match the frontend convention.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class EvaluationRequest(BaseModel):
    """Request body for evaluation with business parameters."""
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    cost_fp: float = Field(500, description="Cost per false positive")
    cost_fn: float = Field(2000, description="Cost per false negative")
    revenue_tp: float = Field(1000, description="Revenue per true positive")
    volume: Optional[int] = Field(None, description="Scale predictions to this volume")


# ============================================================================
# RESPONSE SUB-SCHEMAS  (camelCase to match frontend)
# ============================================================================

class ConfusionMatrixResponse(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int
    total: int

    class Config:
        populate_by_name = True


class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1Score: float = Field(..., alias="f1Score")
    aucRoc: float = Field(..., alias="aucRoc")

    class Config:
        populate_by_name = True


class RatesResponse(BaseModel):
    falsePositiveRate: float
    falseNegativeRate: float

    class Config:
        populate_by_name = True


class ThresholdEvaluationResponse(BaseModel):
    confusionMatrix: ConfusionMatrixResponse
    metrics: MetricsResponse
    rates: RatesResponse

    class Config:
        populate_by_name = True


# --- Business Impact ---

class CostsResponse(BaseModel):
    totalCost: float
    falsePositiveCost: float
    falseNegativeCost: float
    costPerFalsePositive: Optional[float] = None
    costPerFalseNegative: Optional[float] = None

    class Config:
        populate_by_name = True


class RevenueResponse(BaseModel):
    truePositiveRevenue: float
    revenuePerTruePositive: Optional[float] = None

    class Config:
        populate_by_name = True


class FinancialResponse(BaseModel):
    profit: float
    improvementVsBaseline: float
    atVolume: Optional[int] = None
    approvalRate: Optional[float] = None

    class Config:
        populate_by_name = True


class ScaledCountsResponse(BaseModel):
    truePositives: int
    falsePositives: int
    falseNegatives: int

    class Config:
        populate_by_name = True


class BusinessImpactResponse(BaseModel):
    costs: CostsResponse
    revenue: RevenueResponse
    financial: FinancialResponse
    scaledCounts: Optional[ScaledCountsResponse] = None

    class Config:
        populate_by_name = True


# --- Production Readiness ---

class ReadinessCriterion(BaseModel):
    name: str
    description: str
    passed: bool
    category: Optional[str] = None

    class Config:
        populate_by_name = True


class ReadinessSummary(BaseModel):
    passed: int
    totalCriteria: int
    passPercentage: float

    class Config:
        populate_by_name = True


class ProductionReadinessResponse(BaseModel):
    overallStatus: str  # "READY" | "WARNING" | "NOT_READY"
    summary: ReadinessSummary
    criteria: List[ReadinessCriterion]

    class Config:
        populate_by_name = True


# --- Curves ---

class ROCCurveResponse(BaseModel):
    fpr: List[float]
    tpr: List[float]
    thresholds: List[float]
    auc: float

    class Config:
        populate_by_name = True


class PRCurveResponse(BaseModel):
    precision: List[float]
    recall: List[float]
    thresholds: List[float]
    ap: float

    class Config:
        populate_by_name = True


class CurvesResponse(BaseModel):
    rocCurve: ROCCurveResponse
    prCurve: PRCurveResponse

    class Config:
        populate_by_name = True


# --- Learning Curve ---

class LearningCurveResponse(BaseModel):
    trainAccuracy: float
    testAccuracy: float
    overfittingRatio: float
    status: str  # "acceptable" | "overfitting" | "underfitting"

    class Config:
        populate_by_name = True


# --- Feature Importance ---

class FeatureDetail(BaseModel):
    name: str
    importancePercent: float
    correlationWithTarget: float
    correlationStrength: str  # "strong" | "moderate" | "weak" | "negligible"

    class Config:
        populate_by_name = True


class FeatureInteraction(BaseModel):
    feature1: str
    feature2: str
    interactionStrength: float
    interactionDirection: str  # "positive" | "negative"

    class Config:
        populate_by_name = True


class FeatureImportanceResponse(BaseModel):
    features: List[FeatureDetail]
    interactions: List[FeatureInteraction]

    class Config:
        populate_by_name = True


# --- Optimal Threshold ---

class OptimalThresholdResponse(BaseModel):
    optimalThreshold: float
    optimalProfit: float
    recommendation: str

    class Config:
        populate_by_name = True


# ============================================================================
# COMPLETE EVALUATION RESPONSE  (matches CompleteEvaluationResponse in TS)
# ============================================================================

class CompleteEvaluationResponse(BaseModel):
    """
    Complete evaluation response matching the frontend TypeScript interface.
    This is the single response that feeds ALL 5 tabs of the Model Evaluation screen.
    """
    thresholdEvaluation: ThresholdEvaluationResponse
    businessImpact: BusinessImpactResponse
    productionReadiness: ProductionReadinessResponse
    curves: Optional[CurvesResponse] = None
    learningCurve: Optional[LearningCurveResponse] = None
    featureImportance: Optional[FeatureImportanceResponse] = None
    optimalThreshold: Optional[OptimalThresholdResponse] = None
    overallScore: Optional[float] = None

    class Config:
        populate_by_name = True


# ============================================================================
# MODEL LIST RESPONSE (for model selector dropdown)
# ============================================================================

class TrainedModelInfo(BaseModel):
    id: str
    name: str
    algorithm: str
    accuracy: Optional[float] = None
    testScore: Optional[float] = None
    problemType: str
    trainedAt: Optional[str] = None

    class Config:
        populate_by_name = True


class TrainedModelsListResponse(BaseModel):
    models: List[TrainedModelInfo]
    bestModelId: Optional[str] = None
    totalModels: int

    class Config:
        populate_by_name = True