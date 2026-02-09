"""
Model Registry - Pydantic Schemas
===================================
Request/Response models for the Model Registry API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class RegisterModelRequest(BaseModel):
    """Request to register a new model from a completed training job"""

    # Required
    project_id: str
    name: str

    # Optional - can be auto-detected from job results
    description: Optional[str] = None
    job_id: Optional[str] = None
    algorithm: Optional[str] = None
    problem_type: Optional[str] = None  # classification, regression, clustering

    # Source tracking
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None

    # Metrics (if not auto-detected from files)
    accuracy: Optional[float] = None
    train_score: Optional[float] = None
    test_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Metadata
    tags: Optional[List[str]] = None
    created_by: Optional[str] = None

    collection_id: Optional[str] = None
    dataset_path: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "project_id": "proj_abc123",
                "name": "Loan Approval Classifier",
                "job_id": "bf5df478-6cf8-4dea-914a-3d110ba5ffb2",
                "description": "Classification model for loan approval",
                "tags": ["finance", "classification"]
            }
        }


class UpdateModelRequest(BaseModel):
    """Request to update model metadata"""
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    labels: Optional[Dict[str, str]] = None


class DeployModelRequest(BaseModel):
    """Request to deploy a model version"""
    version: Optional[str] = None          # Defaults to current version
    environment: str = "production"        # production, staging, development
    notes: Optional[str] = None


class PromoteModelRequest(BaseModel):
    """Request to promote model status"""
    version: Optional[str] = None
    target_status: str = "production"      # staging, production


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class ArtifactResponse(BaseModel):
    """Single artifact info"""
    id: str
    artifact_name: str
    artifact_type: str
    file_path: str
    file_size_bytes: Optional[int] = None
    created_at: Optional[str] = None


class VersionResponse(BaseModel):
    """Single model version info"""
    id: str
    version: str
    version_number: int
    status: str
    algorithm: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    train_score: Optional[float] = None
    test_score: Optional[float] = None
    roc_auc: Optional[float] = None
    is_current: bool = False
    job_id: Optional[str] = None
    model_size_mb: Optional[float] = None
    training_time_seconds: Optional[float] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    tags: Optional[List[str]] = None
    artifacts: Optional[List[ArtifactResponse]] = None


class RegisteredModelResponse(BaseModel):
    """Single registered model info (for listing)"""
    id: str
    project_id: str
    name: str
    description: Optional[str] = None
    problem_type: Optional[str] = None
    current_version: Optional[str] = None
    latest_version: Optional[str] = None
    total_versions: int = 1
    status: str = "draft"
    best_accuracy: Optional[float] = None
    best_algorithm: Optional[str] = None
    is_deployed: bool = False
    deployment_url: Optional[str] = None
    deployed_version: Optional[str] = None
    deployed_at: Optional[str] = None
    source_dataset_id: Optional[str] = None
    source_dataset_name: Optional[str] = None
    training_job_id: Optional[str] = None
    collection_id: Optional[str] = None
    dataset_path: Optional[str] = None
    tags: Optional[List[str]] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    versions: Optional[List[VersionResponse]] = None


class ModelDetailResponse(RegisteredModelResponse):
    """Full model detail (includes versions with artifacts)"""
    versions: Optional[List[VersionResponse]] = None
    feature_names: Optional[List[str]] = None
    feature_importances: Optional[Dict[str, float]] = None
    training_config: Optional[Dict[str, Any]] = None
    confusion_matrix: Optional[Any] = None


class ModelListResponse(BaseModel):
    """Response for listing models"""
    models: List[RegisteredModelResponse]
    total: int
    limit: int
    offset: int


class ModelStatsResponse(BaseModel):
    """Summary stats for registry dashboard cards"""
    total_models: int = 0
    deployed: int = 0
    production: int = 0
    staging: int = 0
    draft: int = 0
    archived: int = 0


class DeploymentResponse(BaseModel):
    """Response after deploying a model"""
    model_id: str
    version: str
    environment: str
    status: str
    deployed_at: str
    message: str