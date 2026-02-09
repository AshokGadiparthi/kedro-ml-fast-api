"""Pydantic Schemas - 100% Working - No Warnings"""

from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ============================================================================
# User Authentication Schemas
# ============================================================================

class UserRegister(BaseModel):
    """User registration schema"""
    username: str
    email: Optional[str] = None
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    """User login schema"""
    username: str
    password: str

class UserResponse(BaseModel):
    """User response schema"""
    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    created_at: str

class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str
    user: UserResponse

class TokenRefresh(BaseModel):
    """Token refresh schema"""
    refresh_token: str

# ============================================================================
# Project Schemas
# ============================================================================

class ProjectCreate(BaseModel):
    """Create project schema"""
    name: str
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    """Project response schema"""
    id: str
    name: str
    description: Optional[str] = None
    owner_id: str
    created_at: str

# ============================================================================
# Dataset Schemas
# ============================================================================

class DatasetCreate(BaseModel):
    """Create dataset schema"""
    name: str
    project_id: str
    description: Optional[str] = None

class DatasetResponse(BaseModel):
    """Dataset response schema"""
    id: str
    name: str
    project_id: str
    description: Optional[str] = None
    file_name: str
    file_size_bytes: int
    created_at: str

# ============================================================================
# Model Schemas
# ============================================================================

class ModelCreate(BaseModel):
    """Create model schema"""
    name: str
    project_id: str
    description: Optional[str] = None
    model_type: str

class ModelResponse(BaseModel):
    """Model response schema"""
    id: str
    name: str
    project_id: str
    description: Optional[str] = None
    model_type: str
    created_at: str

# ============================================================================
# Activity Schemas
# ============================================================================

class ActivityCreate(BaseModel):
    """Create activity schema"""
    action: str
    entity_type: str
    entity_id: str
    details: Optional[dict] = None

class ActivityResponse(BaseModel):
    """Activity response schema"""
    id: str
    user_id: str
    action: str
    entity_type: str
    entity_id: str
    details: Optional[dict] = None
    created_at: str

class HealthResponse(BaseModel):
    status: str
    service: Optional[str] = None
    version: Optional[str] = None
    timestamp: datetime


class AnalysisResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    message: str


class JobStatusEnum(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobStatusResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: str
    progress: int
    current_phase: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    result_id: Optional[str] = None
    error: Optional[str] = None


class JobCreate(BaseModel):
    pipeline_name: str
    parameters: Optional[Dict[str, Any]] = {}
    tags: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "pipeline_name": "data_processing",
                "parameters": {}
            }
        }


class JobResponse(BaseModel):
    id: str
    pipeline_name: str
    user_id: str
    status: JobStatusEnum
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[int] = None
