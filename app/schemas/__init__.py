"""
App Schemas Package
Exports all schemas for easy importing
"""

# Import from core schemas
from .schemas import (
    UserRegister,
    UserLogin,
    UserResponse,
    TokenResponse,
    TokenRefresh,
    ProjectCreate,
    ProjectResponse,
    DatasetCreate,
    DatasetResponse,
    ModelCreate,
    ModelResponse,
    ActivityCreate,
    ActivityResponse,
)

# Import EDA schemas
from .eda_schemas import (
    HealthResponse,
    JobStartResponse,
    JobStatusResponse,
    DataProfile,
    StatisticsResponse,
    QualityReportResponse,
    CorrelationResponse,
    VisualizationsResponse,
    FullReportResponse,
)

__all__ = [
    # Core schemas
    'UserRegister',
    'UserLogin',
    'UserResponse',
    'TokenResponse',
    'TokenRefresh',
    'ProjectCreate',
    'ProjectResponse',
    'DatasetCreate',
    'DatasetResponse',
    'ModelCreate',
    'ModelResponse',
    'ActivityCreate',
    'ActivityResponse',
    # EDA schemas
    'HealthResponse',
    'JobStartResponse',
    'JobStatusResponse',
    'DataProfile',
    'StatisticsResponse',
    'QualityReportResponse',
    'CorrelationResponse',
    'VisualizationsResponse',
    'FullReportReport',
]
