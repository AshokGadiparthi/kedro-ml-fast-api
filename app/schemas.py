"""
Pydantic Schemas - PHASE 0
============================
Request/Response validation and data modeling

WHY PYDANTIC?
- Validates incoming data from frontend
- Converts JSON to Python objects
- Type checking and error messages
- Automatic OpenAPI documentation
- Easy serialization to JSON
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# ============================================================================
# USER SCHEMAS
# ============================================================================

class UserRegister(BaseModel):
    """User registration request"""
    
    email: EmailStr = Field(
        ...,
        description="Email address (must be valid email)"
    )
    username: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Username (3-100 characters)"
    )
    password: str = Field(
        ...,
        min_length=6,
        description="Password (minimum 6 characters)"
    )
    full_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Full name (optional)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "username": "john_doe",
                "password": "securepass123",
                "full_name": "John Doe"
            }
        }


class UserLogin(BaseModel):
    """User login request"""
    
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., description="Password")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepass123"
            }
        }


class UserResponse(BaseModel):
    """User response (sent to frontend)"""
    
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    username: str = Field(..., description="Username")
    full_name: Optional[str] = Field(None, description="Full name")
    is_active: bool = Field(..., description="Is account active")
    created_at: datetime = Field(..., description="Registration date")
    
    class Config:
        from_attributes = True  # Convert SQLAlchemy objects to Pydantic
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@example.com",
                "username": "john_doe",
                "full_name": "John Doe",
                "is_active": True,
                "created_at": "2024-01-15T10:30:00"
            }
        }


class TokenResponse(BaseModel):
    """Token response after successful login"""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserResponse = Field(..., description="User information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
                "token_type": "bearer",
                "user": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "email": "user@example.com",
                    "username": "john_doe",
                    "full_name": "John Doe",
                    "is_active": True,
                    "created_at": "2024-01-15T10:30:00"
                }
            }
        }


# ============================================================================
# WORKSPACE SCHEMAS
# ============================================================================

class WorkspaceCreate(BaseModel):
    """Create workspace request"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Workspace name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Workspace description"
    )
    slug: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern="^[a-z0-9-]+$",
        description="URL-friendly slug (lowercase, hyphens only)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "My ML Workspace",
                "description": "For development and testing",
                "slug": "my-ml-workspace"
            }
        }


class WorkspaceUpdate(BaseModel):
    """Update workspace request"""
    
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New workspace name"
    )
    description: Optional[str] = Field(
        default=None,
        description="New description"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Updated Workspace Name",
                "description": "Updated description"
            }
        }


class WorkspaceResponse(BaseModel):
    """Workspace response (sent to frontend)"""
    
    id: str = Field(..., description="Workspace ID")
    name: str = Field(..., description="Workspace name")
    slug: str = Field(..., description="URL-friendly slug")
    description: Optional[str] = Field(None, description="Description")
    owner_id: str = Field(..., description="Owner user ID")
    is_active: bool = Field(..., description="Is workspace active")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "My ML Workspace",
                "slug": "my-ml-workspace",
                "description": "For development and testing",
                "owner_id": "550e8400-e29b-41d4-a716-446655440001",
                "is_active": True,
                "created_at": "2024-01-15T10:30:00",
                "updated_at": "2024-01-15T10:30:00"
            }
        }


class WorkspaceListResponse(BaseModel):
    """Response for listing workspaces"""
    
    count: int = Field(..., description="Number of workspaces")
    workspaces: List[WorkspaceResponse] = Field(..., description="List of workspaces")


# ============================================================================
# PROJECT SCHEMAS - PHASE 1
# ============================================================================

class ProjectCreate(BaseModel):
    """Create project request"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Project name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Project description"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Data Analysis Project",
                "description": "Analyzing sales data"
            }
        }


class ProjectUpdate(BaseModel):
    """Update project request"""
    
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New project name"
    )
    description: Optional[str] = Field(
        default=None,
        description="New description"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Updated Project Name",
                "description": "Updated description"
            }
        }


class ProjectResponse(BaseModel):
    """Project response (sent to frontend)"""
    
    id: str = Field(..., description="Project ID")
    owner_id: str = Field(..., description="Owner User ID")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Description")
    is_active: bool = Field(..., description="Is project active")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440002",
                "owner_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Data Analysis Project",
                "description": "Analyzing sales data",
                "is_active": True,
                "created_at": "2024-01-31T10:00:00",
                "updated_at": "2024-01-31T10:00:00"
            }
        }


class ProjectListResponse(BaseModel):
    """Response for listing projects"""
    
    count: int = Field(..., description="Number of projects")
    projects: List[ProjectResponse] = Field(..., description="List of projects")


# ============================================================================
# DATASOURCE SCHEMAS - PHASE 2
# ============================================================================

class DatasourceCreate(BaseModel):
    """Create datasource request"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Datasource name"
    )
    type: str = Field(
        ...,
        description="Type: csv, excel, json, parquet, bigquery, snowflake, postgresql, mysql, etc"
    )
    description: Optional[str] = Field(
        default=None,
        description="Datasource description"
    )
    connection_config: Optional[dict] = Field(
        default=None,
        description="Connection details (for database sources)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Customer Data",
                "type": "csv",
                "description": "Customer information CSV file",
                "connection_config": None
            }
        }


class DatasourceResponse(BaseModel):
    """Datasource response"""
    
    id: str = Field(..., description="Datasource ID")
    project_id: str = Field(..., description="Project ID")
    name: str = Field(..., description="Datasource name")
    type: str = Field(..., description="Datasource type")
    description: Optional[str] = Field(None, description="Description")
    file_path: Optional[str] = Field(None, description="Path to uploaded file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_name: Optional[str] = Field(None, description="file name")
    is_active: bool = Field(..., description="Is active")
    is_connected: bool = Field(..., description="Connection test passed")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440004",
                "project_id": "550e8400-e29b-41d4-a716-446655440002",
                "name": "Customer Data",
                "type": "csv",
                "description": "Customer information",
                "file_path": "/uploads/customer_data.csv",
                "file_name": "File Name",
                "file_size": 1024000,
                "is_active": True,
                "is_connected": True,
                "created_at": "2024-01-31T11:30:00",
                "updated_at": "2024-01-31T11:30:00"
            }
        }


class DatasetCreate(BaseModel):
    """Create dataset request"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Dataset name"
    )
    datasource_id: str = Field(
        ...,
        description="Datasource ID to create dataset from"
    )
    description: Optional[str] = Field(
        default=None,
        description="Dataset description"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Customer Data v1",
                "datasource_id": "550e8400-e29b-41d4-a716-446655440004",
                "description": "Processed customer data ready for ML"
            }
        }


class DatasetResponse(BaseModel):
    """Dataset response"""
    
    id: str = Field(..., description="Dataset ID")
    project_id: str = Field(..., description="Project ID")
    datasource_id: Optional[str] = Field(None, description="Source datasource ID")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Description")
    row_count: Optional[int] = Field(None, description="Number of rows")
    column_count: Optional[int] = Field(None, description="Number of columns")
    columns_info: Optional[dict] = Field(None, description="Column information")
    missing_values_count: Optional[int] = Field(None, description="Missing values count")
    duplicate_rows_count: Optional[int] = Field(None, description="Duplicate rows count")
    is_processed: bool = Field(..., description="Is processed and ready")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440005",
                "project_id": "550e8400-e29b-41d4-a716-446655440002",
                "datasource_id": "550e8400-e29b-41d4-a716-446655440004",
                "name": "Customer Data v1",
                "description": "Processed customer data ready for ML",
                "row_count": 10000,
                "column_count": 15,
                "columns_info": {"name": "text", "age": "integer", "email": "text"},
                "missing_values_count": 42,
                "duplicate_rows_count": 5,
                "is_processed": True,
                "created_at": "2024-01-31T11:40:00",
                "updated_at": "2024-01-31T11:40:00"
            }
        }


# ============================================================================
# MODEL SCHEMAS - PHASE 3
# ============================================================================

class ModelCreate(BaseModel):
    """Create model request"""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Model name"
    )
    algorithm: str = Field(
        ...,
        description="Algorithm used"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Customer Churn v1",
                "algorithm": "Random Forest"
            }
        }


class ModelResponse(BaseModel):
    """Model response"""
    
    id: str = Field(..., description="Model ID")
    project_id: str = Field(..., description="Project ID")
    name: str = Field(..., description="Model name")
    algorithm: str = Field(..., description="Algorithm")
    accuracy: Optional[float] = Field(None, description="Accuracy score")
    precision: Optional[float] = Field(None, description="Precision score")
    recall: Optional[float] = Field(None, description="Recall score")
    f1_score: Optional[float] = Field(None, description="F1 score")
    status: str = Field(..., description="Model status")
    training_duration_seconds: Optional[int] = Field(None, description="Training time")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "model-uuid",
                "project_id": "project-uuid",
                "name": "Customer Churn v1",
                "algorithm": "Random Forest",
                "accuracy": 0.92,
                "precision": 0.90,
                "recall": 0.94,
                "f1_score": 0.92,
                "status": "Trained",
                "training_duration_seconds": 3600,
                "created_at": "2024-01-31T12:00:00",
                "updated_at": "2024-01-31T12:00:00"
            }
        }


# ============================================================================
# ACTIVITY SCHEMAS - PHASE 3
# ============================================================================

class ActivityCreate(BaseModel):
    """Create activity request"""
    
    action: str = Field(..., description="Action type")
    target_type: str = Field(..., description="Target type (project, dataset, model, etc)")
    target_id: Optional[str] = Field(None, description="Target ID")
    description: Optional[str] = Field(None, description="Description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": "model_trained",
                "target_type": "model",
                "target_id": "model-uuid",
                "description": "Model trained successfully"
            }
        }


class ActivityResponse(BaseModel):
    """Activity response"""
    
    id: str = Field(..., description="Activity ID")
    project_id: str = Field(..., description="Project ID")
    action: str = Field(..., description="Action")
    target_type: str = Field(..., description="Target type")
    target_id: Optional[str] = Field(None, description="Target ID")
    description: Optional[str] = Field(None, description="Description")
    created_at: str = Field(..., description="When it happened (ISO format)")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "activity-uuid",
                "project_id": "project-uuid",
                "action": "model_trained",
                "target_type": "model",
                "target_id": "model-uuid",
                "description": "Model trained successfully",
                "created_at": "2024-01-31T12:00:00"
            }
        }


# ============================================================================
# ERROR SCHEMAS
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    
    detail: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")


class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    
    detail: List[dict] = Field(..., description="List of validation errors")


# ============================================================================
# UTILITY SCHEMAS
# ============================================================================

class MessageResponse(BaseModel):
    """Simple message response"""
    
    message: str = Field(..., description="Response message")


class SuccessResponse(BaseModel):
    """Success response"""
    
    success: bool = Field(default=True)
    message: str = Field(..., description="Success message")

# ============================================================================
# WORKSPACE SCHEMAS
# ============================================================================

class WorkspaceCreate(BaseModel):
    """Create workspace request"""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Workspace name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Workspace description"
    )
    slug: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern="^[a-z0-9-]+$",
        description="URL-friendly slug (lowercase, hyphens only)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My ML Workspace",
                "description": "For development and testing",
                "slug": "my-ml-workspace"
            }
        }


class WorkspaceUpdate(BaseModel):
    """Update workspace request"""

    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New workspace name"
    )
    description: Optional[str] = Field(
        default=None,
        description="New description"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Updated Workspace Name",
                "description": "Updated description"
            }
        }


class WorkspaceResponse(BaseModel):
    """Workspace response (sent to frontend)"""

    id: str = Field(..., description="Workspace ID")
    name: str = Field(..., description="Workspace name")
    slug: str = Field(..., description="URL-friendly slug")
    description: Optional[str] = Field(None, description="Description")
    owner_id: str = Field(..., description="Owner user ID")
    is_active: bool = Field(..., description="Is workspace active")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "My ML Workspace",
                "slug": "my-ml-workspace",
                "description": "For development and testing",
                "owner_id": "550e8400-e29b-41d4-a716-446655440001",
                "is_active": True,
                "created_at": "2024-01-15T10:30:00",
                "updated_at": "2024-01-15T10:30:00"
            }
        }


class WorkspaceListResponse(BaseModel):
    """Response for listing workspaces"""

    count: int = Field(..., description="Number of workspaces")
    workspaces: List[WorkspaceResponse] = Field(..., description="List of workspaces")