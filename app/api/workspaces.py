"""
Workspaces Routes - PHASE 0
=============================
Workspace CRUD operations

ENDPOINTS:
- GET    /api/workspaces                  List workspaces
- POST   /api/workspaces                  Create workspace
- GET    /api/workspaces/{workspace_id}   Get workspace details
- PUT    /api/workspaces/{workspace_id}   Update workspace
- DELETE /api/workspaces/{workspace_id}   Delete workspace
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from typing import List
import logging
import uuid

from app.core.database import get_db
from app.models.models import User, Workspace
from app.schemas import (
    WorkspaceCreate,
    WorkspaceResponse,
    WorkspaceUpdate,
    WorkspaceListResponse
)
from app.core.auth import verify_token, extract_token_from_header

logger = logging.getLogger(__name__)

router = APIRouter()

# ============================================================================
# HELPER - Get current user
# ============================================================================

def get_current_user(
        authorization: str = Header(None),
        db: Session = Depends(get_db)
) -> User:
    """Extract and verify user from Authorization header"""
    token = extract_token_from_header(authorization)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )

    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


# ============================================================================
# LIST WORKSPACES
# ============================================================================

@router.get("", response_model=List[WorkspaceResponse])
def list_workspaces(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """List all workspaces for current user"""
    logger.info(f"📋 Listing workspaces for user: {current_user.username}")

    workspaces = db.query(Workspace).filter(
        Workspace.owner_id == current_user.id,
        Workspace.is_active == True
    ).all()

    logger.info(f"✅ Found {len(workspaces)} workspaces")
    return workspaces


# ============================================================================
# CREATE WORKSPACE
# ============================================================================

@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(
        workspace_data: WorkspaceCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Create a new workspace"""
    logger.info(f"➕ Creating workspace: {workspace_data.name}")

    new_workspace = Workspace(
        id=str(uuid.uuid4()),
        owner_id=current_user.id,
        name=workspace_data.name,
        description=workspace_data.description or None,
        is_active=True
    )

    db.add(new_workspace)
    db.commit()
    db.refresh(new_workspace)

    logger.info(f"✅ Workspace created: {new_workspace.id}")
    return new_workspace


# ============================================================================
# GET WORKSPACE
# ============================================================================

@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(
        workspace_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get workspace details"""
    logger.info(f"🔍 Getting workspace: {workspace_id}")

    workspace = db.query(Workspace).filter(
        Workspace.id == workspace_id,
        Workspace.owner_id == current_user.id
    ).first()

    if not workspace:
        logger.warning(f"❌ Workspace not found: {workspace_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    logger.info(f"✅ Workspace found: {workspace.name}")
    return workspace


# ============================================================================
# UPDATE WORKSPACE
# ============================================================================

@router.put("/{workspace_id}", response_model=WorkspaceResponse)
def update_workspace(
        workspace_id: str,
        workspace_data: WorkspaceUpdate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Update a workspace"""
    logger.info(f"✏️ Updating workspace: {workspace_id}")

    workspace = db.query(Workspace).filter(
        Workspace.id == workspace_id,
        Workspace.owner_id == current_user.id
    ).first()

    if not workspace:
        logger.warning(f"❌ Workspace not found: {workspace_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    if workspace_data.name:
        workspace.name = workspace_data.name
    if workspace_data.description is not None:
        workspace.description = workspace_data.description

    db.commit()
    db.refresh(workspace)

    logger.info(f"✅ Workspace updated: {workspace.name}")
    return workspace


# ============================================================================
# DELETE WORKSPACE
# ============================================================================

@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workspace(
        workspace_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Delete a workspace"""
    logger.info(f"🗑️ Deleting workspace: {workspace_id}")

    workspace = db.query(Workspace).filter(
        Workspace.id == workspace_id,
        Workspace.owner_id == current_user.id
    ).first()

    if not workspace:
        logger.warning(f"❌ Workspace not found: {workspace_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    db.delete(workspace)
    db.commit()

    logger.info(f"✅ Workspace deleted: {workspace_id}")
    return None
