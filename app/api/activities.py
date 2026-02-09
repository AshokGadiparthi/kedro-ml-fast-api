"""
Activities API Endpoints - Complete
Log and track user activities with project tracking
FIXED: Includes project_id support
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
import logging
from app.core.database import get_db
from app.models.models import Activity
from app.schemas import ActivityCreate, ActivityResponse
import json  # Add at top if not there

logger = logging.getLogger(__name__)

router = APIRouter(tags=[""])  # ✅ NO PREFIX HERE


@router.get("/")
async def list_activities(
        db: Session = Depends(get_db),
        project_id: Optional[str] = Query(None),
        user_id: Optional[str] = Query(None),
        action: Optional[str] = Query(None),
        skip: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=100),
):
    """
    List activities with optional filters

    Args:
        db: Database session
        project_id: Filter by project ID
        user_id: Filter by user ID
        action: Filter by action type
        skip: Number of records to skip
        limit: Number of records to return

    Returns:
        List of activities
    """
    try:
        logger.info(f"📋 Listing activities (project_id={project_id})")

        query = db.query(Activity)

        if project_id:
            query = query.filter(Activity.project_id == project_id)
        if user_id:
            query = query.filter(Activity.user_id == user_id)
        if action:
            query = query.filter(Activity.action == action)

        # Order by most recent first
        activities = query.order_by(Activity.created_at.desc()).offset(skip).limit(limit).all()

        logger.info(f"✅ Found {len(activities)} activities")

        return [
            ActivityResponse(
                id=a.id,
                user_id=a.user_id,
                project_id=a.project_id,
                action=a.action,
                entity_type=a.entity_type,
                entity_id=a.entity_id,
                #entity_name=a.entity_name,
                details=json.loads(a.details) if a.details else None,  # ← PARSE JSON,
                created_at=a.created_at.isoformat()
            )
            for a in activities
        ]
    except Exception as e:
        logger.error(f"❌ Error listing activities: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/", response_model=ActivityResponse, status_code=status.HTTP_201_CREATED)
async def create_activity(
        activity_data: ActivityCreate,
        db: Session = Depends(get_db)
):
    """
    Create a new activity log entry

    Args:
        activity_data: Activity creation data
        db: Database session

    Returns:
        Created activity
    """
    try:
        logger.info(f"🆕 Creating activity: {activity_data.action} on {activity_data.entity_type}")

        activity = Activity(
            id=str(uuid4()),
            user_id="user_mock",  # In production: get from JWT token
            project_id=None,  # Can be set via activity_data if provided
            action=activity_data.action,
            entity_type=activity_data.entity_type,
            entity_id=activity_data.entity_id,
            #entity_name=getattr(activity_data, 'entity_name', None),
            details=activity_data.details,
            created_at=datetime.utcnow()
        )

        db.add(activity)
        db.commit()
        db.refresh(activity)

        logger.info(f"✅ Activity created: {activity.id}")

        return ActivityResponse(
            id=activity.id,
            user_id=activity.user_id,
            project_id=activity.project_id,
            action=activity.action,
            entity_type=activity.entity_type,
            entity_id=activity.entity_id,
            #entity_name=activity.entity_name,
            details=activity.details,
            created_at=activity.created_at.isoformat()
        )
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error creating activity: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{activity_id}", response_model=ActivityResponse)
async def get_activity(
        activity_id: str,
        db: Session = Depends(get_db)
):
    """
    Get a specific activity

    Args:
        activity_id: Activity ID
        db: Database session

    Returns:
        Activity data
    """
    try:
        logger.info(f"📖 Getting activity: {activity_id}")

        activity = db.query(Activity).filter(Activity.id == activity_id).first()

        if not activity:
            logger.warning(f"⚠️ Activity not found: {activity_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Activity {activity_id} not found"
            )

        return ActivityResponse(
            id=activity.id,
            user_id=activity.user_id,
            project_id=activity.project_id,
            action=activity.action,
            entity_type=activity.entity_type,
            entity_id=activity.entity_id,
            #entity_name=activity.entity_name,
            details=activity.details,
            created_at=activity.created_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting activity: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{activity_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_activity(
        activity_id: str,
        db: Session = Depends(get_db)
):
    """
    Delete an activity (typically not used, but available for testing)

    Args:
        activity_id: Activity ID
        db: Database session
    """
    try:
        logger.info(f"🗑️ Deleting activity: {activity_id}")

        activity = db.query(Activity).filter(Activity.id == activity_id).first()

        if not activity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Activity {activity_id} not found"
            )

        db.delete(activity)
        db.commit()

        logger.info(f"✅ Activity deleted: {activity_id}")

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error deleting activity: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/project/{project_id}/summary")
async def get_project_activity_summary(
        project_id: str,
        db: Session = Depends(get_db)
):
    """
    Get activity summary for a project

    Args:
        project_id: Project ID
        db: Database session

    Returns:
        Activity summary statistics
    """
    try:
        logger.info(f"📊 Getting activity summary for project: {project_id}")

        activities = db.query(Activity).filter(Activity.project_id == project_id).all()

        # Count by action
        action_counts = {}
        for activity in activities:
            action = activity.action
            action_counts[action] = action_counts.get(action, 0) + 1

        # Count by entity type
        entity_counts = {}
        for activity in activities:
            entity = activity.entity_type
            entity_counts[entity] = entity_counts.get(entity, 0) + 1

        return {
            "project_id": project_id,
            "total_activities": len(activities),
            "by_action": action_counts,
            "by_entity": entity_counts,
            "last_activity": activities[-1].created_at.isoformat() if activities else None
        }
    except Exception as e:
        logger.error(f"❌ Error getting summary: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))