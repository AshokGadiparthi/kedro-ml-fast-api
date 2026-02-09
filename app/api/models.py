"""Models API Routes"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from app.core.database import get_db
from app.schemas import ModelCreate, ModelResponse

router = APIRouter(prefix="", tags=["Models"])

@router.get("/", response_model=list)
async def list_models(db: Session = Depends(get_db)):
    """List all models"""
    # Return empty list for now (models table may not have full implementation)
    return []

@router.post("/", response_model=ModelResponse)
async def create_model(model: ModelCreate, db: Session = Depends(get_db)):
    """Create new model"""
    return {
        "id": str(uuid4()),
        "name": model.name,
        "project_id": model.project_id,
        "description": model.description,
        "model_type": model.model_type,
        "created_at": datetime.now().isoformat()
    }
