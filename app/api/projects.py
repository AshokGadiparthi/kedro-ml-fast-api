"""Projects API Routes"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from app.core.database import get_db
from app.models.models import Project
from app.schemas import ProjectCreate, ProjectResponse

router = APIRouter(prefix="", tags=["Projects"])

@router.get("/", response_model=list)
async def list_projects(db: Session = Depends(get_db)):
    """List all projects"""
    projects = db.query(Project).all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "owner_id": p.owner_id,
            "created_at": p.created_at.isoformat() if p.created_at else ""
        }
        for p in projects
    ]

@router.post("/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create new project"""
    new_project = Project(
        id=str(uuid4()),
        name=project.name,
        description=project.description,
        owner_id="user-001",
        created_at=datetime.now()
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    return {
        "id": new_project.id,
        "name": new_project.name,
        "description": new_project.description,
        "owner_id": new_project.owner_id,
        "created_at": new_project.created_at.isoformat()
    }
