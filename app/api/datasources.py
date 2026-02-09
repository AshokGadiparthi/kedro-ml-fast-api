"""Datasources API Routes"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from app.core.database import get_db
from app.models.models import Datasource

router = APIRouter(prefix="", tags=["Datasources"])

@router.get("/")
async def list_datasources(db: Session = Depends(get_db)):
    """List all datasources"""
    datasources = db.query(Datasource).all()
    return [
        {
            "id": ds.id,
            "name": ds.name,
            "type": ds.type,
            "description": ds.description,
            "project_id": ds.project_id,
            "host": ds.host,
            "port": ds.port,
            "database_name": ds.database_name,
            "username": ds.username,
            "created_at": ds.created_at.isoformat() if ds.created_at else ""
        }
        for ds in datasources
    ]

@router.post("/")
async def create_datasource(data: dict, db: Session = Depends(get_db)):
    """Create new datasource and save to database"""
    try:
        new_datasource = Datasource(
            id=str(uuid4()),
            name=data.get("name"),
            type=data.get("type"),
            description=data.get("description"),
            project_id=data.get("project_id"),
            host=data.get("host"),
            port=data.get("port"),
            database_name=data.get("database_name"),
            username=data.get("username"),
            password=data.get("password"),
            created_at=datetime.now()
        )
        db.add(new_datasource)
        db.flush()  # Ensure it's written
        db.commit()
        db.refresh(new_datasource)

        return {
            "id": new_datasource.id,
            "name": new_datasource.name,
            "type": new_datasource.type,
            "description": new_datasource.description,
            "project_id": new_datasource.project_id,
            "host": new_datasource.host,
            "port": new_datasource.port,
            "database_name": new_datasource.database_name,
            "username": new_datasource.username
        }
    except Exception as e:
        db.rollback()
        return {"error": str(e)}
