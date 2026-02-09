"""
Health Check API Endpoints
"""

from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """Application health check"""
    return {
        "status": "healthy",
        "message": "FastAPI is running",
        "version": "1.0.0"
    }
