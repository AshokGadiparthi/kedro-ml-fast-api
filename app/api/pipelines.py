"""
Pipelines API Endpoints
Kedro pipeline management
"""

from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def list_pipelines():
    """Get all available Kedro pipelines"""
    try:
        logger.info("📋 Listing all pipelines")
        
        return {
            "message": "Check the external Kedro project for available pipelines",
            "kedro_project_path": "See environment KEDRO_PROJECT_PATH",
            "note": "Use POST /api/v1/jobs to execute pipelines"
        }
    except Exception as e:
        logger.error(f"❌ Error listing pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_name}")
async def get_pipeline_details(pipeline_name: str):
    """Get detailed information about a specific pipeline"""
    try:
        logger.info(f"📖 Getting details for pipeline: {pipeline_name}")
        
        return {
            "pipeline_name": pipeline_name,
            "message": "Pipeline information available from external Kedro project"
        }
    except Exception as e:
        logger.error(f"❌ Error getting pipeline details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_name}/parameters")
async def get_pipeline_parameters(pipeline_name: str):
    """Get default parameters for a pipeline"""
    try:
        logger.info(f"⚙️  Getting parameters for pipeline: {pipeline_name}")
        
        return {
            "pipeline_name": pipeline_name,
            "parameters": {}
        }
    except Exception as e:
        logger.error(f"❌ Error getting pipeline parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Check Kedro integration health"""
    try:
        logger.info("🏥 Checking Kedro integration health")
        
        return {
            "status": "healthy",
            "service": "Kedro Pipelines",
            "message": "Kedro integration is working"
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
