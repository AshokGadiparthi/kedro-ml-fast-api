"""
Model Registry - API Endpoints
================================
REST API for the Model Registry.

Routes:
    GET    /api/v1/models/registry                         → List models
    GET    /api/v1/models/registry/stats                   → Dashboard stats
    POST   /api/v1/models/registry                         → Register a model
    GET    /api/v1/models/registry/{model_id}              → Get model detail
    PATCH  /api/v1/models/registry/{model_id}              → Update model metadata
    DELETE /api/v1/models/registry/{model_id}              → Delete model
    GET    /api/v1/models/registry/{model_id}/versions     → Get all versions
    POST   /api/v1/models/registry/{model_id}/deploy       → Deploy model
    POST   /api/v1/models/registry/{model_id}/promote      → Promote model status
    POST   /api/v1/models/registry/{model_id}/archive      → Archive model
    GET    /api/v1/models/registry/{model_id}/artifacts     → List artifacts
    GET    /api/v1/models/registry/artifacts/{artifact_id}/download → Download artifact
    GET    /api/v1/models/registry/auto-detect              → Preview auto-detected training results
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Optional
from datetime import datetime
import logging

from app.core.registry_manager import RegistryManager
from app.schemas.registry_schemas import (
    RegisterModelRequest,
    UpdateModelRequest,
    DeployModelRequest,
    PromoteModelRequest,
    RegisteredModelResponse,
    ModelDetailResponse,
    ModelListResponse,
    ModelStatsResponse,
    DeploymentResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Model Registry"])

# Initialize manager
registry = RegistryManager()

logger.info("✅ Model Registry router created")


# ============================================================================
# LIST MODELS
# ============================================================================

@router.get("/", response_model=ModelListResponse)
async def list_models(
        project_id: str = Query(..., description="Project ID (required)"),
        status: Optional[str] = Query(None, description="Filter: draft, staging, production, archived"),
        search: Optional[str] = Query(None, description="Search in name, description, algorithm"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
):
    """
    List all registered models for a project.

    Used by the Model Registry grid/list view.

    Example:
        GET /api/v1/models/registry?project_id=abc123
        GET /api/v1/models/registry?project_id=abc123&status=production&search=loan
    """
    logger.info(f"📋 List models: project={project_id}, status={status}, search={search}")

    try:
        result = registry.list_models(
            project_id=project_id,
            status=status,
            search=search,
            limit=limit,
            offset=offset,
        )
        return result

    except Exception as e:
        logger.error(f"❌ Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STATS (Dashboard Cards)
# ============================================================================

@router.get("/stats", response_model=ModelStatsResponse)
async def get_model_stats(
        project_id: str = Query(..., description="Project ID"),
):
    """
    Get summary stats for Model Registry dashboard cards.

    Returns counts: total_models, deployed, production, staging, draft, archived.

    Example:
        GET /api/v1/models/registry/stats?project_id=abc123
    """
    logger.info(f"📊 Get registry stats: project={project_id}")

    try:
        return registry.get_stats(project_id)
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AUTO-DETECT (Preview training results before registering)
# ============================================================================

@router.get("/auto-detect")
async def auto_detect_results():
    """
    Preview auto-detected training results from Kedro output files.

    Call this BEFORE registering to see what will be auto-filled.
    Useful for debugging or showing a preview in the UI.

    Example:
        GET /api/v1/models/registry/auto-detect
    """
    logger.info("🔍 Auto-detecting training results...")

    try:
        detected = registry.auto_detect_training_results()
        return {
            "status": "ok",
            "detected": detected,
            "message": "Training results detected from Kedro output files"
        }
    except Exception as e:
        logger.error(f"❌ Error auto-detecting: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# REGISTER MODEL
# ============================================================================

@router.post("/", status_code=201)
async def register_model(request: RegisterModelRequest):
    """
    Register a new model in the registry.

    Auto-detects training results from Kedro output files.
    If a model with the same name already exists in the project,
    a new version is created automatically.

    Example (minimal):
        POST /api/v1/models/registry
        {"project_id": "abc123", "name": "Loan Classifier"}

    Example (with details):
        POST /api/v1/models/registry
        {
            "project_id": "abc123",
            "name": "Loan Classifier",
            "job_id": "bf5df478-...",
            "description": "Classification model for loan approval",
            "tags": ["finance", "classification"]
        }
    """
    logger.info(f"📝 Register model: name={request.name}, project={request.project_id}")

    try:
        result = registry.register_model(
            project_id=request.project_id,
            name=request.name,
            description=request.description,
            job_id=request.job_id,
            algorithm=request.algorithm,
            problem_type=request.problem_type,
            dataset_id=request.dataset_id,
            dataset_name=request.dataset_name,
            accuracy=request.accuracy,
            train_score=request.train_score,
            test_score=request.test_score,
            precision=request.precision,
            recall=request.recall,
            f1_score=request.f1_score,
            tags=request.tags,
            created_by=request.created_by,
            collection_id=request.collection_id,
            dataset_path=request.dataset_path,
        )

        return {
            "status": "registered",
            "message": f"Model '{request.name}' registered successfully",
            "model": result,
        }

    except Exception as e:
        logger.error(f"❌ Error registering model: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# GET MODEL DETAIL
# ============================================================================

@router.get("/{model_id}")
async def get_model_detail(model_id: str):
    """
    Get full model details including all versions and artifacts.

    Used for the Model Details modal (6 tabs).

    Example:
        GET /api/v1/models/registry/abc123-def456
    """
    logger.info(f"🔍 Get model detail: {model_id}")

    try:
        model = registry.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return model

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UPDATE MODEL METADATA
# ============================================================================

@router.patch("/{model_id}")
async def update_model(model_id: str, request: UpdateModelRequest):
    """
    Update model metadata (name, description, tags, labels).

    Example:
        PATCH /api/v1/models/registry/abc123
        {"name": "Updated Name", "tags": ["v2", "production"]}
    """
    logger.info(f"✏️ Update model: {model_id}")

    try:
        updates = request.dict(exclude_unset=True)
        model = registry.update_model(model_id, updates)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return {
            "status": "updated",
            "model": model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DELETE MODEL
# ============================================================================

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a model and all its versions and artifacts.

    Example:
        DELETE /api/v1/models/registry/abc123
    """
    logger.info(f"🗑️ Delete model: {model_id}")

    try:
        deleted = registry.delete_model(model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return {
            "status": "deleted",
            "message": f"Model {model_id} deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL VERSIONS
# ============================================================================

@router.get("/{model_id}/versions")
async def get_model_versions(model_id: str):
    """
    Get all versions of a model.

    Example:
        GET /api/v1/models/registry/abc123/versions
    """
    logger.info(f"📋 Get versions: model={model_id}")

    try:
        versions = registry.get_model_versions(model_id)
        return {
            "model_id": model_id,
            "versions": versions,
            "total": len(versions),
        }

    except Exception as e:
        logger.error(f"❌ Error getting versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DEPLOY MODEL
# ============================================================================

@router.post("/{model_id}/deploy")
async def deploy_model(model_id: str, request: DeployModelRequest):
    """
    Deploy a model version.

    Example:
        POST /api/v1/models/registry/abc123/deploy
        {"version": "v1.0", "environment": "production"}
    """
    logger.info(f"🚀 Deploy model: {model_id}, version={request.version}, env={request.environment}")

    try:
        model = registry.deploy_model(
            model_id=model_id,
            version=request.version,
            environment=request.environment,
        )

        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return {
            "status": "deployed",
            "model_id": model_id,
            "version": request.version or model.get("current_version"),
            "environment": request.environment,
            "deployed_at": datetime.utcnow().isoformat(),
            "message": f"Model deployed to {request.environment}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deploying model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PROMOTE MODEL
# ============================================================================

@router.post("/{model_id}/promote")
async def promote_model(model_id: str, request: PromoteModelRequest):
    """
    Promote model status (draft → staging → production).

    Example:
        POST /api/v1/models/registry/abc123/promote
        {"target_status": "production"}
    """
    logger.info(f"⬆️ Promote model: {model_id} → {request.target_status}")

    try:
        model = registry.promote_model(
            model_id=model_id,
            target_status=request.target_status,
            version=request.version,
        )

        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return {
            "status": "promoted",
            "model": model,
            "message": f"Model promoted to {request.target_status}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error promoting model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ARCHIVE MODEL
# ============================================================================

@router.post("/{model_id}/archive")
async def archive_model(model_id: str):
    """
    Archive a model (removes from active listing, stops deployment).

    Example:
        POST /api/v1/models/registry/abc123/archive
    """
    logger.info(f"📦 Archive model: {model_id}")

    try:
        model = registry.archive_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return {
            "status": "archived",
            "model": model,
            "message": "Model archived successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error archiving model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ARTIFACTS
# ============================================================================

@router.get("/{model_id}/artifacts")
async def get_model_artifacts(
        model_id: str,
        version: Optional[str] = Query(None, description="Filter by version (defaults to current)"),
):
    """
    List all artifacts for a model version.

    Example:
        GET /api/v1/models/registry/abc123/artifacts
        GET /api/v1/models/registry/abc123/artifacts?version=v1.0
    """
    logger.info(f"📂 Get artifacts: model={model_id}, version={version}")

    try:
        artifacts = registry.get_artifacts(model_id, version)
        return {
            "model_id": model_id,
            "version": version or "current",
            "artifacts": artifacts,
            "total": len(artifacts),
        }

    except Exception as e:
        logger.error(f"❌ Error getting artifacts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/artifacts/{artifact_id}/download")
async def download_artifact(artifact_id: str):
    """
    Download a specific artifact file.

    Example:
        GET /api/v1/models/registry/artifacts/art123/download
    """
    logger.info(f"📥 Download artifact: {artifact_id}")

    try:
        file_path = registry.get_artifact_path(artifact_id)
        if not file_path:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

        return FileResponse(
            path=file_path,
            filename=file_path.split("/")[-1],
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading artifact: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


logger.info("✅ Model Registry router fully initialized")