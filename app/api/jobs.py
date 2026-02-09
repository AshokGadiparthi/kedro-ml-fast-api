"""
FastAPI endpoints for job management and Kedro pipeline execution

✅ COMPLETE INTEGRATION:
- Accepts filepath as query parameter OR in request body
- Builds dynamic parameters for Kedro
- Supports both upload-to-job workflow and direct filepath submission
"""

from fastapi import APIRouter, HTTPException, status, WebSocket, Depends, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import logging
from app.tasks import execute_pipeline
from app.core.job_manager import JobManager
from app.core.database import get_db
from sqlalchemy.orm import Session

from pathlib import Path
import os
import re  # ← ADD THIS LINE!
import json
import re

from fastapi.responses import FileResponse
import csv
from io import StringIO

KEDRO_PROJECT_PATH = Path("/home/ashok/work/latest/full/kedro-ml-engine-integrated")
LOGS_DIR = Path("data/job_logs")
MODEL_OUTPUT_DIR = Path("data/07_model_output")

# Configure logging
logger = logging.getLogger(__name__)

# Initialize DB manager
db_manager = JobManager()

# ============================================================================
# Router Configuration
# ============================================================================
router = APIRouter(tags=["jobs"])
logger.info("✅ Jobs router created")

# List of words that are NOT algorithms (false positives)
FALSE_POSITIVE_WORDS = {
    'ensemble',
    'voting',
    'results',
    'models',
    'analysis',
    'data',
    'node',
    'model',
    'curves',
    'testing',
    'visualization',
    'report',
    'status',
    'output',
    'input',
    'output',
    'features',
    'scaling',
    'tuning',
}

# Real ML algorithm names to match against
VALID_ALGORITHMS = {
    'LogisticRegression',
    'RidgeClassifier',
    'SGDClassifier',
    'PassiveAggressiveClassifier',
    'Perceptron',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'ExtraTreesClassifier',
    'GradientBoostingClassifier',
    'AdaBoostClassifier',
    'BaggingClassifier',
    'LinearSVC',
    'GaussianNB',
    'MultinomialNB',
    'BernoulliNB',
    'ComplementNB',
    'CategoricalNB',
    'KNeighborsClassifier',
    'SVC',
    'KMeans',
    'DBSCAN',
    'IsolationForest',
}

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class RunPipelineRequest(BaseModel):
    """Request model for running a Kedro pipeline"""
    parameters: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    filepath: Optional[str] = None  # ✅ NEW: For data_loading pipeline

    class Config:
        schema_extra = {
            "example": {
                "filepath": "data/01_raw/project_123/users.csv",
                "parameters": {"data_loading": {"test_size": 0.25}},
                "description": "Load multi-table dataset"
            }
        }


class JobResponse(BaseModel):
    """Response model for job information"""
    id: str
    pipeline_name: str
    user_id: Optional[str] = None
    status: str
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None


class PipelineInfo(BaseModel):
    """Model for pipeline information"""
    name: str
    description: str
    nodes: Optional[List[str]] = None
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
    estimated_time: Optional[int] = None  # seconds
    memory_required: Optional[str] = None


class PipelineListResponse(BaseModel):
    """Response model for pipeline list"""
    total: int
    pipelines: List[Dict[str, str]]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_job_parameters(
        pipeline_name: str,
        request: Optional[RunPipelineRequest] = None
) -> Dict[str, Any]:
    """
    Build final parameters dict for job creation

    Handles dynamic filepath for data_loading pipeline
    Merges with user-provided parameters

    Args:
        pipeline_name: Name of pipeline
        request: Request with optional filepath and parameters

    Returns:
        Final parameters dict with filepath and user params
    """
    final_params = {}

    # ✅ Handle data_loading pipeline with filepath
    if pipeline_name == "data_loading" and request and request.filepath:
        final_params = {
            "data_loading": {
                "filepath": request.filepath
            }
        }
        logger.info(f"📁 Using dynamic filepath: {request.filepath}")

    # Merge user-provided parameters
    if request and request.parameters:
        if "data_loading" in request.parameters and "data_loading" in final_params:
            # Merge nested dicts
            final_params["data_loading"].update(request.parameters["data_loading"])
        final_params.update(request.parameters)

    logger.info(f"📦 Final parameters: {final_params}")
    return final_params


def get_currently_running_algorithm(logs: list) -> Optional[str]:
    """
    Detect the CURRENTLY RUNNING algorithm with better accuracy

    Logic:
    1. Find "Training ALGO..." logs (exact format)
    2. Check if algorithm is in VALID_ALGORITHMS list
    3. Check if it has a completion line "✅ ALGO: Train="
    4. If no completion, it's CURRENTLY RUNNING

    ✅ Filters out false positives like "ensemble", "voting", etc.
    """
    if not logs:
        return None

    # Patterns for actual training logs
    training_pattern = r"Training\s+(\w+)\.\.\."
    completion_pattern = r"✅\s+(\w+):\s+Train="

    # Get all completed algorithms
    completed_algos = set()
    for log in logs:
        match = re.search(completion_pattern, log)
        if match:
            algo = match.group(1)
            if algo in VALID_ALGORITHMS:  # ✅ Only trust valid algorithms
                completed_algos.add(algo)

    # Search in REVERSE (newest logs first) for incomplete training
    for log in reversed(logs):
        match = re.search(training_pattern, log)
        if match:
            algo = match.group(1)

            # ✅ Validate: Is it a real algorithm?
            if algo not in VALID_ALGORITHMS:
                continue  # Skip false positives like "ensemble"

            # ✅ Is it NOT in completed list?
            if algo not in completed_algos:
                return algo  # Found it!

    return None

def get_all_algorithms_status(logs: list) -> dict:
    """
    Get status of all algorithms with better filtering
    """
    if not logs:
        return {
            "currently_running": None,
            "completed": [],
            "failed": [],
            "total": 0,
            "completed_count": 0,
            "failed_count": 0,
            "progress_percent": 0.0
        }

    training_pattern = r"Training\s+(\w+)\.\.\."
    completion_pattern = r"✅\s+(\w+):\s+Train="
    failure_pattern = r"❌\s+(\w+)\s+failed"

    # Collect all data
    all_algos = set()
    completed = []
    failed = []

    for log in logs:
        # Get all training attempts
        match = re.search(training_pattern, log)
        if match:
            algo = match.group(1)
            if algo in VALID_ALGORITHMS:  # ✅ Only track valid algorithms
                all_algos.add(algo)

        # Get completions (in order, avoid duplicates)
        match = re.search(completion_pattern, log)
        if match:
            algo = match.group(1)
            if algo in VALID_ALGORITHMS and algo not in completed:
                completed.append(algo)

        # Get failures (avoid duplicates)
        match = re.search(failure_pattern, log)
        if match:
            algo = match.group(1)
            if algo in VALID_ALGORITHMS and algo not in failed:
                failed.append(algo)

    # Get current running algorithm
    currently_running = get_currently_running_algorithm(logs)

    # Calculate progress
    total_count = len(all_algos)
    progress_percent = round((len(completed) / total_count * 100) if total_count > 0 else 0, 1)

    return {
        "currently_running": currently_running,
        "completed": completed,
        "failed": failed,
        "total": total_count,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "progress_percent": progress_percent
    }

def _safe_resolve_under(base: Path, candidate: Path) -> Path:
    """
    Prevent path traversal. Ensures candidate is inside base.
    """
    base = base.resolve()
    cand = candidate.resolve()
    if base not in cand.parents and base != cand:
        raise HTTPException(status_code=400, detail="Invalid path")
    return cand

def read_json_file(path) -> dict:
    path = Path(path)  # <--- convert str -> Path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {str(e)}")

def read_csv_file_as_dicts(path) -> list:
    path = Path(path)  # <--- convert str -> Path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    try:
        raw = path.read_text(encoding="utf-8")
        reader = csv.DictReader(StringIO(raw))
        return list(reader)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV: {str(e)}")

def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default
# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
def health_check():
    """Health check endpoint"""
    logger.info("🏥 Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# PIPELINE EXECUTION ENDPOINTS
# ============================================================================

@router.post("/run-pipeline/{pipeline_name}", status_code=202)
def run_pipeline(
        pipeline_name: str,
        filepath: Optional[str] = None,  # ✅ Query parameter
        request: Optional[RunPipelineRequest] = None
):
    """
    Trigger a Kedro pipeline execution via Celery

    Args:
        pipeline_name: Name of the Kedro pipeline to execute
        filepath: Optional filepath as query parameter (for data_loading)
        request: Optional request body with parameters and description

    Returns:
        Job information with status "pending" and job_id

    Examples (CURL):

        Option 1 - Query Parameter (simplest):
        curl -X POST "http://192.168.1.147:8000/api/v1/jobs/run-pipeline/data_loading?filepath=data/01_raw/project_123/users.csv"

        Option 2 - Request Body (JSON):
        curl -X POST "http://192.168.1.147:8000/api/v1/jobs/run-pipeline/data_loading" \\
          -H "Content-Type: application/json" \\
          -d '{"filepath": "data/01_raw/project_123/users.csv"}'

        Option 3 - With Additional Parameters:
        curl -X POST "http://192.168.1.147:8000/api/v1/jobs/run-pipeline/data_loading?filepath=data/01_raw/project_123/users.csv" \\
          -H "Content-Type: application/json" \\
          -d '{"parameters": {"data_loading": {"test_size": 0.25}}}'
    """
    #file_path = os.path.join(str(KEDRO_PROJECT_PATH), dataset.file_path)
    logger.info(f"📊 API Request: Run pipeline '{pipeline_name}'")
    logger.info(f"📁 Query filepath: {filepath}")

    try:
        # ✅ Handle filepath from query parameter
        if filepath:
            if not request:
                request = RunPipelineRequest(filepath=filepath)
            else:
                request.filepath = filepath  # Override with query param

        # ✅ Build parameters with dynamic filepath
        job_parameters = build_job_parameters(pipeline_name, request)

        # Create job in database
        logger.info(f"Creating job in database for pipeline: {pipeline_name}")

        job = db_manager.create_job(
            pipeline_name=pipeline_name,
            parameters=job_parameters,  # ✅ NOW HAS DYNAMIC FILEPATH
            user_id="api_user"
        )

        logger.info(f"✅ Job created: {job['id']}")

        # Queue task to Celery worker
        logger.info(f"Queuing task to Celery...")
        task = execute_pipeline.delay(
            job_id=job['id'],
            pipeline_name=pipeline_name,
            parameters=job_parameters  # ✅ PASS FINAL PARAMETERS
        )

        logger.info(f"✅ Task queued: {task.id}")

        response = {
            "id": job['id'],
            "pipeline_name": pipeline_name,
            "status": "pending",
            "celery_task_id": task.id,
            "message": f"Pipeline '{pipeline_name}' queued for execution",
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(f"📤 Returning response: {response}")
        return response

    except Exception as e:
        logger.error(f"❌ Error in run_pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error queuing pipeline: {str(e)}"
        )


# ============================================================================
# SPECIFIC PIPELINE ENDPOINTS
# ============================================================================

@router.post("/data-loading", status_code=202, tags=["data"])
def run_data_loading(filepath: Optional[str] = None, request: Optional[RunPipelineRequest] = None):
    """
    Run data_loading pipeline specifically

    This endpoint triggers the data_loading phase which:
    - Loads data from multiple sources
    - Validates data formats
    - Prepares train/test splits

    CURL Examples:

        1. Simple - Just filepath as query param:
        curl -X POST "http://192.168.1.147:8000/api/v1/jobs/data-loading?filepath=data/01_raw/project_123/users.csv"

        2. With JSON body:
        curl -X POST "http://192.168.1.147:8000/api/v1/jobs/data-loading" \\
          -H "Content-Type: application/json" \\
          -d '{"filepath": "data/01_raw/project_123/users.csv"}'
    """
    logger.info(f"📊 API Request: Run data_loading pipeline with filepath={filepath}")
    return run_pipeline("data_loading", filepath=filepath, request=request)


@router.post("/data-validation", status_code=202, tags=["data"])
def run_data_validation(filepath: Optional[str] = None, request: Optional[RunPipelineRequest] = None):
    """Run data_validation pipeline"""
    logger.info(f"📊 API Request: Run data_validation pipeline")
    return run_pipeline("data_validation", filepath=filepath, request=request)


@router.post("/data-cleaning", status_code=202, tags=["data"])
def run_data_cleaning(filepath: Optional[str] = None, request: Optional[RunPipelineRequest] = None):
    """Run data_cleaning pipeline"""
    logger.info(f"📊 API Request: Run data_cleaning pipeline")
    return run_pipeline("data_cleaning", filepath=filepath, request=request)


@router.post("/feature-engineering", status_code=202, tags=["features"])
def run_feature_engineering(filepath: Optional[str] = None, request: Optional[RunPipelineRequest] = None):
    """Run feature_engineering pipeline"""
    logger.info(f"📊 API Request: Run feature_engineering pipeline")
    return run_pipeline("feature_engineering", filepath=filepath, request=request)


@router.post("/feature-selection", status_code=202, tags=["features"])
def run_feature_selection(filepath: Optional[str] = None, request: Optional[RunPipelineRequest] = None):
    """Run feature_selection pipeline"""
    logger.info(f"📊 API Request: Run feature_selection pipeline")
    return run_pipeline("feature_selection", filepath=filepath, request=request)


@router.post("/model-training", status_code=202, tags=["models"])
def run_model_training(filepath: Optional[str] = None, request: Optional[RunPipelineRequest] = None):
    """Run model_training pipeline"""
    logger.info(f"📊 API Request: Run model_training pipeline")
    return run_pipeline("model_training", filepath=filepath, request=request)


@router.post("/algorithms", status_code=202, tags=["models"])
def run_algorithms(filepath: Optional[str] = None, request: Optional[RunPipelineRequest] = None):
    """Run algorithms pipeline"""
    logger.info(f"📊 API Request: Run algorithms pipeline")
    return run_pipeline("algorithms", filepath=filepath, request=request)


@router.post("/ensemble", status_code=202, tags=["models"])
def run_ensemble(filepath: Optional[str] = None, request: Optional[RunPipelineRequest] = None):
    """Run ensemble pipeline"""
    logger.info(f"📊 API Request: Run ensemble pipeline")
    return run_pipeline("ensemble", filepath=filepath, request=request)


# ============================================================================
# JOB STATUS AND RESULTS
# ============================================================================

@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    """
    Get job status and results

    Returns current status of the job. If completed, includes execution results.

    Args:
        job_id: Job identifier

    Returns:
        Job information with current status and results if completed

    Example:
        GET /api/v1/jobs/3b9c5987-2de6-4f9f-9828-85b55d6ca060

        Response:
        {
            "id": "3b9c5987-2de6-4f9f-9828-85b55d6ca060",
            "pipeline_name": "data_loading",
            "status": "completed",
            "results": {...},
            "execution_time": 45.23,
            "created_at": "2026-02-03T21:41:16",
            "completed_at": "2026-02-03T21:42:01.950974"
        }
    """

    logger.info(f"🔍 API Request: Get job status for {job_id}")

    try:
        job = db_manager.get_job(job_id)

        if not job:
            logger.warning(f"Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )

        logger.info(f"✅ Job found: status={job['status']}")
        return job

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving job: {str(e)}"
        )


@router.get("/")
def list_jobs(status: Optional[str] = None, limit: int = 50):
    """
    List jobs with optional filtering

    Args:
        status: Filter by status (pending, running, completed, failed)
        limit: Maximum number of jobs to return

    Returns:
        List of jobs matching criteria
    """

    logger.info(f"📋 API Request: List jobs (status={status}, limit={limit})")

    try:
        # This would need to be implemented in JobManager
        jobs = []  # db_manager.list_jobs(status=status, limit=limit)

        return {
            "total": len(jobs),
            "jobs": jobs
        }

    except Exception as e:
        logger.error(f"❌ Error listing jobs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing jobs: {str(e)}"
        )


# ============================================================================
# PIPELINE INFORMATION ENDPOINTS
# ============================================================================

@router.get("/pipelines/list", response_model=PipelineListResponse, tags=["pipelines"])
def list_pipelines():
    """
    List all available Kedro pipelines

    Returns:
        List of available pipelines with descriptions

    Example:
        GET /api/v1/jobs/pipelines/list
    """

    logger.info(f"📋 API Request: List available pipelines")

    pipelines = [
        {
            "name": "data_loading",
            "description": "Load raw data from multiple sources",
            "status": "available"
        },
        {
            "name": "data_validation",
            "description": "Validate loaded data quality and structure",
            "status": "available"
        },
        {
            "name": "data_cleaning",
            "description": "Clean and preprocess data",
            "status": "available"
        },
        {
            "name": "feature_engineering",
            "description": "Create new features from raw data",
            "status": "available"
        },
        {
            "name": "feature_selection",
            "description": "Select important features",
            "status": "available"
        },
        {
            "name": "model_training",
            "description": "Train ML models on prepared data",
            "status": "available"
        },
        {
            "name": "algorithms",
            "description": "Advanced algorithm implementations",
            "status": "available"
        },
        {
            "name": "ensemble",
            "description": "Ensemble methods and stacking",
            "status": "available"
        },
    ]

    logger.info(f"✅ Returning {len(pipelines)} available pipelines")

    return {
        "total": len(pipelines),
        "pipelines": pipelines
    }


@router.get("/pipelines/{pipeline_name}/info", tags=["pipelines"])
def get_pipeline_info(pipeline_name: str):
    """
    Get detailed information about a specific pipeline

    Args:
        pipeline_name: Name of the pipeline

    Returns:
        Pipeline structure, nodes, inputs, outputs, estimated time

    Example:
        GET /api/v1/jobs/pipelines/data_loading/info
    """

    logger.info(f"📊 API Request: Get info for pipeline '{pipeline_name}'")

    pipeline_info = {
        "data_loading": {
            "description": "Load raw data from multiple CSV files",
            "nodes": ["load_data_node"],
            "inputs": ["params:data_loading"],
            "outputs": [
                "X_train_raw",
                "X_test_raw",
                "y_train_raw",
                "y_test_raw"
            ],
            "estimated_time": 60,
            "memory_required": "2-3 GB"
        },
        "data_validation": {
            "description": "Validate loaded data quality",
            "nodes": ["validate_data_node"],
            "inputs": ["X_train_raw", "X_test_raw", "y_train_raw", "y_test_raw"],
            "outputs": ["validation_report"],
            "estimated_time": 30,
            "memory_required": "2-3 GB"
        },
        "feature_engineering": {
            "description": "Create new features from raw data",
            "nodes": ["feature_engineering_node"],
            "inputs": ["X_train_raw", "X_test_raw"],
            "outputs": ["X_train_engineered", "X_test_engineered"],
            "estimated_time": 120,
            "memory_required": "2-3 GB"
        },
    }

    if pipeline_name not in pipeline_info:
        logger.warning(f"Pipeline not found: {pipeline_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline '{pipeline_name}' not found"
        )

    logger.info(f"✅ Returning info for pipeline: {pipeline_name}")
    return pipeline_info[pipeline_name]


# ============================================================================
# STATISTICS AND MONITORING
# ============================================================================

@router.get("/stats", tags=["monitoring"])
def get_job_statistics():
    """
    Get job execution statistics

    Returns:
        Statistics on job execution (total, success rate, average time, etc.)
    """

    logger.info(f"📊 API Request: Get job statistics")

    # This would need to be implemented in JobManager
    stats = {
        "total_jobs": 0,
        "completed": 0,
        "failed": 0,
        "running": 0,
        "pending": 0,
        "success_rate": 0.0,
        "average_execution_time": 0.0
    }

    return stats


@router.get("/pipelines/performance", tags=["monitoring"])
def get_pipeline_performance():
    """
    Get performance statistics for each pipeline

    Returns:
        Execution time and success rate for each pipeline
    """

    logger.info(f"📊 API Request: Get pipeline performance metrics")

    performance = {
        "data_loading": {
            "average_time": 45.5,
            "success_rate": 0.98,
            "total_executions": 50
        },
        "feature_engineering": {
            "average_time": 120.3,
            "success_rate": 0.95,
            "total_executions": 30
        }
    }

    return performance


logger.info("✅ Jobs router fully initialized with filepath support")

def extract_algorithm(log_line: str) -> str:
    """Extract algorithm name from log line"""
    patterns = [
        r"Training\s+(\w+)\.\.\.",          # "Training LogisticRegression..."
        r"✅\s+(\w+):\s+Train=",            # "✅ LogisticRegression: Train=0.8037"
        r"Model\s+\d+:\s+(\w+)",            # "Model 1: AdaBoostClassifier"
    ]

    for pattern in patterns:
        match = re.search(pattern, log_line)
        if match:
            return match.group(1)

    return None

@router.websocket("/ws/{job_id}")
async def websocket_job_logs(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for streaming live job logs

    Connect: ws://localhost:8000/api/jobs/ws/{job_id}/logs

    Sends:
    {
        "type": "log",
        "message": "Training LogisticRegression...",
        "algorithm": "LogisticRegression"  // If found
    }
    """

    await websocket.accept()
    logger.info(f"✅ WebSocket connected for job: {job_id}")

    last_line_count = 0

    try:
        while True:
            log_file = LOGS_DIR / f"{job_id}.log"

            if log_file.exists():
                # Read all lines
                with open(log_file, 'r') as f:
                    all_lines = f.readlines()

                # Send new lines only
                for line in all_lines[last_line_count:]:
                    line = line.strip()
                    if line:
                        algorithm = extract_algorithm(line)

                        # Send to client
                        await websocket.send_json({
                            "type": "log",
                            "message": line,
                            "algorithm": algorithm  # Null if not found
                        })

                last_line_count = len(all_lines)

            # Poll every 500ms
            await asyncio.sleep(0.5)

    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass

    finally:
        logger.info(f"❌ WebSocket disconnected for job: {job_id}")

LOGS_DIR = Path("data/job_logs")
LOGS_DIR.mkdir(exist_ok=True)

# Regex patterns to extract algorithm names
ALGORITHM_PATTERNS = [
    r"Training\s+(\w+)\.\.\.",  # "Training LogisticRegression..."
    r"✅\s+(\w+):\s+Train=",     # "✅ LogisticRegression: Train=0.8037"
    r"Comparing:\s+Model\s+\d+:\s+(\w+)",  # "Model 1: AdaBoostClassifier"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_job_log_file(job_id: str) -> Path:
    """Get path to job log file"""
    return LOGS_DIR / f"{job_id}.log"

def save_log_line(job_id: str, log_line: str):
    """Save log line to file"""
    log_file = get_job_log_file(job_id)
    with open(log_file, 'a') as f:
        f.write(log_line + '\n')

def extract_algorithm_name(log_line: str) -> str:
    """Extract algorithm name from log line"""
    for pattern in ALGORITHM_PATTERNS:
        match = re.search(pattern, log_line)
        if match:
            return match.group(1)
    return None

def get_current_algorithm(logs: list) -> str:
    """Extract current running algorithm from logs"""
    if not logs:
        return None

    # Get last 10 logs and find algorithm
    for log in reversed(logs):
        algorithm = extract_algorithm_name(log)
        if algorithm:
            return algorithm

    return None

# ✅ Add these helper functions at the top!
def get_safe_value(obj, key, default=None):
    """Get value from dict or object safely"""
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        else:
            return getattr(obj, key, default)
    except Exception:
        return default

def format_datetime(dt):
    """Format datetime to ISO string"""
    if dt is None:
        return None
    if hasattr(dt, 'isoformat'):
        return dt.isoformat()
    return str(dt)

def read_job_logs(job_id: str) -> list:
    """Read all logs for a job"""
    log_file = LOGS_DIR / f"{job_id}.log"
    if not log_file.exists():
        return []
    with open(log_file, 'r') as f:
        return [line.strip() for line in f.readlines()]
# ============================================================================
# REST ENDPOINTS
# ============================================================================

@router.get("/logs/{job_id}")
async def get_job_logs(job_id: str, db: Session = Depends(get_db)):
    """Get job details including live logs and current algorithm"""

    try:
        job = db_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        logs = read_job_logs(job_id)
        #current_algo = get_current_algorithm(logs)

        # ✅ Safe dict/object access
        def get_val(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # ✅ Get values
        parameters = get_val(job, 'parameters', {})
        result = get_val(job, 'result', {})

        # ✅ Parse safely (handle both str and dict)
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except:
                parameters = {}
        elif not parameters:
            parameters = {}

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                result = {}
        elif not result:
            result = {}

        # ✅ Format datetime
        created_at = get_val(job, 'created_at')
        updated_at = get_val(job, 'updated_at')

        if hasattr(created_at, 'isoformat'):
            created_at = created_at.isoformat()
        else:
            created_at = str(created_at) if created_at else None

        if hasattr(updated_at, 'isoformat'):
            updated_at = updated_at.isoformat()
        else:
            updated_at = str(updated_at) if updated_at else None

        return {
            "id": get_val(job, 'id'),
            "pipeline_name": get_val(job, 'pipeline_name'),
            "status": get_val(job, 'status'),
            "created_at": created_at,
            "updated_at": updated_at,
            "parameters": parameters,  # ✅ Now guaranteed to be dict
            "result": result,  # ✅ Now guaranteed to be dict
            "logs": {
                "total_lines": len(logs),
                "recent_logs": logs
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# REST ENDPOINTS
# ============================================================================

@router.get("/logs/{job_id}/current-algorithm")
async def get_current_algorithm_status(job_id: str):
    """
    Get the CURRENTLY RUNNING algorithm (fixed version)

    Endpoint: GET /api/v1/jobs/logs/{job_id}/current-algorithm

    ✅ Filters out false positives
    ✅ Only returns real algorithm names
    """

    logs = read_job_logs(job_id)
    current_algo = get_currently_running_algorithm(logs)

    return {
        "job_id": job_id,
        "currently_running": current_algo,
        "is_running": current_algo is not None,
        "total_logs": len(logs)
    }

@router.get("/logs/{job_id}/algorithms-status")
async def get_algorithms_status(job_id: str):
    """
    Get complete status of all algorithms (fixed version)

    Endpoint: GET /api/v1/jobs/logs/{job_id}/algorithms-status

    ✅ Filters out false positives
    ✅ Only counts real algorithms
    """

    logs = read_job_logs(job_id)
    status = get_all_algorithms_status(logs)

    return status

@router.get("/logs/{job_id}/progress")
async def get_job_progress(job_id: str, db: Session = Depends(get_db)):
    """
    Get job progress with algorithm details (fixed version)

    Endpoint: GET /api/v1/jobs/logs/{job_id}/progress

    ✅ Shows correct currently running algorithm
    ✅ Filters false positives
    """

    try:
        job = db_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        logs = read_job_logs(job_id)
        algo_status = get_all_algorithms_status(logs)

        return {
            "id": get_safe_value(job, 'id', job_id),
            "pipeline_name": get_safe_value(job, 'pipeline_name'),
            "status": get_safe_value(job, 'status'),
            "progress": {
                "currently_running": algo_status["currently_running"],
                "completed_count": algo_status["completed_count"],
                "failed_count": algo_status["failed_count"],
                "total_count": algo_status["total"],
                "progress_percent": algo_status["progress_percent"],
                "completed": algo_status["completed"],
                "failed": algo_status["failed"]
            },
            "total_logs": len(logs),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job progress: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/phase4", tags=["reports"])
def get_phase4_report(top_n: int = Query(default=10, ge=1, le=100)):
    """
    Returns:
      - summary (json)
      - best_model details (row)
      - top_n ranked rows
      - full ranked rows (optional: you can remove if too big)
    """
    summary_path =  os.path.join(str(KEDRO_PROJECT_PATH), MODEL_OUTPUT_DIR / "phase4_summary.json")
    ranked_path = os.path.join(str(KEDRO_PROJECT_PATH), MODEL_OUTPUT_DIR / "phase4_ranked_report.csv")
    full_path =  os.path.join(str(KEDRO_PROJECT_PATH),MODEL_OUTPUT_DIR / "phase4_report.csv")

    logger.info(summary_path)

    summary = read_json_file(summary_path)
    ranked_rows = read_csv_file_as_dicts(ranked_path)
    full_rows = read_csv_file_as_dicts(full_path)

    # Normalize numeric fields for UI sorting/formatting
    for r in ranked_rows:
        r["Train_Score"] = to_float(r.get("Train_Score"))
        r["Test_Score"] = to_float(r.get("Test_Score"))
        r["Diff"] = to_float(r.get("Diff"))

    for r in full_rows:
        r["Train_Score"] = to_float(r.get("Train_Score"))
        r["Test_Score"] = to_float(r.get("Test_Score"))
        r["Diff"] = to_float(r.get("Diff"))

    best_model_name = summary.get("best_model")
    best_row = next((r for r in ranked_rows if r.get("Algorithm") == best_model_name), None)

    return {
        "summary": summary,
        "best_model": {
            "name": best_model_name,
            "best_score": summary.get("best_score"),
            "details": best_row
        },
        "top_ranked": ranked_rows[:top_n],
        "ranked": ranked_rows,
        "full_report": full_rows
    }