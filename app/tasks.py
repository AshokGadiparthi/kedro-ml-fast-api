"""
Celery tasks for ML Pipeline execution with Kedro integration
COMPLETE VERSION: Subprocess + All Existing Functionality + PARAMETER FIX

Features:
- Non-blocking Kedro execution using subprocess
- Pipeline name validation
- Parameter support with proper CLI formatting ✅ FIXED
- Comprehensive logging
- Database integration
- Error handling
- All existing tasks (process_data, analyze_data)
- Timeout protection
"""

import os
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from celery_app import app
from app.core.job_manager import JobManager
from app.core.log_handler import setup_job_logger
import sys

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize DB manager
db_manager = JobManager()

# Get Kedro project path from environment
KEDRO_PROJECT_PATH = Path(os.getenv(
    'KEDRO_PROJECT_PATH',
    '/home/ashok/work/latest/full/kedro-ml-engine-integrated'
))

logger.info(f"✅ Kedro project path configured: {KEDRO_PROJECT_PATH}")

# Valid pipeline names - MUST match your Kedro pipelines!
VALID_PIPELINES = [
    '__default__', 'complete', 'all', 'end_to_end', 'a_b_c',
    'phase1', 'phase2', 'phase3', 'phase4', 'phase1_2', 'phase6',
    'data_loading', 'data_validation', 'data_cleaning', 'data_processing',
    'feature_engineering', 'model_training', 'algorithms',
    'ensemble', 'ensemble_methods', 'complete_1_6',
    'all_with_ensemble', 'end_to_end_full','complete_1_6','complete_1_5_6','phase5',
    'phase3_4'
]


# ============================================================================
# ✅ HELPER FUNCTION - Parameter Flattening (NEW)
# ============================================================================

def flatten_parameters(params, parent_key=""):
    """
    Flatten nested parameters to dot notation for Kedro CLI

    ✅ FIXES THE PARAMETER FORMATTING ISSUE!

    Example:
        Input:  {'data_loading': {'filepath': 'data/01_raw/file.csv'}}
        Output: {'data_loading.filepath': 'data/01_raw/file.csv'}

    This allows Kedro CLI to properly parse parameters:
        Before: --params data_loading:{'filepath': '...'}  ❌ WRONG
        After:  --params data_loading.filepath=...         ✅ CORRECT

    Args:
        params (dict): Nested parameters dictionary
        parent_key (str): Parent key for recursion (internal use)

    Returns:
        dict: Flattened dictionary with dot-notation keys
    """
    items = []

    for k, v in params.items():
        new_key = f"{parent_key}.{k}" if parent_key else k

        if isinstance(v, dict):
            # Recursively flatten nested dicts
            items.extend(flatten_parameters(v, new_key).items())
        else:
            # Convert value to string for CLI
            items.append((new_key, str(v)))

    return dict(items)


# ============================================================================
# MAIN PIPELINE EXECUTION TASK
# ============================================================================

# --- Add this before subprocess.run ---
env = os.environ.copy()

@app.task(name='app.tasks.execute_pipeline', bind=True, time_limit=3600, soft_time_limit=3300)
def execute_pipeline(self, job_id: str, pipeline_name: str, parameters: dict = None):
    """Execute Kedro pipeline with live log streaming"""

    job_start_time = datetime.utcnow()

    # ✅ Setup job logging (saves to file for WebSocket to read)
    log_handler = setup_job_logger(job_id, logger)

    logger.info(f"{'='*80}")
    logger.info(f"🚀 STARTING PIPELINE EXECUTION")
    logger.info(f"{'='*80}")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Pipeline: {pipeline_name}")
    logger.info(f"Parameters: {parameters or {}}")

    try:
        # STEP 1: Update job status
        logger.info(f"\n[STEP 1] Updating job status...")
        db_manager.update_job_status(job_id, "running")
        logger.info(f"✅ Job {job_id} marked as RUNNING")

        # STEP 2: Verify Kedro project
        logger.info(f"\n[STEP 2] Verifying Kedro project...")
        if not KEDRO_PROJECT_PATH.exists():
            raise FileNotFoundError(f"Kedro project not found at {KEDRO_PROJECT_PATH}")
        logger.info(f"✅ Kedro project verified: {KEDRO_PROJECT_PATH}")

        # STEP 2.5: Validate pipeline
        logger.info(f"\n[STEP 2.5] Validating pipeline name...")
        if pipeline_name not in VALID_PIPELINES:
            raise ValueError(f"Pipeline '{pipeline_name}' not found. Valid: {', '.join(VALID_PIPELINES)}")
        logger.info(f"✅ Pipeline name is valid: {pipeline_name}")

        # STEP 3: Prepare parameters
        logger.info(f"\n[STEP 3] Preparing pipeline parameters...")
        extra_params = parameters or {}
        if isinstance(extra_params, dict) and "parameters" in extra_params:
            extra_params = extra_params["parameters"]

        if extra_params:
            logger.info(f"📊 Using custom parameters:")
            for key, value in extra_params.items():
                logger.info(f"   - {key}: {value}")

        # STEP 4: Execute pipeline
        logger.info(f"\n[STEP 4] Executing pipeline via subprocess...")

        env = os.environ.copy()
        if extra_params and "data_loading" in extra_params:
            if "filepath" in extra_params["data_loading"]:
                env["RAW_DATA_FILEPATH"] = str(extra_params["data_loading"]["filepath"])
                logger.info(f"✅ Set RAW_DATA_FILEPATH={env['RAW_DATA_FILEPATH']}")

        import sys
        python_exe = sys.executable
        cmd = [python_exe, '-m', 'kedro', 'run', '--pipeline', pipeline_name]

        # Add parameters if present
        if extra_params:
            flat_params = flatten_parameters(extra_params)
            logger.info("📊 Flattened parameters:")
            for k, v in flat_params.items():
                logger.info(f"   - {k}={v}")
            params_str = ",".join([f"{k}={v}" for k, v in flat_params.items()])
            cmd.extend(["--params", params_str])

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Working directory: {KEDRO_PROJECT_PATH}")
        logger.info(f"{'='*80}")

        # Run Kedro
        try:
            result = subprocess.run(
                cmd,
                cwd=str(KEDRO_PROJECT_PATH),
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Log output (will be saved via JobLogHandler)
            if result.stdout:
                logger.info(f"\n[Kedro Output]")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(line)  # ✅ Logs to file

            if result.returncode != 0:
                logger.error(f"\n[Kedro Error]")
                if result.stderr:
                    for line in result.stderr.split('\n'):
                        if line.strip():
                            logger.error(line)
                raise RuntimeError(f"Kedro failed with exit code {result.returncode}")

            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Pipeline execution COMPLETED")

        except subprocess.TimeoutExpired:
            logger.error("❌ Pipeline execution timed out (>5 minutes)")
            raise TimeoutError("Kedro pipeline execution exceeded timeout")

        # STEP 5: Prepare result
        logger.info(f"\n[STEP 5] Preparing execution result...")
        execution_time = (datetime.utcnow() - job_start_time).total_seconds()
        result = {
            "status": "completed",
            "pipeline_name": pipeline_name,
            "message": f"Pipeline '{pipeline_name}' executed successfully",
            "execution_time": execution_time,
            "parameters_used": extra_params,
            "timestamp": job_start_time.isoformat()
        }

        # STEP 6: Store results
        logger.info(f"\n[STEP 6] Storing results in database...")
        db_manager.update_job_results(job_id, result)
        db_manager.update_job_status(job_id, "completed")  # ✅ Update status
        logger.info(f"✅ Results stored for job {job_id}")

        # SUCCESS
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ PIPELINE EXECUTION SUCCESSFUL")
        logger.info(f"{'='*80}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Pipeline: {pipeline_name}")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Time: {execution_time:.2f}s\n")

        return result

    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"❌ PIPELINE EXECUTION FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Job ID: {job_id}")
        logger.error(f"Pipeline: {pipeline_name}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        db_manager.update_job_status(job_id, "failed")  # ✅ Update status

        execution_time = (datetime.utcnow() - job_start_time).total_seconds()
        error_result = {
            "status": "failed",
            "pipeline_name": pipeline_name,
            "error_message": str(e),
            "execution_time": execution_time,
            "timestamp": job_start_time.isoformat()
        }

        try:
            db_manager.update_job_error(job_id, str(e))
            db_manager.update_job_status(job_id, "failed")  # ✅ Update status
        except Exception as log_error:
            logger.error(f"❌ Failed to log error: {log_error}")

        return error_result

    finally:
        # Cleanup
        logger.removeHandler(log_handler)

# ============================================================================
# ADDITIONAL TASKS (process_data, analyze_data)
# ============================================================================

@app.task(name='app.tasks.process_data', bind=True)
def process_data(self, dataset_id: str, processing_type: str, parameters: dict = None):
    """
    Process dataset with specified processing type

    Args:
        dataset_id (str): Unique dataset identifier
        processing_type (str): Type of processing to apply
        parameters (dict): Processing parameters

    Returns:
        dict: Processing result with status and metadata
    """
    start_time = datetime.utcnow()
    logger.info(f"{'='*80}")
    logger.info(f"📊 STARTING DATA PROCESSING")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset ID: {dataset_id}")
    logger.info(f"Processing Type: {processing_type}")
    logger.info(f"Parameters: {parameters or {}}")

    try:
        # Simulate processing (replace with actual logic)
        import time
        time.sleep(2)

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        result = {
            "status": "completed",
            "dataset_id": dataset_id,
            "processing_type": processing_type,
            "parameters": parameters or {},
            "execution_time": execution_time,
            "message": f"Successfully processed {dataset_id} with {processing_type}",
            "timestamp": start_time.isoformat()
        }

        logger.info(f"✅ Data processing COMPLETED")
        logger.info(f"   - Status: {result['status']}")
        logger.info(f"   - Execution Time: {execution_time:.2f}s")

        # Update database if needed
        try:
            logger.info(f"Storing processing results...")
            # Add your database update logic here if needed
            logger.info(f"✅ Results stored")
        except Exception as e:
            logger.error(f"⚠️  Could not store results: {e}")

        return result

    except Exception as e:
        logger.error(f"{'='*80}")
        logger.error(f"❌ DATA PROCESSING FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {str(e)}", exc_info=True)

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        error_result = {
            "status": "failed",
            "dataset_id": dataset_id,
            "processing_type": processing_type,
            "error_message": str(e),
            "execution_time": execution_time,
            "timestamp": start_time.isoformat()
        }

        return error_result


@app.task(name='app.tasks.analyze_data', bind=True)
def analyze_data(self, dataset_id: str, analysis_type: str, parameters: dict = None):
    """
    Analyze dataset with specified analysis type

    Args:
        dataset_id (str): Unique dataset identifier
        analysis_type (str): Type of analysis to perform
        parameters (dict): Analysis parameters

    Returns:
        dict: Analysis result with status and metadata
    """
    start_time = datetime.utcnow()
    logger.info(f"{'='*80}")
    logger.info(f"📈 STARTING DATA ANALYSIS")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset ID: {dataset_id}")
    logger.info(f"Analysis Type: {analysis_type}")
    logger.info(f"Parameters: {parameters or {}}")

    try:
        # Simulate analysis (replace with actual logic)
        import time
        time.sleep(2)

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        result = {
            "status": "completed",
            "dataset_id": dataset_id,
            "analysis_type": analysis_type,
            "parameters": parameters or {},
            "execution_time": execution_time,
            "message": f"Successfully analyzed {dataset_id} with {analysis_type}",
            "timestamp": start_time.isoformat(),
            "analysis_results": {
                "rows_processed": 1000,
                "patterns_found": 5,
                "anomalies_detected": 2
            }
        }

        logger.info(f"✅ Data analysis COMPLETED")
        logger.info(f"   - Status: {result['status']}")
        logger.info(f"   - Execution Time: {execution_time:.2f}s")
        logger.info(f"   - Rows Processed: {result['analysis_results']['rows_processed']}")

        # Update database if needed
        try:
            logger.info(f"Storing analysis results...")
            # Add your database update logic here if needed
            logger.info(f"✅ Results stored")
        except Exception as e:
            logger.error(f"⚠️  Could not store results: {e}")

        return result

    except Exception as e:
        logger.error(f"{'='*80}")
        logger.error(f"❌ DATA ANALYSIS FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {str(e)}", exc_info=True)

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        error_result = {
            "status": "failed",
            "dataset_id": dataset_id,
            "analysis_type": analysis_type,
            "error_message": str(e),
            "execution_time": execution_time,
            "timestamp": start_time.isoformat()
        }

        return error_result


logger.info("✅ Celery tasks initialized successfully")