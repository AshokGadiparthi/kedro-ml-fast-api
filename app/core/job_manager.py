"""
Job Manager - SQLAlchemy Version
Manages pipeline job operations using SQLAlchemy ORM
Database initialization handled by app/models/models.py
"""

from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging
import uuid
import json
from app.models.models import Job
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage job database operations using SQLAlchemy"""

    def __init__(self):
        """Initialize DatabaseManager"""
        logger.info("✅ DatabaseManager initialized (SQLAlchemy mode)")

    # Job operations - using SQLAlchemy
    def create_job(self, pipeline_name: str, user_id: str = None, parameters: dict = None) -> dict:
        """Create a new job"""
        job_id = str(uuid.uuid4())

        try:
            db = SessionLocal()

            job = Job(
                id=job_id,
                pipeline_name=pipeline_name,
                user_id=user_id or "anonymous",
                parameters=json.dumps(parameters) if parameters else "{}",
                status="pending"
            )

            db.add(job)
            db.commit()
            db.refresh(job)

            logger.info(f"✅ Job created: {job_id}")
            return self._job_to_dict(job)

        except Exception as e:
            logger.error(f"❌ Error creating job: {e}")
            raise
        finally:
            db.close()

    def get_job(self, job_id: str) -> dict:
        """Get job details"""
        try:
            db = SessionLocal()
            job = db.query(Job).filter(Job.id == job_id).first()

            if not job:
                logger.warning(f"Job not found: {job_id}")
                return None

            return self._job_to_dict(job)

        except Exception as e:
            logger.error(f"❌ Error getting job: {e}")
            raise
        finally:
            db.close()

    def update_job_status(self, job_id: str, status: str):
        """Update job status"""
        try:
            db = SessionLocal()
            job = db.query(Job).filter(Job.id == job_id).first()

            if job:
                job.status = status
                db.commit()
                logger.info(f"✅ Job status updated: {job_id} -> {status}")
            else:
                logger.warning(f"Job not found for update: {job_id}")

        except Exception as e:
            logger.error(f"❌ Error updating job status: {e}")
            raise
        finally:
            db.close()

    def update_job_results(self, job_id: str, results: dict):
        """Update job results"""
        try:
            db = SessionLocal()
            job = db.query(Job).filter(Job.id == job_id).first()

            if job:
                job.results = json.dumps(results) if results else None
                job.completed_at = datetime.utcnow()
                db.commit()
                logger.info(f"✅ Job results updated: {job_id}")
            else:
                logger.warning(f"Job not found for results update: {job_id}")

        except Exception as e:
            logger.error(f"❌ Error updating job results: {e}")
            raise
        finally:
            db.close()

    def update_job_error(self, job_id: str, error: str):
        """Update job error"""
        try:
            db = SessionLocal()
            job = db.query(Job).filter(Job.id == job_id).first()

            if job:
                job.error_message = error
                job.status = "failed"
                job.completed_at = datetime.utcnow()
                db.commit()
                logger.info(f"✅ Job error recorded: {job_id}")
            else:
                logger.warning(f"Job not found for error update: {job_id}")

        except Exception as e:
            logger.error(f"❌ Error updating job error: {e}")
            raise
        finally:
            db.close()

    def list_jobs(self, limit: int = 50) -> list:
        """List jobs ordered by creation date"""
        try:
            db = SessionLocal()
            jobs = db.query(Job).order_by(desc(Job.created_at)).limit(limit).all()

            result = [self._job_to_dict(job) for job in jobs]
            logger.info(f"✅ Retrieved {len(result)} jobs")
            return result

        except Exception as e:
            logger.error(f"❌ Error listing jobs: {e}")
            raise
        finally:
            db.close()

    def get_jobs_by_status(self, status: str, limit: int = 50) -> list:
        """Get jobs filtered by status"""
        try:
            db = SessionLocal()
            jobs = db.query(Job).filter(Job.status == status).order_by(desc(Job.created_at)).limit(limit).all()

            result = [self._job_to_dict(job) for job in jobs]
            logger.info(f"✅ Retrieved {len(result)} jobs with status: {status}")
            return result

        except Exception as e:
            logger.error(f"❌ Error getting jobs by status: {e}")
            raise
        finally:
            db.close()

    def get_jobs_by_pipeline(self, pipeline_name: str, limit: int = 50) -> list:
        """Get jobs for a specific pipeline"""
        try:
            db = SessionLocal()
            jobs = db.query(Job).filter(Job.pipeline_name == pipeline_name).order_by(desc(Job.created_at)).limit(limit).all()

            result = [self._job_to_dict(job) for job in jobs]
            logger.info(f"✅ Retrieved {len(result)} jobs for pipeline: {pipeline_name}")
            return result

        except Exception as e:
            logger.error(f"❌ Error getting jobs by pipeline: {e}")
            raise
        finally:
            db.close()

    def _job_to_dict(self, job) -> dict:
        """Convert Job model to dictionary"""
        return {
            'id': job.id,
            'pipeline_name': job.pipeline_name,
            'user_id': job.user_id,
            'status': job.status,
            'parameters': json.loads(job.parameters) if job.parameters else {},
            'results': json.loads(job.results) if job.results else None,
            'error_message': job.error_message,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'execution_time': job.execution_time
        }


# Backward compatibility alias
JobManager = DatabaseManager