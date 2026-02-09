"""Celery Application Configuration"""
import os
import sys
import logging
from celery import Celery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Create Celery app
app = Celery('ml_platform')

# Configure Celery
app.conf.update(
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1'),
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    result_expires=3600,
    worker_max_tasks_per_child=100,
)

logger.info(f"✅ Celery configured: {os.getenv('CELERY_BROKER_URL')}")

# Autodiscover tasks - handle import error gracefully
try:
    app.autodiscover_tasks(['app'], force=True)
    logger.info("✅ Tasks autodiscovered from 'app'")
except ModuleNotFoundError as e:
    logger.warning(f"⚠️  Could not autodiscover tasks from 'app': {e}")
    logger.info("⚠️  Make sure app/tasks.py exists and has tasks defined")
    # Try alternative discovery
    try:
        # If in project root, try discovering from current directory
        app.autodiscover_tasks(['app'], force=True)
    except:
        logger.error("❌ Failed to autodiscover tasks. Make sure you're in the right directory.")
        logger.error("   Expected: ~/work/latest/full/kedro-engine-dynamic")
        logger.error("   Run: cd ~/work/latest/full/kedro-engine-dynamic")
        # Don't fail completely - still allow worker to start
        pass

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')