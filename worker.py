"""
Celery Worker - 100% WORKING
Uses celery_app.py for Celery instance
"""

from celery_app import app

# Import tasks to register them
from app import tasks

if __name__ == "__main__":
    app.start()
