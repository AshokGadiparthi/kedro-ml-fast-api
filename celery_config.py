"""Celery Configuration Class"""
import os
from dotenv import load_dotenv

load_dotenv()

class CeleryConfig:
    broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
    task_serializer = 'json'
    result_serializer = 'json'
    accept_content = ['json']
    timezone = 'UTC'
    enable_utc = True
    task_track_started = True
    task_time_limit = 3600
    task_soft_time_limit = 3300
    result_expires = 3600
    worker_max_tasks_per_child = 100
    worker_prefetch_multiplier = 1
    task_default_queue = 'default'
    task_default_routing_key = 'task.default'