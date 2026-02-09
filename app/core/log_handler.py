"""
Custom log handler for streaming job logs to file
"""

import logging
from pathlib import Path

LOGS_DIR = Path("data/job_logs")
LOGS_DIR.mkdir(exist_ok=True)

class JobLogHandler(logging.Handler):
    """Save logs to file for real-time streaming"""

    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id
        self.log_file = LOGS_DIR / f"{job_id}.log"
        # Create empty log file
        self.log_file.touch()

    def emit(self, record):
        """Save log record to file"""
        try:
            msg = self.format(record)
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
        except Exception:
            self.handleError(record)

def setup_job_logger(job_id: str, logger_instance):
    """Setup logger for a specific job"""
    handler = JobLogHandler(job_id)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_instance.addHandler(handler)
    return handler