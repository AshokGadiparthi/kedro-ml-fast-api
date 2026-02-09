# Clean FastAPI for Pipeline Job Management

**Simple, Clean FastAPI** - No Kedro code mixed in!

This is a standalone FastAPI application that:
- ✅ Submits pipeline jobs via REST API
- ✅ Stores job records in SQLite database
- ✅ Uses Celery for background execution
- ✅ Calls external Kedro project for pipeline execution
- ✅ Minimal dependencies, maximum clarity

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FASTAPI (This Project)                   │
├─────────────────────────────────────────────────────────────────┤
│  • HTTP REST endpoints for job management                       │
│  • SQLite database for job records                              │
│  • Task sending to Celery workers                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Uses Celery
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CELERY WORKER (This Project)                 │
├─────────────────────────────────────────────────────────────────┤
│  • Background task execution                                    │
│  • Calls external Kedro project via CLI                         │
│  • Updates FastAPI database with results                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Subprocess call
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXTERNAL KEDRO PROJECT (Separate Repo)             │
├─────────────────────────────────────────────────────────────────┤
│  • Independent Kedro pipeline project                           │
│  • Located at: /home/ashok/work/latest/full/kedro-engine-dynamic│
│  • Called via: kedro run --pipeline <name>                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
.
├── main.py                    # FastAPI application entry point
├── worker.py                  # Celery worker configuration
├── celery_config.py           # Celery settings
├── requirements.txt           # Python dependencies
├── jobs.db                    # SQLite database (created on first run)
│
└── app/                       # Application package
    ├── __init__.py
    ├── tasks.py               # Celery tasks (calls external Kedro)
    │
    ├── api/                   # API endpoints
    │   ├── __init__.py
    │   ├── jobs.py            # Job submission/status endpoints
    │   ├── pipelines.py       # Pipeline info endpoints
    │   └── health.py          # Health check endpoints
    │
    ├── core/                  # Core utilities
    │   ├── __init__.py
    │   └── job_manager.py     # Database job management
    │
    └── schemas/               # Pydantic models
        ├── __init__.py
        └── job_schemas.py     # Job request/response models
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Update Kedro Project Path

Edit `app/tasks.py` and update this line:

```python
KEDRO_PROJECT_PATH = os.getenv(
    'KEDRO_PROJECT_PATH',
    '/home/ashok/work/latest/full/kedro-engine-dynamic'  # ← Change to YOUR path
)
```

Or set environment variable:

```bash
export KEDRO_PROJECT_PATH=/path/to/your/kedro/project
```

### 3. Start Redis (required for Celery)

```bash
redis-server
```

### 4. Start Celery Worker

```bash
celery -A worker worker --loglevel=info
```

### 5. Start FastAPI

In another terminal:

```bash
python main.py
```

---

## 📡 API Usage

### 1. Submit a Job

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_name": "data_loading",
    "parameters": {}
  }'
```

Response:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "pipeline_name": "data_loading",
  "user_id": "anonymous",
  "status": "pending",
  "parameters": {},
  "results": null,
  "error_message": null,
  "created_at": "2026-02-03T17:30:00",
  "started_at": null,
  "completed_at": null,
  "execution_time": null
}
```

### 2. Check Job Status

```bash
curl http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000
```

Status values:
- `pending` - Waiting to be processed
- `running` - Pipeline executing
- `completed` - Finished successfully
- `failed` - Execution failed

### 3. List Recent Jobs

```bash
curl http://localhost:8000/api/v1/jobs?limit=10
```

### 4. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

---

## 🔄 How It Works

### Job Submission Flow

```
1. POST /api/v1/jobs
   ↓
2. FastAPI creates job in database (status: pending)
   ↓
3. FastAPI sends Celery task
   ↓
4. Celery task receives message from Redis
   ↓
5. Celery worker calls external Kedro project:
   $ cd /path/to/kedro && kedro run --pipeline <name>
   ↓
6. Celery updates job in database:
   - status: running → completed/failed
   - results: pipeline outputs
   - error_message: if failed
   ↓
7. Client polls GET /api/v1/jobs/{id} to check status
```

---

## 🔧 Configuration

### Redis Connection

Edit `celery_config.py`:

```python
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/1'
```

### Kedro Project Path

Option 1: Edit `app/tasks.py`

```python
KEDRO_PROJECT_PATH = '/path/to/kedro/project'
```

Option 2: Environment variable

```bash
export KEDRO_PROJECT_PATH=/path/to/kedro/project
python main.py
```

### Database

SQLite database created automatically at project root as `jobs.db`

---

## 📊 Database Schema

```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    user_id TEXT,
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
    parameters TEXT,                 -- JSON
    results TEXT,                    -- JSON
    error_message TEXT,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time REAL
)
```

---

## 🐛 Troubleshooting

### Tasks not being executed

1. Check Redis is running:
   ```bash
   redis-cli ping
   # Should return: PONG
   ```

2. Check Celery worker is running:
   ```bash
   # Should show: celery@<hostname> ready.
   ```

3. Check Kedro project path exists:
   ```bash
   ls /path/to/kedro/project/kedro.yml
   ```

### Kedro execution fails

Check the error message in job results:

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

Common issues:
- Kedro project path is wrong
- Pipeline name doesn't exist
- Kedro not installed in system PATH

### Database errors

Delete old database and restart:

```bash
rm jobs.db
python main.py
```

---

## 📝 Logging

View detailed logs:

- **FastAPI logs**: console output when running `python main.py`
- **Celery logs**: console output when running `celery -A worker worker`
- **Database logs**: stored in `jobs.db`

---

## ✅ Example Workflow

```bash
# 1. Start services
redis-server &
celery -A worker worker --loglevel=info &
python main.py &

# 2. Submit job
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"pipeline_name": "data_loading"}' | jq -r '.id')

echo "Job ID: $JOB_ID"

# 3. Poll status
while true; do
  STATUS=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  
  sleep 2
done

# 4. View results
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq
```

---

## 🎯 Key Features

✅ **Separation of Concerns**
- FastAPI handles HTTP
- Celery handles background jobs
- External Kedro project handles ML pipelines

✅ **Simple & Clean**
- Minimal code
- Clear structure
- Easy to understand and modify

✅ **Reliable**
- SQLite for job persistence
- Redis for message passing
- Celery for distributed execution

✅ **Scalable**
- Add more Celery workers as needed
- Scale Redis independently
- FastAPI can run on multiple servers

---

## 📚 Further Reading

- FastAPI: https://fastapi.tiangolo.com/
- Celery: https://docs.celeryproject.org/
- Redis: https://redis.io/
- Kedro: https://kedro.readthedocs.io/

---

## 💡 Tips

1. **For production**, use a proper database (PostgreSQL) instead of SQLite
2. **Add authentication** to API endpoints as needed
3. **Monitor Celery** with Flower: `pip install flower && celery -A worker flower`
4. **Scale workers** by running multiple `celery` commands on different machines
5. **Use environment variables** for configuration instead of hardcoding

---

## 📞 Support

For issues:
1. Check the logs
2. Verify all services are running
3. Check Kedro project path is correct
4. Verify Redis and Celery are properly configured
