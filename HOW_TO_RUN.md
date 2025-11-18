# How to Run the MLOps Project

This guide covers all the ways to run your Titanic ML Pipeline with Airflow orchestration.

## Quick Start

### Option 1: Run Pipeline Directly (Fastest)

```bash
# Run the complete ML pipeline
python main.py
```

This will:
- ✓ Ingest Titanic dataset
- ✓ Process and engineer features
- ✓ Train 8 classification models
- ✓ Train 2 clustering models
- ✓ Track everything in MLflow

**Expected runtime:** ~30-60 seconds

---

### Option 2: Run with Airflow Orchestration

#### Step 1: Start Airflow (if not running)

```bash
# Check if Airflow is running
ps aux | grep airflow | grep -v grep

# If not running, start it
./scripts/start_airflow.sh
```

#### Step 2: Access Airflow UI

Open your browser to: **http://localhost:8080**

Login credentials:
- Username: `admin`
- Password: `admin`

#### Step 3: Run a DAG

**Via UI:**
1. Click on the DAG name (e.g., `complete_ml_pipeline`)
2. Toggle the switch to "On" (unpause)
3. Click the "Play" ▶ button → "Trigger DAG"
4. Monitor execution in the Graph view

**Via CLI:**
```bash
# Set Airflow home
export AIRFLOW_HOME=/workspaces/mlopsproject/airflow

# Trigger the complete ML pipeline
airflow dags trigger complete_ml_pipeline

# Trigger data ingestion only
airflow dags trigger data_ingestion_pipeline

# Check DAG status
airflow dags list-runs -d complete_ml_pipeline
```

---

## Available DAGs

### 1. Data Ingestion Pipeline
- **DAG ID:** `data_ingestion_pipeline`
- **Schedule:** Daily at midnight
- **Duration:** ~5 seconds
- **What it does:**
  - Loads Titanic dataset from seaborn
  - Validates data quality
  - Saves to `data/raw/titanic.csv`

```bash
airflow dags trigger data_ingestion_pipeline
```

### 2. Complete ML Pipeline
- **DAG ID:** `complete_ml_pipeline`
- **Schedule:** Weekly (every Sunday at midnight)
- **Duration:** ~60 seconds
- **What it does:**
  - Data ingestion
  - Feature engineering
  - Trains 8 classification models
  - Trains 2 clustering models
  - Generates performance report
  - Logs everything to MLflow

```bash
airflow dags trigger complete_ml_pipeline
```

---

## Viewing Results

### 1. Terminal Output

```bash
# View detailed results in terminal
python view_results.py
```

### 2. Export to Files

```bash
# Export results to CSV and JSON
python export_results.py
```

This creates:
- `results/model_comparison.csv` - Model performance metrics
- `results/model_comparison.json` - Detailed results with metadata

### 3. MLflow UI

```bash
# Start MLflow tracking server
mlflow ui --backend-store-uri ./mlruns

# Access at http://localhost:5000
```

In MLflow UI you can:
- Compare all model runs
- View hyperparameters
- Download model artifacts
- Visualize metrics over time

---

## Running Individual Components

You can run each component separately for testing:

### Data Ingestion Only
```bash
python src/components/data_ingestion.py
```
Output: `data/raw/titanic.csv`

### Data Processing Only
```bash
# Requires raw data to exist first
python src/components/data_processing.py
```
Output: `data/processed/titanic_processed.csv`

### Model Training Only
```bash
# Requires processed data to exist first
python src/components/multi_model_training.py
```

### Clustering Only
```bash
# Requires processed data to exist first
python src/components/clustering_models.py
```

---

## Airflow Management

### Start Airflow
```bash
./scripts/start_airflow.sh
```

### Stop Airflow
```bash
./scripts/stop_airflow.sh
```

### View Logs
```bash
# Scheduler logs
tail -f airflow/logs/scheduler.log

# Webserver logs
tail -f airflow/logs/webserver.log

# Specific task logs (after running)
# Check in Airflow UI → DAG → Task → Logs
```

### Useful Airflow Commands
```bash
export AIRFLOW_HOME=/workspaces/mlopsproject/airflow

# List all DAGs
airflow dags list

# List DAG runs
airflow dags list-runs -d complete_ml_pipeline

# Unpause/pause a DAG
airflow dags unpause complete_ml_pipeline
airflow dags pause complete_ml_pipeline

# Test a specific task
airflow tasks test complete_ml_pipeline ingest_data 2024-01-01

# Clear task state (to re-run)
airflow tasks clear complete_ml_pipeline

# View DAG details
airflow dags show complete_ml_pipeline
```

---

## Scheduled Execution

DAGs run automatically based on their schedule:

- **Data Ingestion:** Every day at midnight
- **ML Pipeline:** Every Sunday at midnight

To change schedules, edit the DAG files:

```python
# In airflow/dags/data_ingestion_dag.py
schedule_interval="@daily"     # Run daily
schedule_interval="@hourly"    # Run hourly
schedule_interval="0 9 * * *"  # Run at 9 AM daily
schedule_interval=None         # Manual trigger only
```

Common cron expressions:
- `@hourly` - Every hour
- `@daily` - Every day at midnight
- `@weekly` - Every Sunday at midnight
- `@monthly` - First day of month at midnight
- `0 9 * * 1-5` - 9 AM on weekdays
- `*/15 * * * *` - Every 15 minutes

---

## Monitoring Pipeline Execution

### 1. Airflow UI (Real-time)
- Graph View: Visual representation of task flow
- Grid View: Historical runs overview
- Task Logs: Detailed execution logs
- Gantt Chart: Task duration visualization

### 2. MLflow UI (Results)
- Experiment comparison
- Parameter tuning analysis
- Model registry
- Artifact storage

### 3. Command Line
```bash
# Watch DAG status
watch -n 5 'airflow dags list-runs -d complete_ml_pipeline | head -10'

# Follow scheduler logs
tail -f airflow/logs/scheduler.log
```

---

## Troubleshooting

### Pipeline fails to run
```bash
# Check if dependencies are installed
pip list | grep -E "(airflow|mlflow|scikit-learn)"

# Reinstall if needed
pip install -r requirements.txt
```

### Airflow UI not accessible
```bash
# Check if webserver is running
ps aux | grep "airflow webserver"

# Check port 8080
curl http://localhost:8080/health

# Restart Airflow
./scripts/stop_airflow.sh
./scripts/start_airflow.sh
```

### DAGs not showing up
```bash
# Check DAG folder
ls -la airflow/dags/

# Test DAG syntax
python airflow/dags/data_ingestion_dag.py
python airflow/dags/ml_pipeline_dag.py

# Check scheduler is scanning DAGs
tail -f airflow/logs/scheduler.log | grep "DAG"
```

### Permission errors
```bash
# Fix ownership
sudo chown -R $USER:$USER airflow/

# Fix permissions
chmod -R 755 airflow/dags/
chmod +x scripts/*.sh
```

---

## Performance Considerations

### Running on limited resources:
- Sequential execution (default with SequentialExecutor)
- One task at a time
- Lower memory usage

### For better performance:
- Use Docker setup with LocalExecutor + PostgreSQL
- Parallel task execution
- Better for production

---

## Example Workflow

Here's a typical workflow:

```bash
# 1. Start services
./scripts/start_airflow.sh

# 2. Run the pipeline manually first time
python main.py

# 3. View results
python view_results.py

# 4. Export results
python export_results.py

# 5. Check MLflow
mlflow ui

# 6. Enable Airflow automation
export AIRFLOW_HOME=/workspaces/mlopsproject/airflow
airflow dags unpause complete_ml_pipeline

# 7. Access Airflow UI to monitor
# Open http://localhost:8080

# 8. When done, stop services
./scripts/stop_airflow.sh
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Run pipeline | `python main.py` |
| View results | `python view_results.py` |
| Export results | `python export_results.py` |
| Start Airflow | `./scripts/start_airflow.sh` |
| Stop Airflow | `./scripts/stop_airflow.sh` |
| Start MLflow | `mlflow ui` |
| Trigger DAG | `airflow dags trigger complete_ml_pipeline` |
| List DAGs | `airflow dags list` |
| Airflow UI | http://localhost:8080 |
| MLflow UI | http://localhost:5000 |

---

## Additional Resources

- Airflow setup guide: [AIRFLOW_SETUP.md](AIRFLOW_SETUP.md)
- Project README: [README.md](README.md)
- Airflow docs: https://airflow.apache.org/docs/
- MLflow docs: https://mlflow.org/docs/latest/
