# Airflow Setup and Usage Guide

This guide explains how to set up and use Apache Airflow for orchestrating the MLOps pipeline.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup Options](#setup-options)
  - [Option 1: Local Setup](#option-1-local-setup)
  - [Option 2: Docker Setup](#option-2-docker-setup)
- [Available DAGs](#available-dags)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Overview

This project uses Apache Airflow to orchestrate two main workflows:

1. **Data Ingestion Pipeline**: Automatically ingests and validates Titanic dataset
2. **Complete ML Pipeline**: Runs the entire ML workflow including data processing, model training, and clustering

## Prerequisites

- Python 3.10+
- pip package manager
- (For Docker) Docker and Docker Compose

## Setup Options

### Option 1: Local Setup

#### Step 1: Install Dependencies

```bash
# Install all project dependencies including Airflow
pip install -r requirements.txt
```

#### Step 2: Initialize Airflow

```bash
# Run the initialization script
./scripts/init_airflow.sh
```

This script will:
- Set up the Airflow database
- Create the admin user (username: `admin`, password: `admin`)
- Configure the Airflow home directory

#### Step 3: Start Airflow Services

```bash
# Start the webserver and scheduler
./scripts/start_airflow.sh
```

The Airflow UI will be available at: http://localhost:8080

#### Step 4: Access Airflow UI

1. Open browser: http://localhost:8080
2. Login with:
   - Username: `admin`
   - Password: `admin`

#### Stop Airflow

```bash
./scripts/stop_airflow.sh
```

### Option 2: Docker Setup

Docker setup provides isolated, reproducible environments and includes both Airflow and MLflow services.

#### Step 1: Configure Environment

```bash
# Copy the docker environment file
cp .env.docker .env
```

#### Step 2: Start All Services

```bash
# Start Airflow, PostgreSQL, and MLflow
docker-compose up -d
```

This will start:
- **Airflow Webserver** (port 8080)
- **Airflow Scheduler**
- **PostgreSQL** (Airflow metadata database)
- **MLflow Server** (port 5000)

#### Step 3: Access Services

- Airflow UI: http://localhost:8080 (admin/admin)
- MLflow UI: http://localhost:5000

#### View Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-webserver
```

#### Stop All Services

```bash
docker-compose down
```

#### Stop and Remove Volumes

```bash
docker-compose down -v
```

## Available DAGs

### 1. Data Ingestion Pipeline

**DAG ID**: `data_ingestion_pipeline`

**Schedule**: Daily (`@daily`)

**Description**: Ingests Titanic dataset from seaborn and validates the data.

**Tasks**:
- `ingest_data`: Load dataset from seaborn
- `validate_data`: Validate the ingested data

**Manual Trigger**:
```bash
airflow dags trigger data_ingestion_pipeline
```

### 2. Complete ML Pipeline

**DAG ID**: `complete_ml_pipeline`

**Schedule**: Weekly (`@weekly`)

**Description**: Runs the entire ML pipeline with multi-model training and clustering.

**Tasks**:
1. `ingest_data`: Load raw data
2. `process_data`: Feature engineering and preprocessing
3. `train_classification_models`: Train 8 classification models
4. `train_clustering_models`: Train clustering models
5. `generate_pipeline_report`: Generate execution report

**Classification Models**:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- SVM
- K-Nearest Neighbors
- Naive Bayes

**Clustering Models**:
- K-Means
- Hierarchical Clustering

**Manual Trigger**:
```bash
airflow dags trigger complete_ml_pipeline
```

## Usage

### Running DAGs from UI

1. Navigate to http://localhost:8080
2. Login with admin credentials
3. Find the DAG you want to run
4. Toggle the DAG to "On" (unpause it)
5. Click the "Play" button to trigger manually
6. Monitor execution in the Graph or Grid view

### Running DAGs from CLI

```bash
# List all DAGs
airflow dags list

# Trigger a DAG
airflow dags trigger data_ingestion_pipeline
airflow dags trigger complete_ml_pipeline

# Check DAG status
airflow dags state data_ingestion_pipeline

# View task logs
airflow tasks logs complete_ml_pipeline ingest_data 2024-01-01
```

### Monitoring Pipeline Execution

#### View Task Logs

1. Click on a DAG in the UI
2. Click on a task in the Graph view
3. Click "Log" to view detailed execution logs

#### View MLflow Results

After running the ML pipeline:

1. Access MLflow UI at http://localhost:5000
2. View all experiment runs
3. Compare model metrics
4. Access model artifacts

Or use the terminal:

```bash
# View results in terminal
python view_results.py

# Export results to files
python export_results.py
```

### Configuring DAG Schedules

Edit the DAG files in `airflow/dags/` to change schedules:

```python
# In data_ingestion_dag.py or ml_pipeline_dag.py
schedule_interval="@daily"   # Run daily
schedule_interval="@weekly"  # Run weekly
schedule_interval="0 0 * * *"  # Cron expression (midnight daily)
schedule_interval=None       # Manual trigger only
```

Common schedule intervals:
- `@once`: Run once
- `@hourly`: Every hour
- `@daily`: Every day at midnight
- `@weekly`: Every Sunday at midnight
- `@monthly`: First day of month at midnight
- `None`: Manual trigger only
- Cron expressions: e.g., `"0 9 * * 1-5"` (9 AM weekdays)

## Project Structure

```
mlopsproject/
├── airflow/
│   ├── dags/
│   │   ├── data_ingestion_dag.py      # Data ingestion DAG
│   │   └── ml_pipeline_dag.py          # Complete ML pipeline DAG
│   ├── logs/                           # Airflow logs (gitignored)
│   ├── plugins/                        # Custom Airflow plugins
│   ├── config/
│   │   └── airflow.env                 # Airflow configuration
│   └── .gitignore
├── scripts/
│   ├── init_airflow.sh                 # Initialize Airflow
│   ├── start_airflow.sh                # Start Airflow services
│   └── stop_airflow.sh                 # Stop Airflow services
├── docker-compose.yml                   # Docker orchestration
├── .env.docker                          # Docker environment variables
└── AIRFLOW_SETUP.md                     # This file
```

## Troubleshooting

### Port Already in Use

If port 8080 is already in use:

```bash
# Find and kill the process
lsof -ti:8080 | xargs kill -9

# Or change the port in airflow/config/airflow.env
AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8081
```

### Database Locked Error

```bash
# Stop all Airflow processes
./scripts/stop_airflow.sh

# Remove the lock
rm -f airflow/airflow.db-shm airflow/airflow.db-wal

# Restart
./scripts/start_airflow.sh
```

### DAGs Not Appearing

1. Check DAG syntax:
```bash
python airflow/dags/data_ingestion_dag.py
```

2. Check DAGs folder path in config:
```bash
echo $AIRFLOW__CORE__DAGS_FOLDER
```

3. Refresh the UI (it scans every 30 seconds by default)

### Import Errors in DAGs

Make sure the project root is in PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/workspaces/mlopsproject"
```

### Task Failures

1. Check task logs in the Airflow UI
2. Verify data paths and permissions
3. Ensure all dependencies are installed
4. Check MLflow tracking server is running

### Reset Airflow Database

```bash
./scripts/stop_airflow.sh
rm -f airflow/airflow.db*
./scripts/init_airflow.sh
./scripts/start_airflow.sh
```

## Best Practices

1. **Development**:
   - Test DAGs locally before deploying
   - Use `catchup=False` to prevent backfilling old runs
   - Set appropriate retry policies

2. **Production**:
   - Use Docker setup for consistency
   - Configure email alerts for failures
   - Monitor resource usage
   - Regular backups of Airflow metadata database

3. **DAG Development**:
   - Keep tasks idempotent (safe to re-run)
   - Use XCom for small data passing between tasks
   - Store large datasets in files/databases, not XCom
   - Add proper logging and error handling

## Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Support

For issues or questions:
1. Check the logs in `airflow/logs/`
2. Review the troubleshooting section above
3. Consult the official Airflow documentation
