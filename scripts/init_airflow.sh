#!/bin/bash
# Airflow Initialization Script
# This script sets up Airflow for the first time

set -e

echo "========================================="
echo "Airflow Initialization"
echo "========================================="

# Set environment variables
export AIRFLOW_HOME=/workspaces/mlopsproject/airflow

# Load Airflow configuration
if [ -f "$AIRFLOW_HOME/config/airflow.env" ]; then
    set -a
    source "$AIRFLOW_HOME/config/airflow.env"
    set +a
fi

echo "1. Installing Airflow dependencies..."
pip install -q apache-airflow==2.10.4 apache-airflow-providers-docker==3.9.1

echo "2. Initializing Airflow database..."
airflow db init

echo "3. Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "User already exists, skipping..."

echo ""
echo "========================================="
echo "Airflow Initialization Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Start Airflow: ./scripts/start_airflow.sh"
echo "  2. Open Airflow UI: http://localhost:8080"
echo "  3. Login with: admin / admin"
echo ""
