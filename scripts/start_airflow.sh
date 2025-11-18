#!/bin/bash
# Airflow Start Script
# Starts the Airflow webserver and scheduler

set -e

# Set environment variables
export AIRFLOW_HOME=/workspaces/mlopsproject/airflow

# Load Airflow configuration
if [ -f "$AIRFLOW_HOME/config/airflow.env" ]; then
    set -a
    source "$AIRFLOW_HOME/config/airflow.env"
    set +a
fi

echo "========================================="
echo "Starting Airflow Services"
echo "========================================="
echo ""
echo "AIRFLOW_HOME: $AIRFLOW_HOME"
echo ""

# Check if Airflow is initialized
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "Error: Airflow not initialized!"
    echo "Run: ./scripts/init_airflow.sh first"
    exit 1
fi

echo "Starting Airflow Scheduler in background..."
nohup airflow scheduler > $AIRFLOW_HOME/logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "Scheduler PID: $SCHEDULER_PID"

echo "Starting Airflow Webserver in background..."
nohup airflow webserver --port 8080 > $AIRFLOW_HOME/logs/webserver.log 2>&1 &
WEBSERVER_PID=$!
echo "Webserver PID: $WEBSERVER_PID"

echo ""
echo "========================================="
echo "Airflow Started Successfully!"
echo "========================================="
echo ""
echo "Access Airflow UI:"
echo "  URL: http://localhost:8080"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "To stop Airflow:"
echo "  ./scripts/stop_airflow.sh"
echo ""
echo "View logs:"
echo "  Scheduler: tail -f $AIRFLOW_HOME/logs/scheduler.log"
echo "  Webserver: tail -f $AIRFLOW_HOME/logs/webserver.log"
echo ""
