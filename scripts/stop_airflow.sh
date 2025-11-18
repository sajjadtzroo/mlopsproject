#!/bin/bash
# Airflow Stop Script
# Stops the Airflow webserver and scheduler

echo "========================================="
echo "Stopping Airflow Services"
echo "========================================="

# Kill scheduler
echo "Stopping Airflow Scheduler..."
pkill -f "airflow scheduler" || echo "Scheduler not running"

# Kill webserver
echo "Stopping Airflow Webserver..."
pkill -f "airflow webserver" || echo "Webserver not running"

# Kill any remaining gunicorn processes (webserver workers)
pkill -f "gunicorn" || echo "No gunicorn processes found"

echo ""
echo "Airflow services stopped!"
echo ""
