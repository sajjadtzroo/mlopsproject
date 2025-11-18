"""
Data Ingestion DAG
This DAG handles periodic data ingestion from seaborn dataset
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.data_ingestion import DataIngestion


def run_data_ingestion():
    """
    Task to run data ingestion
    """
    ingestion = DataIngestion()
    data_path = ingestion.initiate_data_ingestion()
    print(f"Data ingestion completed. Data saved to: {data_path}")
    return data_path


def validate_data(**context):
    """
    Task to validate ingested data
    """
    import pandas as pd

    data_path = context["ti"].xcom_pull(task_ids="ingest_data")
    df = pd.read_csv(data_path)

    # Basic validation
    assert df.shape[0] > 0, "Dataset is empty"
    assert "survived" in df.columns, "Target column 'survived' not found"

    print(f"Data validation passed!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


# Default arguments for the DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}

# Define the DAG
with DAG(
    dag_id="data_ingestion_pipeline",
    default_args=default_args,
    description="Ingest Titanic dataset from seaborn",
    schedule_interval="@daily",  # Run daily
    catchup=False,
    tags=["data-ingestion", "titanic", "mlops"],
) as dag:
    # Task 1: Ingest data
    ingest_data_task = PythonOperator(
        task_id="ingest_data",
        python_callable=run_data_ingestion,
        doc_md="""
        ## Ingest Data
        Loads the Titanic dataset from seaborn and saves it to the raw data path.
        """,
    )

    # Task 2: Validate data
    validate_data_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        provide_context=True,
        doc_md="""
        ## Validate Data
        Validates the ingested data to ensure it meets basic requirements.
        """,
    )

    # Set task dependencies
    ingest_data_task >> validate_data_task
