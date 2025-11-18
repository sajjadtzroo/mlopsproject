"""
Complete ML Pipeline DAG
This DAG orchestrates the entire ML pipeline including:
- Data ingestion
- Data processing
- Multi-model training
- Clustering analysis
- MLflow tracking
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.clustering_models import ClusteringModels
from src.components.data_ingestion import DataIngestion
from src.components.data_processing import DataProcessing
from src.components.multi_model_training import MultiModelTraining


# Task Functions
def ingest_data(**context):
    """Task to ingest data"""
    ingestion = DataIngestion()
    data_path = ingestion.initiate_data_ingestion()
    context["ti"].xcom_push(key="raw_data_path", value=data_path)
    return data_path


def process_data(**context):
    """Task to process data"""
    raw_data_path = context["ti"].xcom_pull(
        task_ids="ingest_data", key="raw_data_path"
    )
    processing = DataProcessing()
    X_train, X_test, y_train, y_test = processing.initiate_data_processing(
        raw_data_path
    )

    # Save processed data to disk for next tasks
    import os

    import pandas as pd

    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

    print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")


def train_classification_models(**context):
    """Task to train all classification models"""
    import pandas as pd

    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    trainer = MultiModelTraining()
    models, results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    comparison_df = trainer.print_comparison()

    # Store best model info in XCom
    best_model = comparison_df.iloc[0]["model_name"]
    best_accuracy = comparison_df.iloc[0]["test_accuracy"]

    context["ti"].xcom_push(key="best_model", value=best_model)
    context["ti"].xcom_push(key="best_accuracy", value=float(best_accuracy))

    print(f"\n✓ Trained {len(models)} classification models")
    print(f"✓ Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")


def train_clustering_models(**context):
    """Task to train clustering models"""
    import pandas as pd

    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    clustering = ClusteringModels()
    cluster_models, cluster_results = clustering.train_all_clustering(X_train, y_train)
    clustering.print_comparison()

    print(f"\n✓ Trained {len(cluster_models)} clustering models")


def generate_pipeline_report(**context):
    """Task to generate final pipeline report"""
    best_model = context["ti"].xcom_pull(
        task_ids="train_classification_models", key="best_model"
    )
    best_accuracy = context["ti"].xcom_pull(
        task_ids="train_classification_models", key="best_accuracy"
    )

    report = f"""
    ===================================================================
    ML PIPELINE EXECUTION REPORT
    ===================================================================
    Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Pipeline Status: SUCCESS

    Classification Results:
    - Best Model: {best_model}
    - Best Accuracy: {best_accuracy:.4f}

    Clustering: Completed

    Next Steps:
    - View MLflow UI for detailed metrics
    - Run: python view_results.py
    - Export: python export_results.py
    ===================================================================
    """

    print(report)


# Default arguments for the DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}

# Define the DAG
with DAG(
    dag_id="complete_ml_pipeline",
    default_args=default_args,
    description="Complete ML pipeline with multi-model training and clustering",
    schedule_interval="@weekly",  # Run weekly
    catchup=False,
    tags=["ml-pipeline", "titanic", "mlops", "classification", "clustering"],
) as dag:
    # Task 1: Ingest data
    ingest_data_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
        provide_context=True,
        doc_md="""
        ## Data Ingestion
        Loads the Titanic dataset from seaborn and saves it to the raw data path.
        """,
    )

    # Task 2: Process data
    process_data_task = PythonOperator(
        task_id="process_data",
        python_callable=process_data,
        provide_context=True,
        doc_md="""
        ## Data Processing
        Performs feature engineering, handles missing values, and splits data.
        """,
    )

    # Task 3: Train classification models
    train_classification_task = PythonOperator(
        task_id="train_classification_models",
        python_callable=train_classification_models,
        provide_context=True,
        doc_md="""
        ## Classification Models
        Trains 8 classification models:
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - SVM
        - KNN
        - Naive Bayes
        """,
    )

    # Task 4: Train clustering models
    train_clustering_task = PythonOperator(
        task_id="train_clustering_models",
        python_callable=train_clustering_models,
        provide_context=True,
        doc_md="""
        ## Clustering Models
        Trains clustering models:
        - K-Means
        - Hierarchical Clustering
        """,
    )

    # Task 5: Generate report
    generate_report_task = PythonOperator(
        task_id="generate_pipeline_report",
        python_callable=generate_pipeline_report,
        provide_context=True,
        doc_md="""
        ## Pipeline Report
        Generates a summary report of the pipeline execution.
        """,
    )

    # Set task dependencies
    ingest_data_task >> process_data_task >> train_classification_task
    train_classification_task >> train_clustering_task >> generate_report_task
