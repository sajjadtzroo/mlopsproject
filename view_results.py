"""
View MLflow Results without UI
This script reads MLflow tracking data and displays it in the terminal
"""

import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def view_mlflow_results():
    """Display MLflow experiment results in terminal"""

    # Set tracking URI
    tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    print("\n" + "=" * 70)
    print("MLFLOW EXPERIMENT RESULTS")
    print("=" * 70)

    # Get client
    client = MlflowClient()

    # Get experiment
    experiment_name = "titanic_random_forest"
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"\nNo experiment found with name: {experiment_name}")
        print("Run the pipeline first: python main.py")
        return

    print(f"\nExperiment: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}")

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10,
    )

    if not runs:
        print("\nNo runs found. Run the pipeline first: python main.py")
        return

    print(f"\nTotal Runs: {len(runs)}")
    print("\n" + "=" * 70)

    # Display each run
    for idx, run in enumerate(runs, 1):
        print(f"\nRun #{idx}")
        print("-" * 70)
        print(f"Run ID: {run.info.run_id}")
        print(f"Status: {run.info.status}")
        print(f"Start Time: {pd.to_datetime(run.info.start_time, unit='ms')}")

        # Parameters
        print("\nParameters:")
        for key, value in sorted(run.data.params.items()):
            print(f"  {key:20s}: {value}")

        # Metrics
        print("\nMetrics:")
        for key, value in sorted(run.data.metrics.items()):
            print(f"  {key:20s}: {value:.4f}")

        # Artifacts
        print("\nArtifacts:")
        artifacts = client.list_artifacts(run.info.run_id)
        for artifact in artifacts:
            print(f"  - {artifact.path}")

        print("-" * 70)

    # Display comparison table
    print("\n" + "=" * 70)
    print("RUNS COMPARISON TABLE")
    print("=" * 70)

    comparison_data = []
    for run in runs:
        row = {
            "run_id": run.info.run_id[:8],
            "test_accuracy": run.data.metrics.get("test_accuracy", 0),
            "test_f1_score": run.data.metrics.get("test_f1_score", 0),
            "test_precision": run.data.metrics.get("test_precision", 0),
            "test_recall": run.data.metrics.get("test_recall", 0),
            "n_estimators": run.data.params.get("n_estimators", "N/A"),
            "max_depth": run.data.params.get("max_depth", "N/A"),
        }
        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))

    # Best run
    print("\n" + "=" * 70)
    print("BEST RUN (by test accuracy)")
    print("=" * 70)
    best_run = max(runs, key=lambda r: r.data.metrics.get("test_accuracy", 0))
    print(f"\nRun ID: {best_run.info.run_id}")
    print(f"Test Accuracy: {best_run.data.metrics.get('test_accuracy', 0):.4f}")
    print(f"Test F1 Score: {best_run.data.metrics.get('test_f1_score', 0):.4f}")
    print(
        f"Parameters: n_estimators={best_run.data.params.get('n_estimators')}, max_depth={best_run.data.params.get('max_depth')}"
    )

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    view_mlflow_results()
