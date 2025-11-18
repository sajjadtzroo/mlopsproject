"""
Export MLflow Results to Files
Creates readable text and CSV files with all experiment results
"""

import json
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def export_mlflow_results(output_dir="results"):
    """Export MLflow results to files"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set tracking URI
    tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    print(f"\nExporting MLflow results to: {output_dir}/")
    print("=" * 70)

    # Get client
    client = MlflowClient()

    # Get experiment
    experiment_name = "titanic_random_forest"
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"No experiment found: {experiment_name}")
        return

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
    )

    if not runs:
        print("No runs found!")
        return

    # Export summary
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MLFLOW EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Experiment: {experiment.name}\n")
        f.write(f"Experiment ID: {experiment.experiment_id}\n")
        f.write(f"Total Runs: {len(runs)}\n\n")

        for idx, run in enumerate(runs, 1):
            f.write("=" * 70 + "\n")
            f.write(f"RUN #{idx}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Run ID: {run.info.run_id}\n")
            f.write(f"Status: {run.info.status}\n")
            f.write(f"Start Time: {pd.to_datetime(run.info.start_time, unit='ms')}\n\n")

            f.write("PARAMETERS:\n")
            for key, value in sorted(run.data.params.items()):
                f.write(f"  {key:20s}: {value}\n")

            f.write("\nMETRICS:\n")
            for key, value in sorted(run.data.metrics.items()):
                f.write(f"  {key:20s}: {value:.4f}\n")

            f.write("\nARTIFACTS:\n")
            artifacts = client.list_artifacts(run.info.run_id)
            for artifact in artifacts:
                f.write(f"  - {artifact.path}\n")
            f.write("\n")

    print(f"✓ Created: {output_dir}/summary.txt")

    # Export metrics comparison CSV
    comparison_data = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            "status": run.info.status,
            **run.data.params,
            **run.data.metrics,
        }
        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(f"{output_dir}/runs_comparison.csv", index=False)
    print(f"✓ Created: {output_dir}/runs_comparison.csv")

    # Export best run details
    best_run = max(runs, key=lambda r: r.data.metrics.get("test_accuracy", 0))
    best_run_data = {
        "run_id": best_run.info.run_id,
        "test_accuracy": best_run.data.metrics.get("test_accuracy", 0),
        "test_f1_score": best_run.data.metrics.get("test_f1_score", 0),
        "test_precision": best_run.data.metrics.get("test_precision", 0),
        "test_recall": best_run.data.metrics.get("test_recall", 0),
        "train_accuracy": best_run.data.metrics.get("train_accuracy", 0),
        "parameters": dict(best_run.data.params),
    }

    with open(f"{output_dir}/best_run.json", "w") as f:
        json.dump(best_run_data, f, indent=2)
    print(f"✓ Created: {output_dir}/best_run.json")

    with open(f"{output_dir}/best_run.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("BEST RUN (by test accuracy)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Run ID: {best_run.info.run_id}\n\n")
        f.write("METRICS:\n")
        f.write(f"  Test Accuracy:  {best_run_data['test_accuracy']:.4f}\n")
        f.write(f"  Test F1 Score:  {best_run_data['test_f1_score']:.4f}\n")
        f.write(f"  Test Precision: {best_run_data['test_precision']:.4f}\n")
        f.write(f"  Test Recall:    {best_run_data['test_recall']:.4f}\n")
        f.write(f"  Train Accuracy: {best_run_data['train_accuracy']:.4f}\n\n")
        f.write("PARAMETERS:\n")
        for key, value in best_run.data.params.items():
            f.write(f"  {key}: {value}\n")
    print(f"✓ Created: {output_dir}/best_run.txt")

    # Try to copy feature importance if it exists
    try:
        # Download feature importance artifact
        artifact_path = client.download_artifacts(
            best_run.info.run_id, "feature_importance.csv"
        )
        import shutil

        shutil.copy(artifact_path, f"{output_dir}/feature_importance.csv")
        print(f"✓ Created: {output_dir}/feature_importance.csv")
    except Exception as e:
        print(f"  (Feature importance not available)")

    print("\n" + "=" * 70)
    print(f"All results exported to: {output_dir}/")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  - summary.txt              (Complete text summary)")
    print(f"  - runs_comparison.csv      (All runs comparison)")
    print(f"  - best_run.txt            (Best model details)")
    print(f"  - best_run.json           (Best model JSON)")
    print(f"  - feature_importance.csv  (Feature importance)")
    print()


if __name__ == "__main__":
    export_mlflow_results()
