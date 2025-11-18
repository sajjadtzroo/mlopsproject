"""
Main entry point for the MLOps project
Run this script to execute the complete multi-model training pipeline
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline.complete_ml_pipeline import CompletePipeline


def main():
    """Run the complete ML pipeline with multiple models"""

    print("\n" + "=" * 70)
    print("TITANIC SURVIVAL PREDICTION - MULTI-MODEL PIPELINE")
    print("=" * 70)
    print("\nThis pipeline will:")
    print("  1. Ingest Titanic dataset from seaborn")
    print("  2. Process and engineer features")
    print("  3. Train 8 classification models:")
    print("     - Logistic Regression")
    print("     - Decision Tree")
    print("     - Random Forest")
    print("     - Gradient Boosting")
    print("     - XGBoost")
    print("     - SVM")
    print("     - K-Nearest Neighbors")
    print("     - Naive Bayes")
    print("  4. Apply clustering algorithms:")
    print("     - K-Means")
    print("     - Hierarchical (Agglomerative)")
    print("  5. Track all experiments with MLflow")
    print("\n" + "=" * 70 + "\n")

    # Run pipeline
    pipeline = CompletePipeline()
    models, results = pipeline.run_pipeline(include_clustering=True)

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("\n1. View results in terminal:")
    print("   python view_results.py")
    print("\n2. Export results to files:")
    print("   python export_results.py")
    print("\n3. View MLflow UI (check PORTS tab → Port 5000)")
    print("\n4. Format code:")
    print("   black . && isort .")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
