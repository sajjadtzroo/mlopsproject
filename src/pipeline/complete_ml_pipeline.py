"""
Complete ML Pipeline
Orchestrates data ingestion, processing, and multi-model training
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from components.clustering_models import ClusteringModels
from components.data_ingestion import DataIngestion
from components.data_processing import DataProcessing
from components.multi_model_training import MultiModelTraining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CompletePipeline:
    """
    End-to-end ML pipeline:
    1. Data Ingestion
    2. Data Processing
    3. Multi-Model Training (8 classifiers)
    4. Clustering Analysis (K-Means, Hierarchical)
    """

    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_processing = DataProcessing()
        self.multi_model_trainer = MultiModelTraining()
        self.clustering_trainer = ClusteringModels()

    def run_pipeline(self, include_clustering=True):
        """Execute complete ML pipeline"""

        logger.info("=" * 70)
        logger.info("COMPLETE ML PIPELINE - TITANIC SURVIVAL PREDICTION")
        logger.info("=" * 70)

        try:
            # Step 1: Data Ingestion
            logger.info(f"\n{'='*70}")
            logger.info("STEP 1: DATA INGESTION")
            logger.info("=" * 70)
            raw_data_path = self.data_ingestion.initiate_data_ingestion()
            logger.info(f"✓ Data loaded: {raw_data_path}")

            # Step 2: Data Processing
            logger.info(f"\n{'='*70}")
            logger.info("STEP 2: DATA PROCESSING")
            logger.info("=" * 70)
            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = self.data_processing.initiate_data_processing(raw_data_path)
            logger.info(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")

            # Step 3: Multi-Model Training
            logger.info(f"\n{'='*70}")
            logger.info("STEP 3: CLASSIFICATION MODELS")
            logger.info("=" * 70)
            models, results = self.multi_model_trainer.train_all_models(
                X_train, X_test, y_train, y_test
            )
            comparison_df = self.multi_model_trainer.print_comparison()

            # Step 4: Clustering (Optional)
            if include_clustering:
                logger.info(f"\n{'='*70}")
                logger.info("STEP 4: CLUSTERING ANALYSIS")
                logger.info("=" * 70)
                (
                    cluster_models,
                    cluster_results,
                ) = self.clustering_trainer.train_all_clustering(X_train, y_train)
                self.clustering_trainer.print_comparison()

            # Final Summary
            logger.info(f"\n{'='*70}")
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            logger.info(f"\n✓ {len(models)} classification models trained")
            if include_clustering:
                logger.info(f"✓ {len(cluster_models)} clustering models trained")
            logger.info(
                f"\nBest Classification Model: {comparison_df.iloc[0]['model_name']}"
            )
            logger.info(
                f"Best Test Accuracy: {comparison_df.iloc[0]['test_accuracy']:.4f}"
            )

            logger.info(f"\n{'='*70}")
            logger.info("VIEW RESULTS:")
            logger.info("=" * 70)
            logger.info("Terminal: python view_results.py")
            logger.info("Export:   python export_results.py")
            logger.info("MLflow UI: Check PORTS tab → Port 5000")
            logger.info("=" * 70)

            return models, results

        except Exception as e:
            logger.error(f"\nPipeline failed: {str(e)}")
            raise e


if __name__ == "__main__":
    pipeline = CompletePipeline()
    pipeline.run_pipeline(include_clustering=True)
