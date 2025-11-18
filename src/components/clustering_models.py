"""
Clustering Models Component
Unsupervised learning for pattern discovery
"""

import logging
import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for clustering models"""

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "titanic_clustering")
    n_clusters: int = int(os.getenv("N_CLUSTERS", 2))
    random_state: int = int(os.getenv("RANDOM_STATE", 42))


class ClusteringModels:
    """Apply clustering algorithms for pattern discovery"""

    def __init__(self):
        self.config = ClusteringConfig()
        self._setup_mlflow()
        self.results = []

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        logger.info(f"MLflow experiment: {self.config.experiment_name}")

    def train_kmeans(self, X, y=None):
        """Train K-Means clustering"""

        logger.info(f"\n{'='*70}")
        logger.info("K-Means Clustering")
        logger.info(f"{'='*70}")

        with mlflow.start_run(run_name="K-Means"):
            # Train model
            kmeans = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
                n_init=10,
            )
            labels = kmeans.fit_predict(X)

            # Calculate metrics
            silhouette = silhouette_score(X, labels)
            inertia = kmeans.inertia_

            # Log parameters
            mlflow.log_param("model_type", "K-Means")
            mlflow.log_param("n_clusters", self.config.n_clusters)
            mlflow.log_param("random_state", self.config.random_state)

            # Log metrics
            mlflow.log_metric("silhouette_score", silhouette)
            mlflow.log_metric("inertia", inertia)

            # If ground truth available, calculate purity
            if y is not None:
                purity = self._calculate_purity(y, labels)
                mlflow.log_metric("purity", purity)
                logger.info(f"Purity: {purity:.4f}")

            # Log model
            mlflow.sklearn.log_model(kmeans, "model")

            logger.info(f"✓ K-Means - Silhouette Score: {silhouette:.4f}")
            logger.info(f"  Inertia: {inertia:.2f}")

            result = {
                "model_name": "K-Means",
                "n_clusters": self.config.n_clusters,
                "silhouette_score": silhouette,
                "inertia": inertia,
            }
            self.results.append(result)

            return kmeans, labels, result

    def train_hierarchical(self, X, y=None):
        """Train Hierarchical Clustering"""

        logger.info(f"\n{'='*70}")
        logger.info("Hierarchical Clustering (Agglomerative)")
        logger.info(f"{'='*70}")

        with mlflow.start_run(run_name="Hierarchical-Agglomerative"):
            # Train model
            hierarchical = AgglomerativeClustering(
                n_clusters=self.config.n_clusters, linkage="ward"
            )
            labels = hierarchical.fit_predict(X)

            # Calculate metrics
            silhouette = silhouette_score(X, labels)

            # Log parameters
            mlflow.log_param("model_type", "Hierarchical")
            mlflow.log_param("n_clusters", self.config.n_clusters)
            mlflow.log_param("linkage", "ward")

            # Log metrics
            mlflow.log_metric("silhouette_score", silhouette)

            # If ground truth available, calculate purity
            if y is not None:
                purity = self._calculate_purity(y, labels)
                mlflow.log_metric("purity", purity)
                logger.info(f"Purity: {purity:.4f}")

            # Note: Hierarchical clustering doesn't support pickle easily
            # So we skip logging the model

            logger.info(f"✓ Hierarchical - Silhouette Score: {silhouette:.4f}")

            result = {
                "model_name": "Hierarchical",
                "n_clusters": self.config.n_clusters,
                "silhouette_score": silhouette,
            }
            self.results.append(result)

            return hierarchical, labels, result

    def _calculate_purity(self, y_true, y_pred):
        """Calculate clustering purity against ground truth"""
        contingency_matrix = pd.crosstab(y_true, y_pred)
        return contingency_matrix.max(axis=0).sum() / len(y_true)

    def train_all_clustering(self, X, y=None):
        """Train all clustering models"""

        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING CLUSTERING MODELS")
        logger.info(f"{'='*70}")

        models = {}

        # K-Means
        try:
            kmeans, kmeans_labels, _ = self.train_kmeans(X, y)
            models["K-Means"] = (kmeans, kmeans_labels)
        except Exception as e:
            logger.error(f"Error in K-Means: {str(e)}")

        # Hierarchical
        try:
            hierarchical, hier_labels, _ = self.train_hierarchical(X, y)
            models["Hierarchical"] = (hierarchical, hier_labels)
        except Exception as e:
            logger.error(f"Error in Hierarchical: {str(e)}")

        return models, self.results

    def print_comparison(self):
        """Print clustering results comparison"""

        if not self.results:
            logger.warning("No clustering results")
            return

        df = pd.DataFrame(self.results)
        df = df.sort_values("silhouette_score", ascending=False)

        logger.info(f"\n{'='*70}")
        logger.info("CLUSTERING COMPARISON")
        logger.info(f"{'='*70}\n")
        logger.info(df.to_string(index=False))

        logger.info(f"\n{'='*70}")
        logger.info(f"BEST CLUSTERING: {df.iloc[0]['model_name']}")
        logger.info(f"Silhouette Score: {df.iloc[0]['silhouette_score']:.4f}")
        logger.info(f"{'='*70}\n")

        return df


if __name__ == "__main__":
    from data_processing import DataProcessing

    # Process data
    processing = DataProcessing()
    X_train, X_test, y_train, y_test = processing.initiate_data_processing(
        "data/raw/titanic.csv"
    )

    # Train clustering models
    clustering = ClusteringModels()
    models, results = clustering.train_all_clustering(X_train, y_train)
    clustering.print_comparison()
