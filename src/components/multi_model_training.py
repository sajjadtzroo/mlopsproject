"""
Multi-Model Training Component
Trains and compares multiple ML models with MLflow tracking
"""

import logging
import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultiModelConfig:
    """Configuration for multi-model training"""

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "titanic_multi_model")
    random_state: int = int(os.getenv("RANDOM_STATE", 42))


class MultiModelTraining:
    """Train and compare multiple classification models"""

    def __init__(self):
        self.config = MultiModelConfig()
        self._setup_mlflow()
        self.results = []

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        logger.info(f"MLflow experiment: {self.config.experiment_name}")

    def get_models(self):
        """Define all models to train"""
        models = {
            "Logistic Regression": LogisticRegression(
                random_state=self.config.random_state, max_iter=1000
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=self.config.random_state, max_depth=10
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.config.random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=self.config.random_state
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                eval_metric="logloss",
            ),
            "SVM": SVC(random_state=self.config.random_state, probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
        }
        return models

    def train_single_model(self, name, model, X_train, X_test, y_train, y_test):
        """Train a single model with MLflow tracking"""

        logger.info(f"\n{'='*70}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*70}")

        with mlflow.start_run(run_name=name):
            # Log model type
            mlflow.log_param("model_type", name)
            mlflow.log_param("random_state", self.config.random_state)

            # Log model-specific parameters
            if hasattr(model, "get_params"):
                params = model.get_params()
                for key, value in params.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(key, value)

            # Train model
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

            # Log metrics
            metrics = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1_score": test_f1,
            }
            mlflow.log_metrics(metrics)

            # Log model
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            # Store results
            result = {
                "model_name": name,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1_score": test_f1,
            }
            self.results.append(result)

            logger.info(f"✓ {name} - Test Accuracy: {test_acc:.4f}")
            logger.info(
                f"  Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}"
            )

            return model, result

    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all models and return results"""

        models = self.get_models()
        trained_models = {}

        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING {len(models)} MODELS")
        logger.info(f"{'='*70}")

        for name, model in models.items():
            try:
                trained_model, result = self.train_single_model(
                    name, model, X_train, X_test, y_train, y_test
                )
                trained_models[name] = trained_model
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue

        return trained_models, self.results

    def print_comparison(self):
        """Print comparison of all models"""

        if not self.results:
            logger.warning("No results to compare")
            return

        df = pd.DataFrame(self.results)
        df = df.sort_values("test_accuracy", ascending=False)

        logger.info(f"\n{'='*70}")
        logger.info("MODEL COMPARISON RESULTS")
        logger.info(f"{'='*70}\n")
        logger.info(df.to_string(index=False))

        logger.info(f"\n{'='*70}")
        logger.info(f"BEST MODEL: {df.iloc[0]['model_name']}")
        logger.info(f"Test Accuracy: {df.iloc[0]['test_accuracy']:.4f}")
        logger.info(f"{'='*70}\n")

        return df


if __name__ == "__main__":
    from data_processing import DataProcessing

    # Process data
    processing = DataProcessing()
    X_train, X_test, y_train, y_test = processing.initiate_data_processing(
        "data/raw/titanic.csv"
    )

    # Train all models
    trainer = MultiModelTraining()
    models, results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    trainer.print_comparison()
