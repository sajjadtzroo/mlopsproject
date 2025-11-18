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
from sklearn.model_selection import GridSearchCV, cross_val_score
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
    use_grid_search: bool = os.getenv("USE_GRID_SEARCH", "True").lower() == "true"
    cv_folds: int = int(os.getenv("CV_FOLDS", 5))


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
                random_state=self.config.random_state
            ),
            "Random Forest": RandomForestClassifier(
                random_state=self.config.random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=self.config.random_state
            ),
            "XGBoost": XGBClassifier(
                random_state=self.config.random_state,
                eval_metric="logloss",
            ),
            "SVM": SVC(random_state=self.config.random_state, probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
        }
        return models

    def get_param_grids(self):
        """Define parameter grids for GridSearchCV"""
        param_grids = {
            "Logistic Regression": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
            },
            "Decision Tree": {
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
            },
            "SVM": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
            "Naive Bayes": {
                "var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        }
        return param_grids

    def train_single_model(self, name, model, X_train, X_test, y_train, y_test):
        """Train a single model with MLflow tracking"""

        logger.info(f"\n{'='*70}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*70}")

        with mlflow.start_run(run_name=name):
            # Log model type and configuration
            mlflow.log_param("model_type", name)
            mlflow.log_param("random_state", self.config.random_state)
            mlflow.log_param("use_grid_search", self.config.use_grid_search)
            mlflow.log_param("cv_folds", self.config.cv_folds)

            # Perform GridSearchCV if enabled
            if self.config.use_grid_search:
                param_grids = self.get_param_grids()
                param_grid = param_grids.get(name, {})

                if param_grid:
                    logger.info(f"Performing GridSearchCV with {self.config.cv_folds}-fold CV...")
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=self.config.cv_folds,
                        scoring="accuracy",
                        n_jobs=-1,
                        verbose=0,
                    )
                    grid_search.fit(X_train, y_train)

                    # Use best model
                    model = grid_search.best_estimator_

                    # Log best parameters
                    logger.info(f"Best parameters: {grid_search.best_params_}")
                    for param_name, param_value in grid_search.best_params_.items():
                        mlflow.log_param(f"best_{param_name}", param_value)

                    # Log cross-validation score
                    cv_score = grid_search.best_score_
                    mlflow.log_metric("cv_score", cv_score)
                    logger.info(f"Best CV Score: {cv_score:.4f}")
                else:
                    logger.info(f"No parameter grid defined, training with default parameters...")
                    model.fit(X_train, y_train)
            else:
                # Train with default parameters
                logger.info(f"Training {name} with default parameters...")

                # Log model-specific parameters
                if hasattr(model, "get_params"):
                    params = model.get_params()
                    for key, value in params.items():
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(key, value)

                model.fit(X_train, y_train)

            # Perform cross-validation on final model
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=self.config.cv_folds, scoring="accuracy"
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            mlflow.log_metric("cv_mean_accuracy", cv_mean)
            mlflow.log_metric("cv_std_accuracy", cv_std)
            logger.info(f"Cross-validation: {cv_mean:.4f} (+/- {cv_std:.4f})")

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
                "cv_mean_accuracy": cv_mean,
                "cv_std_accuracy": cv_std,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1_score": test_f1,
            }
            self.results.append(result)

            logger.info(f"✓ {name} - Test Accuracy: {test_acc:.4f}")
            logger.info(f"  CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
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
