"""
Data Processing Component
This component handles data cleaning, feature engineering, and preprocessing
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataProcessingConfig:
    """Configuration for data processing"""

    processed_data_path: str = os.getenv(
        "PROCESSED_DATA_PATH", "data/processed/titanic_processed.csv"
    )
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))
    random_state: int = int(os.getenv("RANDOM_STATE", 42))


class DataProcessing:
    """
    Component for processing and transforming raw data
    Handles missing values, feature engineering, and data splitting
    """

    def __init__(self):
        self.config = DataProcessingConfig()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and dropping unnecessary columns

        Args:
            df: Raw dataframe

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Cleaning data...")

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Fill missing values
        df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
        df_clean["embarked"] = df_clean["embarked"].fillna(
            df_clean["embarked"].mode()[0]
        )
        df_clean["fare"] = df_clean["fare"].fillna(df_clean["fare"].median())

        # Drop columns with too many missing values or not useful
        columns_to_drop = ["deck", "embark_town"]
        df_clean = df_clean.drop(columns=columns_to_drop, errors="ignore")

        logger.info(f"Data cleaned. Shape: {df_clean.shape}")
        return df_clean

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features and encode categorical variables

        Args:
            df: Cleaned dataframe

        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Engineering features...")

        df_fe = df.copy()

        # Create family size feature
        df_fe["family_size"] = df_fe["sibsp"] + df_fe["parch"] + 1

        # Create is_alone feature
        df_fe["is_alone"] = (df_fe["family_size"] == 1).astype(int)

        # Extract title from name
        if "name" in df_fe.columns:
            df_fe["title"] = df_fe["name"].str.extract(" ([A-Za-z]+)\.", expand=False)
            # Group rare titles
            df_fe["title"] = df_fe["title"].replace(
                [
                    "Lady",
                    "Countess",
                    "Capt",
                    "Col",
                    "Don",
                    "Dr",
                    "Major",
                    "Rev",
                    "Sir",
                    "Jonkheer",
                    "Dona",
                ],
                "Rare",
            )
            df_fe["title"] = df_fe["title"].replace("Mlle", "Miss")
            df_fe["title"] = df_fe["title"].replace("Ms", "Miss")
            df_fe["title"] = df_fe["title"].replace("Mme", "Mrs")

        # Encode categorical variables
        df_fe["sex"] = df_fe["sex"].map({"male": 0, "female": 1})

        # One-hot encode embarked
        if "embarked" in df_fe.columns:
            df_fe = pd.get_dummies(df_fe, columns=["embarked"], prefix="embarked")

        # One-hot encode title
        if "title" in df_fe.columns:
            df_fe = pd.get_dummies(df_fe, columns=["title"], prefix="title")

        # Drop non-numeric columns
        columns_to_drop = [
            "name",
            "ticket",
            "cabin",
            "alive",
            "class",
            "who",
            "adult_male",
        ]
        df_fe = df_fe.drop(columns=columns_to_drop, errors="ignore")

        logger.info(f"Features engineered. Shape: {df_fe.shape}")
        logger.info(f"Final columns: {df_fe.columns.tolist()}")

        return df_fe

    def initiate_data_processing(
        self, raw_data_path: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Process the raw data and split into train/test sets

        Args:
            raw_data_path: Path to raw data file

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Starting data processing")

        try:
            # Load raw data
            logger.info(f"Loading raw data from {raw_data_path}")
            df = pd.read_csv(raw_data_path)

            # Clean data
            df_clean = self.clean_data(df)

            # Feature engineering
            df_processed = self.feature_engineering(df_clean)

            # Separate features and target
            X = df_processed.drop("survived", axis=1)
            y = df_processed["survived"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y,
            )

            logger.info(f"Train set size: {X_train.shape}")
            logger.info(f"Test set size: {X_test.shape}")

            # Save processed data
            os.makedirs(os.path.dirname(self.config.processed_data_path), exist_ok=True)
            df_processed.to_csv(self.config.processed_data_path, index=False)
            logger.info(f"Processed data saved to {self.config.processed_data_path}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
            raise e


if __name__ == "__main__":
    # Example usage
    processing = DataProcessing()
    X_train, X_test, y_train, y_test = processing.initiate_data_processing(
        "data/raw/titanic.csv"
    )
    print(f"Data processing completed.")
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
