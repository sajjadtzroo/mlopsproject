"""
Data Ingestion Component
This component handles loading the Titanic dataset from seaborn
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""

    raw_data_path: str = os.getenv("RAW_DATA_PATH", "data/raw/titanic.csv")


class DataIngestion:
    """
    Component for ingesting raw data from various sources
    For this example, we load the Titanic dataset from seaborn
    """

    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> str:
        """
        Load the Titanic dataset and save it to the raw data path

        Returns:
            str: Path to the saved raw data file
        """
        logger.info("Starting data ingestion process")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Load Titanic dataset from seaborn
            logger.info("Loading Titanic dataset from seaborn")
            df = sns.load_dataset("titanic")

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw data saved to {self.config.raw_data_path}")

            # Log dataset info
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Dataset columns: {df.columns.tolist()}")

            return self.config.raw_data_path

        except Exception as e:
            logger.error(f"Error during data ingestion: {str(e)}")
            raise e


if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion()
    data_path = ingestion.initiate_data_ingestion()
    print(f"Data ingestion completed. Data saved to: {data_path}")
