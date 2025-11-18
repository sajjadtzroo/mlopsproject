# MLOps Project: Titanic Survival Prediction

An educational MLOps project demonstrating best practices for machine learning workflows using MLflow, Apache Airflow, scikit-learn, and multiple ML models.

## 📋 Project Structure

```
mlopsproject/
├── data/
│   ├── raw/                         # Raw data storage
│   └── processed/                   # Processed data storage
├── src/
│   ├── components/                  # ML components
│   │   ├── data_ingestion.py       # Data loading from seaborn
│   │   ├── data_processing.py      # Feature engineering & preprocessing
│   │   ├── multi_model_training.py # Multi-model training with MLflow
│   │   └── clustering_models.py    # Clustering algorithms
│   └── pipeline/
│       └── complete_ml_pipeline.py  # End-to-end pipeline orchestration
├── airflow/
│   ├── dags/                        # Airflow DAG definitions
│   │   ├── data_ingestion_dag.py   # Data ingestion workflow
│   │   └── ml_pipeline_dag.py      # Complete ML pipeline workflow
│   ├── logs/                        # Airflow execution logs
│   ├── plugins/                     # Custom Airflow plugins
│   └── config/                      # Airflow configuration
├── scripts/
│   ├── init_airflow.sh             # Initialize Airflow
│   ├── start_airflow.sh            # Start Airflow services
│   └── stop_airflow.sh             # Stop Airflow services
├── mlruns/                          # MLflow tracking data
├── docker-compose.yml               # Docker orchestration
├── .env                             # Environment variables
├── requirements.txt                 # Python dependencies
├── main.py                          # Main entry point
└── AIRFLOW_SETUP.md                 # Airflow setup guide
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python main.py
```

This will:
- Load Titanic dataset from seaborn
- Clean and process the data
- Engineer features
- Train 8 classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM, KNN, Naive Bayes)
- Perform clustering analysis (K-Means, Hierarchical)
- Track everything in MLflow

### 3. View MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open http://localhost:5000 in your browser.

## 🔄 Airflow Orchestration

This project includes Apache Airflow for orchestrating ML workflows. See [AIRFLOW_SETUP.md](AIRFLOW_SETUP.md) for detailed setup instructions.

### Quick Setup - Local

```bash
# Initialize Airflow
./scripts/init_airflow.sh

# Start Airflow services
./scripts/start_airflow.sh

# Access Airflow UI at http://localhost:8080
# Login: admin / admin
```

### Quick Setup - Docker

```bash
# Start all services (Airflow + MLflow)
docker-compose up -d

# Access services:
# - Airflow UI: http://localhost:8080 (admin/admin)
# - MLflow UI: http://localhost:5000
```

### Available DAGs

1. **Data Ingestion Pipeline** (`data_ingestion_pipeline`)
   - Schedule: Daily
   - Tasks: Ingest and validate Titanic dataset

2. **Complete ML Pipeline** (`complete_ml_pipeline`)
   - Schedule: Weekly
   - Tasks: Data ingestion → Processing → Multi-model training → Clustering → Report generation

### Trigger DAGs Manually

```bash
# Trigger data ingestion
airflow dags trigger data_ingestion_pipeline

# Trigger complete ML pipeline
airflow dags trigger complete_ml_pipeline
```

## 📊 Components

### Data Ingestion (`src/components/data_ingestion.py`)
- Loads Titanic dataset from seaborn
- Saves raw data to `data/raw/`
- Handles dataset validation

### Data Processing (`src/components/data_processing.py`)
- Cleans missing values
- Engineers features:
  - Family size
  - Is alone indicator
  - Title extraction from names
- Encodes categorical variables
- Splits data into train/test sets

### Model Training (`src/components/model_training.py`)
- Trains Random Forest classifier
- Tracks experiments with MLflow:
  - Parameters (n_estimators, max_depth, etc.)
  - Metrics (accuracy, precision, recall, F1)
  - Model artifacts
  - Feature importances
- Registers model in MLflow Model Registry

## 🔧 Configuration

Edit `.env` file to customize:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=titanic_random_forest

# Model Parameters
RANDOM_STATE=42
TEST_SIZE=0.2
N_ESTIMATORS=100
MAX_DEPTH=10
```

## 🧪 Generate Sample Data

```bash
# Generate 100 sample rows
python generate_sample_data.py

# Generate custom number of samples
python generate_sample_data.py --samples 50 --output data/my_sample.csv
```

## 🎨 Code Formatting

Format code with Black and isort:

```bash
# Format with Black
black .

# Sort imports with isort
isort .

# Or both
black . && isort .
```

Configuration is in `pyproject.toml`.

## 📈 MLflow Features

This project tracks:
- **Parameters**: Model hyperparameters
- **Metrics**: Accuracy, precision, recall, F1 score
- **Artifacts**:
  - Trained model
  - Feature importance CSV
- **Models**: Registered in MLflow Model Registry

## 🎓 Educational Concepts

This project demonstrates:

1. **Component-Based Architecture**: Modular, reusable components
2. **MLflow Integration**: Experiment tracking and model registry
3. **Airflow Orchestration**: Workflow scheduling and monitoring with DAGs
4. **Multi-Model Training**: Training and comparing 8+ ML models
5. **Configuration Management**: Environment variables via .env
6. **Code Quality**: Black and isort for consistent formatting
7. **Containerization**: Docker Compose for reproducible deployments
8. **Logging**: Comprehensive logging throughout pipeline
9. **Best Practices**:
   - Type hints
   - Docstrings
   - Error handling
   - Separation of concerns
   - Pipeline orchestration

## 📝 Running Individual Components

Each component can be run independently:

```bash
# Data ingestion only
python src/components/data_ingestion.py

# Data processing only
python src/components/data_processing.py

# Model training only (requires processed data)
python src/components/model_training.py
```

## 🔍 Dataset Information

**Titanic Dataset** from seaborn/scikit-learn:
- **Samples**: ~891 passengers
- **Target**: Survived (0 = No, 1 = Yes)
- **Features**:
  - Passenger class
  - Age, Sex
  - Siblings/Spouses aboard
  - Parents/Children aboard
  - Fare
  - Embarked port
  - And more...

## 📚 Next Steps

1. Set up Airflow for automated pipeline execution (see [AIRFLOW_SETUP.md](AIRFLOW_SETUP.md))
2. Experiment with different hyperparameters in `.env`
3. Configure DAG schedules for your use case
4. Add email notifications for Airflow task failures
5. Implement model serving with MLflow
6. Add cross-validation to model training
7. Set up data quality checks in Airflow
8. Add unit tests
9. Set up CI/CD pipeline
10. Deploy with Docker Compose in production

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with the code
- Add new features
- Try different ML algorithms
- Improve documentation

## 📄 License

This project is for educational purposes.