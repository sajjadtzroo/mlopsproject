# MLOps Project: Titanic Survival Prediction

An educational MLOps project demonstrating best practices for machine learning workflows using MLflow, scikit-learn, and Random Forest.

## 📋 Project Structure

```
mlopsproject/
├── data/
│   ├── raw/                    # Raw data storage
│   └── processed/              # Processed data storage
├── src/
│   ├── components/             # ML components
│   │   ├── data_ingestion.py  # Data loading from seaborn
│   │   ├── data_processing.py # Feature engineering & preprocessing
│   │   └── model_training.py  # Model training with MLflow
│   └── pipeline/
│       └── training_pipeline.py # End-to-end pipeline orchestration
├── mlruns/                     # MLflow tracking data
├── logs/                       # Application logs
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Black & isort configuration
├── main.py                    # Main entry point
└── generate_sample_data.py    # Sample data generation
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
- Train a Random Forest model
- Track everything in MLflow

### 3. View MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open http://localhost:5000 in your browser.

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
3. **Configuration Management**: Environment variables via .env
4. **Code Quality**: Black and isort for consistent formatting
5. **Logging**: Comprehensive logging throughout pipeline
6. **Best Practices**:
   - Type hints
   - Docstrings
   - Error handling
   - Separation of concerns

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

1. Experiment with different hyperparameters in `.env`
2. Try different models (Logistic Regression, XGBoost, etc.)
3. Add cross-validation
4. Implement model serving with MLflow
5. Add unit tests
6. Set up CI/CD pipeline

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with the code
- Add new features
- Try different ML algorithms
- Improve documentation

## 📄 License

This project is for educational purposes.