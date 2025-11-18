# Quick Reference Guide

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (10 models)
python main.py

# View results in terminal
python view_results.py

# Export to files
python export_results.py
```

## 📊 What Gets Trained

### 8 Classification Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest ⭐ (Best: 82%)
4. Gradient Boosting
5. XGBoost
6. SVM
7. K-Nearest Neighbors
8. Naive Bayes

### 2 Clustering Models:
1. K-Means
2. Hierarchical (Agglomerative)

## 📁 Project Structure

```
mlopsproject/
├── main.py                      # Run this!
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_processing.py
│   │   ├── multi_model_training.py   # 8 classifiers
│   │   └── clustering_models.py      # 2 clustering
│   └── pipeline/
│       └── complete_ml_pipeline.py
├── view_results.py              # View in terminal
├── export_results.py            # Export to files
└── .env                         # Configuration
```

## ⚙️ Configuration (.env)

```bash
# MLflow
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=titanic_multi_model

# Data
RAW_DATA_PATH=data/raw/titanic.csv
PROCESSED_DATA_PATH=data/processed/titanic_processed.csv

# Parameters
RANDOM_STATE=42
TEST_SIZE=0.2
N_CLUSTERS=2
```

## 📈 View Results

### Terminal:
```bash
python view_results.py
```

### Files:
```bash
python export_results.py
# Creates results/ folder with:
# - summary.txt
# - runs_comparison.csv
# - best_run.txt
# - best_run.json
```

### MLflow UI:
```bash
# Already running on port 5000
# Check PORTS tab in VS Code → Port 5000 → Click globe icon
```

## 🎯 Expected Results

**Best Models:**
- Random Forest: ~82%
- Gradient Boosting: ~81%
- XGBoost: ~80%
- Logistic Regression: ~80%

**Weak Models:**
- SVM: ~63% (needs tuning)
- KNN: ~67%

## 🔧 Customize

### Change Parameters

**Edit individual models:**
```python
# src/components/multi_model_training.py
# Find get_models() method and modify parameters
```

### Disable Clustering

```python
# main.py
pipeline.run_pipeline(include_clustering=False)
```

### Add New Model

```python
# In get_models() method:
"My Model": MyClassifier(param=value)
```

## 📚 Documentation

- **MODELS_GUIDE.md** - Detailed model information
- **MLFLOW_UI_GUIDE.md** - MLflow UI tutorial
- **MLFLOW_QUICK_START.md** - Quick MLflow guide
- **README.md** - Full project documentation

## 🎓 Common Commands

```bash
# Format code
black . && isort .

# Generate sample data
python generate_sample_data.py --samples 100

# View specific file
cat results/best_run.txt

# Check project structure
tree -L 3 -I '__pycache__|*.pyc'
```

## ⚡ Troubleshooting

**XGBoost error:**
```bash
pip install xgboost==2.0.3
```

**MLflow UI not working:**
```bash
# Use terminal viewer instead
python view_results.py
```

**Out of memory:**
```bash
# Edit main.py
pipeline.run_pipeline(include_clustering=False)
```

## 🎯 Next Steps

1. **View results:** `python view_results.py`
2. **Explore MLflow UI:** PORTS tab → Port 5000
3. **Read guides:** Start with `MODELS_GUIDE.md`
4. **Experiment:** Change parameters in `.env`
5. **Compare:** Run again with different settings

---

**Quick Help:**
- Run pipeline: `python main.py`
- View results: `python view_results.py`
- MLflow UI: Check PORTS tab → Port 5000
