# Multi-Model ML Pipeline Guide

Complete guide to the multi-model machine learning pipeline.

## 🎯 Overview

This pipeline trains and compares **10 different models**:

### Classification Models (8)
1. **Logistic Regression** - Linear classifier
2. **Decision Tree** - Tree-based classifier
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential ensemble method
5. **XGBoost** - Optimized gradient boosting
6. **SVM** - Support Vector Machine
7. **K-Nearest Neighbors** - Distance-based classifier
8. **Naive Bayes** - Probabilistic classifier

### Clustering Models (2)
1. **K-Means** - Partition-based clustering
2. **Hierarchical** - Agglomerative clustering

---

## 🚀 Quick Start

```bash
# Run complete pipeline
python main.py

# View results
python view_results.py

# Export results
python export_results.py
```

---

## 📊 Model Details

### 1. Logistic Regression
**Type:** Linear Classifier
**Best for:** Baseline, interpretable results
**Parameters:** max_iter=1000
**Expected Accuracy:** ~78-80%

### 2. Decision Tree
**Type:** Tree-based
**Best for:** Feature importance analysis
**Parameters:** max_depth=10
**Expected Accuracy:** ~75-78%

### 3. Random Forest
**Type:** Ensemble (Bagging)
**Best for:** High accuracy, robust
**Parameters:** n_estimators=100, max_depth=10
**Expected Accuracy:** ~82-84%

### 4. Gradient Boosting
**Type:** Ensemble (Boosting)
**Best for:** High performance
**Parameters:** n_estimators=100
**Expected Accuracy:** ~82-85%

### 5. XGBoost
**Type:** Optimized Gradient Boosting
**Best for:** Competition-level performance
**Parameters:** n_estimators=100, max_depth=10
**Expected Accuracy:** ~83-86%

### 6. SVM (Support Vector Machine)
**Type:** Kernel-based
**Best for:** Non-linear patterns
**Parameters:** Default RBF kernel
**Expected Accuracy:** ~80-82%

### 7. K-Nearest Neighbors
**Type:** Instance-based
**Best for:** Simple, no training
**Parameters:** n_neighbors=5
**Expected Accuracy:** ~78-80%

### 8. Naive Bayes
**Type:** Probabilistic
**Best for:** Fast, simple
**Parameters:** Gaussian distribution
**Expected Accuracy:** ~75-78%

### 9. K-Means Clustering
**Type:** Unsupervised
**Best for:** Pattern discovery
**Metrics:** Silhouette score, purity
**Parameters:** n_clusters=2

### 10. Hierarchical Clustering
**Type:** Unsupervised
**Best for:** Hierarchical patterns
**Metrics:** Silhouette score
**Parameters:** n_clusters=2, linkage='ward'

---

## 📈 Metrics Tracked

### Classification Metrics:
- **Train Accuracy** - Performance on training data
- **Test Accuracy** - Performance on test data (main metric)
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1 Score** - Harmonic mean of precision and recall

### Clustering Metrics:
- **Silhouette Score** - How well-separated clusters are (-1 to 1)
- **Inertia** - Within-cluster sum of squares (K-Means only)
- **Purity** - Cluster alignment with ground truth

---

## 🔧 Customization

### Change Model Parameters

**Edit configuration in code:**

```python
# src/components/multi_model_training.py

# Example: Increase Random Forest trees
"Random Forest": RandomForestClassifier(
    n_estimators=200,  # Change from 100
    max_depth=15,      # Change from 10
    random_state=42
)
```

### Add New Models

```python
# In get_models() method:

"Your Model": YourClassifier(
    param1=value1,
    random_state=self.config.random_state
)
```

### Disable Clustering

```python
# main.py
pipeline.run_pipeline(include_clustering=False)
```

---

## 📁 File Structure

```
src/
├── components/
│   ├── data_ingestion.py         # Load Titanic dataset
│   ├── data_processing.py        # Feature engineering
│   ├── multi_model_training.py   # 8 classifiers
│   └── clustering_models.py      # K-Means + Hierarchical
└── pipeline/
    └── complete_ml_pipeline.py   # Orchestration
```

---

## 🎯 Expected Results

### Top Performers (Test Accuracy):
1. **XGBoost**: 83-86%
2. **Gradient Boosting**: 82-85%
3. **Random Forest**: 82-84%
4. **SVM**: 80-82%
5. **Logistic Regression**: 78-80%

### Fast Models:
- Naive Bayes (fastest)
- Logistic Regression
- Decision Tree

### Slow Models:
- SVM (slowest on large data)
- K-Nearest Neighbors
- Random Forest

---

## 💡 Model Selection Guide

### Choose **Logistic Regression** if:
- Need interpretability
- Want fast training
- Linear relationships expected

### Choose **Random Forest** if:
- Need good accuracy
- Want feature importance
- Don't mind longer training

### Choose **XGBoost** if:
- Need best possible accuracy
- Have time for tuning
- Want competition-level performance

### Choose **SVM** if:
- Data is not too large
- Non-linear patterns
- Need robust results

### Choose **K-Means** if:
- Exploring data patterns
- Unsupervised learning
- Customer segmentation

---

## 📊 MLflow Experiments

### Experiments Created:

1. **titanic_multi_model** - All classification models
2. **titanic_clustering** - Clustering models

### View in MLflow UI:
- Each model = 1 run
- Compare all models side-by-side
- Download best model
- Track hyperparameters

---

## 🔍 Interpreting Results

### High Train, Low Test Accuracy
**Problem:** Overfitting
**Solution:** Reduce max_depth, add regularization

### Low Train and Test Accuracy
**Problem:** Underfitting
**Solution:** Increase model complexity

### Similar Train and Test Accuracy
**Good:** Model generalizes well
**Action:** This is what you want!

### High Precision, Low Recall
**Means:** Few false positives, many false negatives
**Impact:** Misses survivors

### High Recall, Low Precision
**Means:** Few false negatives, many false positives
**Impact:** Predicts too many survivors

---

## 🚀 Advanced Usage

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Example for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

---

## 🐛 Troubleshooting

### XGBoost Import Error
```bash
pip install xgboost==2.0.3
```

### MLflow Error
```bash
pip install --upgrade mlflow
```

### Out of Memory
- Reduce n_estimators
- Use fewer models
- Disable clustering

### Slow Training
- Reduce dataset size
- Use fewer models
- Decrease n_estimators

---

## 📚 Resources

- **Scikit-learn Docs:** https://scikit-learn.org
- **XGBoost Docs:** https://xgboost.readthedocs.io
- **MLflow Docs:** https://mlflow.org/docs

---

## 🎓 Learning Path

1. **Beginner:** Run pipeline, view results
2. **Intermediate:** Modify parameters, compare results
3. **Advanced:** Add new models, hyperparameter tuning
4. **Expert:** Custom metrics, ensemble methods

---

**Happy Learning!** 🚀
