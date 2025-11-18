# GridSearchCV and Cross-Validation Features

This document describes the hyperparameter tuning capabilities added to the ML pipeline.

## Overview

The `multi_model_training.py` module now includes:
- **GridSearchCV** for automated hyperparameter tuning
- **Cross-Validation** for robust model evaluation
- **MLflow tracking** of best parameters and CV scores

## Features Added

### 1. GridSearchCV Integration

Automatically searches for optimal hyperparameters for each model using grid search with cross-validation.

### 2. Parameter Grids

Comprehensive parameter grids defined for all 8 models:

#### Logistic Regression
- C: [0.1, 1.0, 10.0]
- penalty: ['l2']
- solver: ['lbfgs', 'liblinear']

#### Decision Tree
- max_depth: [5, 10, 15, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

#### Random Forest
- n_estimators: [50, 100, 200]
- max_depth: [5, 10, 15, None]
- min_samples_split: [2, 5]
- min_samples_leaf: [1, 2]

#### Gradient Boosting
- n_estimators: [50, 100, 200]
- learning_rate: [0.01, 0.1, 0.2]
- max_depth: [3, 5, 7]

#### XGBoost
- n_estimators: [50, 100, 200]
- learning_rate: [0.01, 0.1, 0.2]
- max_depth: [3, 5, 7]
- subsample: [0.8, 1.0]

#### SVM
- C: [0.1, 1.0, 10.0]
- kernel: ['rbf', 'linear']
- gamma: ['scale', 'auto']

#### K-Nearest Neighbors
- n_neighbors: [3, 5, 7, 9]
- weights: ['uniform', 'distance']
- metric: ['euclidean', 'manhattan']

#### Naive Bayes
- var_smoothing: [1e-9, 1e-8, 1e-7]

### 3. Cross-Validation

- **5-fold cross-validation** by default (configurable)
- Tracks mean CV accuracy and standard deviation
- Logged to MLflow for comparison

### 4. Configuration Options

New environment variables in `.env`:

```bash
# Enable/disable GridSearchCV
USE_GRID_SEARCH=True

# Number of cross-validation folds
CV_FOLDS=5
```

## Usage

### Enable GridSearchCV (Default)

```bash
export USE_GRID_SEARCH=True
export CV_FOLDS=5
python main.py
```

### Disable GridSearchCV (Faster Training)

```bash
export USE_GRID_SEARCH=False
python main.py
```

### Custom CV Folds

```bash
export CV_FOLDS=10  # 10-fold cross-validation
python main.py
```

## Example Output

```
======================================================================
Training: Random Forest
======================================================================
Performing GridSearchCV with 5-fold CV...
Best parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
Best CV Score: 0.8245
Cross-validation: 0.8245 (+/- 0.0320)
✓ Random Forest - Test Accuracy: 0.8045
  CV Accuracy: 0.8245 (+/- 0.0320)
  Precision: 0.7833, Recall: 0.6812, F1: 0.7287
```

## MLflow Tracking

Each model run now logs:

### Parameters
- `use_grid_search`: Whether GridSearchCV was used
- `cv_folds`: Number of cross-validation folds
- `best_<param>`: Best value for each tuned parameter (e.g., `best_max_depth`)

### Metrics
- `cv_score`: Best GridSearchCV score
- `cv_mean_accuracy`: Mean cross-validation accuracy
- `cv_std_accuracy`: Standard deviation of CV accuracy
- All existing metrics (test accuracy, precision, recall, F1)

## Performance Comparison

### Without GridSearchCV (Default Parameters)
- Faster training (~30-60 seconds)
- May not achieve optimal performance
- Good for quick prototyping

### With GridSearchCV (Optimized Parameters)
- Longer training time (~5-10 minutes)
- Better model performance
- Production-ready models
- Comprehensive hyperparameter search

## Results Structure

The results DataFrame now includes CV metrics:

| Column | Description |
|--------|-------------|
| model_name | Name of the model |
| train_accuracy | Training set accuracy |
| test_accuracy | Test set accuracy |
| **cv_mean_accuracy** | **Mean cross-validation accuracy** |
| **cv_std_accuracy** | **Standard deviation of CV accuracy** |
| test_precision | Precision on test set |
| test_recall | Recall on test set |
| test_f1_score | F1 score on test set |

## Best Practices

1. **Development**: Disable GridSearchCV for faster iterations
2. **Production**: Enable GridSearchCV for optimal models
3. **Experimentation**: Adjust parameter grids based on your dataset
4. **Validation**: Monitor CV scores alongside test scores to detect overfitting

## Example Results

Based on recent runs with GridSearchCV:

| Model | Test Acc | CV Acc (mean ± std) | Best Parameters |
|-------|----------|---------------------|-----------------|
| Random Forest | 80.45% | 82.45% ± 3.20% | n_estimators=50, max_depth=10 |
| Gradient Boosting | 80.45% | 82.31% ± 3.25% | learning_rate=0.2, max_depth=3 |
| XGBoost | 79.89% | 82.31% ± 3.13% | learning_rate=0.1, max_depth=3 |
| Logistic Regression | 79.89% | 80.34% ± 2.37% | C=0.1, solver='lbfgs' |
| Decision Tree | 78.21% | 81.19% ± 3.47% | max_depth=10, min_samples_split=10 |

## Technical Details

### GridSearchCV Configuration
- **Scoring**: Accuracy
- **Jobs**: -1 (use all available CPU cores)
- **Verbose**: 0 (quiet output)
- **Cross-validation**: Stratified K-Fold

### Performance Optimization
- Parallel processing for grid search (`n_jobs=-1`)
- Efficient parameter combinations
- Early stopping for tree-based models (where applicable)

## Troubleshooting

### GridSearchCV takes too long
- Reduce parameter grid size
- Decrease CV folds (e.g., CV_FOLDS=3)
- Use fewer parameter combinations

### Memory issues
- Reduce batch size
- Limit number of models trained simultaneously
- Use smaller parameter grids

### Inconsistent CV and test scores
- Normal if test set is small
- Consider using stratified splits
- Check for data leakage

## Future Enhancements

Potential improvements:
- RandomizedSearchCV for larger parameter spaces
- Bayesian optimization for hyperparameter tuning
- Early stopping based on CV performance
- Automated parameter range selection
- Hyperband or successive halving algorithms

## References

- [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
