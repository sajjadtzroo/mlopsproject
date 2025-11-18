# MLflow UI - Quick Start Guide

## 🚀 5-Minute Quick Start

### Step 1: Access MLflow UI (30 seconds)
```
1. Look at bottom of VS Code → PORTS tab
2. Find Port 5000
3. Click 🌐 globe icon
4. Browser opens with MLflow UI
```

### Step 2: View Your Experiment (1 minute)
```
1. Left sidebar → Click "titanic_random_forest"
2. You'll see 2 runs with metrics
3. Look for test_accuracy column (should be ~0.82)
```

### Step 3: Explore a Run (2 minutes)
```
1. Click on any run row
2. Tabs appear: Overview, Parameters, Metrics, Artifacts
3. Click "Metrics" → See all accuracy scores
4. Click "Artifacts" → See trained model files
5. Click "feature_importance.csv" → Download it
```

### Step 4: Compare Runs (1.5 minutes)
```
1. Go back (click experiment name)
2. Check boxes next to 2 runs
3. Click "Compare" button at top
4. See side-by-side comparison
5. Scroll down for charts
```

---

## 📊 Key UI Sections

### Main Table Columns:
- **Run Name** - Identifies each training run
- **test_accuracy** - 82% (your model's performance)
- **test_f1_score** - 75% (balanced metric)
- **n_estimators** - 100 (number of trees)
- **max_depth** - 10 (tree depth)

### Tabs in Run Details:
1. **Overview** - Run metadata
2. **Parameters** - Hyperparameters used
3. **Metrics** - Performance scores
4. **Artifacts** - Model files, CSVs
5. **Tags** - Custom labels

---

## 🎯 Most Useful Features

### 1. Sort by Best Performance
**Click the "test_accuracy" column header**
- Highest value = best model
- See which parameters work best

### 2. Download Feature Importance
```
Run details → Artifacts tab →
Click "feature_importance.csv" → Download
```
**Shows:**
- sex: 29.89% (most important!)
- fare: 24.28%
- age: 20.16%

### 3. Compare Different Models
```
Select 2+ runs → Compare button
```
**See:**
- Which parameters changed
- Which metrics improved
- Visual charts comparing performance

### 4. Download Trained Model
```
Artifacts tab → random_forest_model → Download
```
**Use it:**
```python
import mlflow
model = mlflow.sklearn.load_model("runs:/RUN_ID/random_forest_model")
predictions = model.predict(new_data)
```

---

## 🔍 Search & Filter

### Filter for Good Models:
```
In search box, type:
metrics.test_accuracy > 0.80
```

### Filter by Parameters:
```
params.n_estimators = "100"
```

### Combine Filters:
```
metrics.test_accuracy > 0.80 AND params.max_depth = "10"
```

---

## 📈 Chart Options

### Available Charts:
1. **Parallel Coordinates** - Multi-dimensional view
2. **Scatter Plot** - X vs Y comparison
3. **Line Chart** - Trends over time
4. **Contour Plot** - Parameter space heatmap

### Create Custom Chart:
```
1. Select runs (checkboxes)
2. Click "Chart" button
3. Choose chart type
4. Select X and Y axes
5. Click "Apply"
```

**Example:**
- X-axis: n_estimators
- Y-axis: test_accuracy
- Find optimal number of trees!

---

## 🏷️ Model Registry Quick View

### Access:
```
Top nav bar → Click "Models"
```

### You'll See:
- **TitanicRandomForest** model
- Version 1, Version 2
- Current stage (None/Staging/Production)

### Promote to Production:
```
1. Click model name
2. Click a version
3. Stage dropdown → Select "Production"
4. Model is now live!
```

### Use Production Model:
```python
import mlflow
model = mlflow.pyfunc.load_model(
    "models:/TitanicRandomForest/Production"
)
```

---

## 💾 Export & Download

### Export Comparison Table:
```
Compare runs → Download CSV button
```

### Download Individual Files:
```
Run → Artifacts → Click filename → Download
```

### Download Entire Model:
```
Artifacts → random_forest_model folder → Download
```

---

## 🎨 Customize View

### Show/Hide Columns:
```
Click "Columns" button (top right)
Check/uncheck parameters and metrics
```

### Recommended Columns:
- ✅ test_accuracy
- ✅ test_f1_score
- ✅ n_estimators
- ✅ max_depth
- ❌ train_accuracy (less important)

---

## 🔥 Pro Tips

### Tip 1: Name Your Runs
```python
# In your code:
with mlflow.start_run(run_name="high_depth_experiment"):
    # training code
```

### Tip 2: Add Tags
```
Run details → Tags tab → + Add Tag
Examples: "baseline", "best_so_far", "production_candidate"
```

### Tip 3: Add Descriptions
```
Click run → Description field →
Write: "Increased max_depth to 15, accuracy improved by 2%"
```

### Tip 4: Bookmark Good Runs
```
Copy run URL from browser
Save in notes for later reference
```

### Tip 5: Regular Cleanup
```
Delete failed/bad runs to keep UI clean
Select run → Delete button (trash icon)
```

---

## ⚡ Keyboard Shortcuts

- `Ctrl/Cmd + Click` - Select multiple runs
- `Shift + Click` - Select range of runs
- `Esc` - Close modal/dialog
- `Ctrl/Cmd + F` - Search

---

## 🆘 Common Issues

**Issue:** Can't see MLflow UI
**Fix:** Check PORTS tab → Port 5000 → Click globe icon

**Issue:** No experiments shown
**Fix:** Make sure you ran: `python main.py`

**Issue:** Artifacts not loading
**Fix:** Refresh browser page

**Issue:** Charts not appearing
**Fix:** Select 2+ runs first, then click Compare

---

## 📱 Mobile Access

MLflow UI works on mobile browsers:
```
1. Forward port 5000
2. Open forwarded URL on phone
3. View-only (editing limited)
```

---

## 🎯 Your Next Actions

### Action 1: Explore Current Runs (5 min)
```bash
# MLflow UI should be open
# Click around and explore the 2 existing runs
```

### Action 2: Run New Experiment (10 min)
```bash
# Edit .env file
nano .env  # Change N_ESTIMATORS to 200

# Run pipeline
python main.py

# Check MLflow UI - new run appears!
# Compare with previous runs
```

### Action 3: Find Best Model (5 min)
```bash
# In MLflow UI:
# 1. Sort by test_accuracy
# 2. Note best parameters
# 3. Download that model
# 4. Use in production
```

---

## 📖 Full Guide

For detailed explanations, see: **MLFLOW_UI_GUIDE.md**

For terminal viewing (no UI): **`python view_results.py`**

---

**You're ready to use MLflow UI!** 🎉

Quick access: Check PORTS tab → Port 5000 → 🌐
