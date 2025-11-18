# MLflow UI User Guide

Complete guide to using the MLflow UI for tracking your Titanic ML experiments.

## 📋 Table of Contents

1. [Accessing the UI](#accessing-the-ui)
2. [Main Interface Overview](#main-interface-overview)
3. [Viewing Experiments](#viewing-experiments)
4. [Comparing Runs](#comparing-runs)
5. [Viewing Metrics & Charts](#viewing-metrics--charts)
6. [Downloading Artifacts](#downloading-artifacts)
7. [Model Registry](#model-registry)
8. [Tips & Tricks](#tips--tricks)

---

## 🌐 Accessing the UI

### From Codespaces:
1. Look at the **PORTS** tab at the bottom of VS Code
2. Find **Port 5000** in the list
3. Click the **🌐 globe icon** to open in browser
4. Or hover and click **"Open in Browser"**

### The URL will look like:
```
https://[codespace-name]-5000.app.github.dev
```

---

## 🏠 Main Interface Overview

When you open MLflow UI, you'll see:

### Top Navigation Bar:
- **Experiments** - View all experiments and runs
- **Models** - Access the Model Registry
- **Settings** - Configure UI preferences

### Left Sidebar:
- **Experiment list** - All your experiments
- **Search & Filter** - Find specific runs
- **Tags** - Organize experiments

### Main Panel:
- **Runs table** - List of all experiment runs
- **Metrics, Parameters, Tags** columns
- **Sort and filter** options

---

## 🔬 Viewing Experiments

### Step 1: Select Your Experiment
1. In the left sidebar, click **"titanic_random_forest"**
2. You'll see all runs for this experiment

### Step 2: Understanding the Runs Table

**Columns you'll see:**

| Column | Description | Your Values |
|--------|-------------|-------------|
| **Run Name** | Auto-generated or custom | Various timestamps |
| **Created** | When the run started | Recent dates |
| **User** | Who ran it | Your username |
| **Source** | Script that created it | model_training.py |
| **Version** | Git commit (if available) | May be empty |

**Metrics columns:**
- `test_accuracy` - ~0.8212 (82.12%)
- `test_f1_score` - ~0.7538 (75.38%)
- `test_precision` - ~0.8033
- `test_recall` - ~0.7101
- `train_accuracy` - ~0.9551

**Parameters columns:**
- `n_estimators` - 100
- `max_depth` - 10
- `random_state` - 42
- `model_type` - RandomForestClassifier

### Step 3: Click on a Run
Click any run row to see **full details**

---

## 📊 Run Details Page

When you click on a run, you'll see several tabs:

### 1️⃣ **Overview Tab**

**What you'll see:**
- Run metadata (ID, start time, duration)
- Git commit info (if available)
- Tags

### 2️⃣ **Parameters Tab**

**All hyperparameters:**
```
n_estimators: 100
max_depth: 10
random_state: 42
model_type: RandomForestClassifier
```

**What to do:**
- Review what settings were used
- Compare with other runs
- Identify best configurations

### 3️⃣ **Metrics Tab**

**All logged metrics:**
```
train_accuracy: 0.9551
test_accuracy: 0.8212
test_precision: 0.8033
test_recall: 0.7101
test_f1_score: 0.7538
```

**Charts available:**
- Line plots (if you log metrics over time)
- Compare multiple runs side-by-side

### 4️⃣ **Artifacts Tab**

**Files stored with this run:**

📁 **Artifacts directory:**
- 📄 `feature_importance.csv` - Which features matter most
- 📦 `random_forest_model/` - Trained model directory
  - `MLmodel` - Model metadata
  - `model.pkl` - Serialized model
  - `conda.yaml` - Environment specs
  - `requirements.txt` - Python dependencies
  - `python_env.yaml` - Python environment

**How to download:**
1. Click on any file/folder name
2. Click **"Download"** button
3. File saves to your computer

### 5️⃣ **Tags Tab**

- Custom tags you add to organize runs
- Add tags by clicking **"+ Add Tag"**

---

## 🔄 Comparing Runs

### Compare Multiple Runs:

**Step 1: Select runs to compare**
1. Go back to the experiment view
2. **Check the boxes** next to 2+ runs
3. Click **"Compare"** button at the top

**Step 2: View Comparison**

You'll see three sections:

#### A) **Parallel Coordinates Plot**
- Visual comparison of all parameters and metrics
- Lines represent different runs
- Hover to see values

#### B) **Scatter Plot**
- X-axis: Choose any parameter/metric
- Y-axis: Choose any parameter/metric
- Example: max_depth vs test_accuracy
- Identify correlations

#### C) **Comparison Table**
- Side-by-side parameter values
- Side-by-side metric values
- **Differences highlighted**
- Easy to spot what changed

**Step 3: Find Best Run**
1. Sort by `test_accuracy` (click column header)
2. Highest value = best model
3. Note the parameters used

---

## 📈 Viewing Metrics & Charts

### Chart Panel (right side):

**1. Metrics Plot**
- Shows metric values across runs
- Can plot multiple metrics together
- Useful for tracking improvement

**2. Parallel Coordinates Plot**
- Multi-dimensional comparison
- See relationships between params and metrics

**3. Scatter Plot**
- X vs Y comparison
- Great for finding optimal hyperparameters

**4. Contour Plot**
- Heatmap view of parameter space
- Shows regions of high/low performance

### Creating Custom Charts:

**Step 1:** Click **"Chart"** button
**Step 2:** Choose chart type
**Step 3:** Select X and Y axes
**Step 4:** Add runs to compare
**Step 5:** Customize colors, ranges

---

## 📥 Downloading Artifacts

### Download Feature Importance:

1. Click on your run
2. Go to **Artifacts** tab
3. Click **`feature_importance.csv`**
4. Click **Download** button
5. Open in Excel/Sheets

**What you'll see:**
```
feature,importance
sex,0.2989
fare,0.2428
age,0.2016
pclass,0.0894
...
```

### Download Trained Model:

1. Click on `random_forest_model` folder
2. Click **Download** next to the folder
3. Gets a ZIP file with:
   - Serialized model
   - Environment files
   - Metadata

### Load Model in Python:
```python
import mlflow

# Load model by run ID
model = mlflow.sklearn.load_model("runs:/<RUN_ID>/random_forest_model")

# Or by model name and version
model = mlflow.sklearn.load_model("models:/TitanicRandomForest/1")

# Make predictions
predictions = model.predict(X_test)
```

---

## 🏷️ Model Registry

### View Registered Models:

1. Click **"Models"** in top navigation
2. You'll see **"TitanicRandomForest"** model
3. Click on it to see versions

### Model Versions:

**What you'll see:**
- Version 1, Version 2, etc.
- Which run created each version
- Stage: None, Staging, Production, or Archived
- Description and tags

### Promote Model to Production:

**Step 1:** Click on a version (e.g., Version 2)
**Step 2:** Click **"Stage"** dropdown
**Step 3:** Select **"Production"**
**Step 4:** Add description: "Best model - 82% accuracy"

**Step 5:** Use in production:
```python
import mlflow

# Load production model
model = mlflow.pyfunc.load_model("models:/TitanicRandomForest/Production")
```

### Compare Model Versions:

1. Select multiple versions (checkboxes)
2. Click **"Compare"**
3. See metric differences
4. Decide which to promote

---

## 💡 Tips & Tricks

### 🔍 Search & Filter

**Filter by metrics:**
```
metrics.test_accuracy > 0.80
```

**Filter by parameters:**
```
params.n_estimators = "100"
```

**Combine filters:**
```
metrics.test_accuracy > 0.80 AND params.max_depth = "10"
```

### 🏃 Run Management

**Add tags to runs:**
- Click run → Tags tab → Add Tag
- Examples: "best_model", "baseline", "experiment_v1"

**Add notes:**
- Click run → Description field
- Document what you tried and why

**Delete runs:**
- Click run → Delete button (trash icon)
- Use cautiously!

### 📊 Export Data

**Export runs table:**
1. Select runs
2. Click **"Download CSV"**
3. Opens in Excel for analysis

### ⚙️ Settings

**Customize columns:**
1. Click **"Columns"** button
2. Show/hide parameters and metrics
3. Reorder columns

**Change timezone:**
- Settings → Timezone
- Adjust to your local time

### 🔗 Share Results

**Share run URL:**
- Copy browser URL
- Send to teammates
- Direct link to specific run

**Share comparison:**
- Create comparison view
- Copy URL
- Share analysis with team

---

## 🎯 Common Workflows

### Workflow 1: Find Best Model

1. Go to experiment
2. Click **"test_accuracy"** column to sort
3. Top row = best model
4. Click to see details
5. Note parameters used

### Workflow 2: Compare Hyperparameters

1. Select multiple runs
2. Click **Compare**
3. View scatter plot
4. X-axis: n_estimators
5. Y-axis: test_accuracy
6. Find optimal value

### Workflow 3: Download Best Model

1. Find best run
2. Go to Artifacts tab
3. Download `random_forest_model`
4. Use in production

### Workflow 4: Track Experiments Over Time

1. Add descriptive tags to runs
2. Filter by tags
3. Compare metrics
4. Document improvements

---

## 🚀 Next Steps

**Experiment with hyperparameters:**
1. Edit `.env` file
2. Change `N_ESTIMATORS` or `MAX_DEPTH`
3. Run `python main.py`
4. Compare in MLflow UI
5. Find best configuration

**Try different models:**
1. Modify `model_training.py`
2. Try LogisticRegression, XGBoost
3. Compare in UI
4. See which performs best

**Track more metrics:**
1. Add custom metrics in code
2. Log to MLflow
3. View in UI
4. Create custom charts

---

## 🆘 Troubleshooting

**Can't see experiments:**
- Refresh browser
- Check tracking URI is correct
- Verify runs exist: `ls mlruns/`

**Port forwarding not working:**
- Check PORTS tab in VS Code
- Make sure port 5000 is forwarded
- Try restarting MLflow UI

**Charts not loading:**
- Refresh page
- Clear browser cache
- Check browser console for errors

**Model Registry empty:**
- Verify model was registered during training
- Check model_training.py logs
- Re-run pipeline if needed

---

## 📚 Resources

- **MLflow Docs:** https://mlflow.org/docs/latest/index.html
- **Tracking Guide:** https://mlflow.org/docs/latest/tracking.html
- **Model Registry:** https://mlflow.org/docs/latest/model-registry.html

---

**Happy Experimenting!** 🎉

For terminal-based viewing (no UI needed):
```bash
python view_results.py
python export_results.py
```
