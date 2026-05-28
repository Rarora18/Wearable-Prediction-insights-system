# Wearable Prediction & Insights System

Sleep quality classification from wearable time-series data using **Python**, **Pandas**, **NumPy**, **scikit-learn**, and an interactive **Streamlit** dashboard.

## What this project does

- Generates (or loads) wearable time-series data: heart rate, activity, sleep stage/cycle proxies
- Engineers temporal + physiological features:
  - rolling HRV approximations (RMSSD-like proxy, rolling std, deltas)
  - activity intensity & load (rolling sums, ratios)
  - circadian rhythm indicators (hour-of-day sin/cos, “night window”, sleep schedule stability)
- Trains a binary classifier to predict **low vs high sleep quality** and reports **ROC-AUC**
- Provides a Streamlit app for:
  - real-time-ish prediction visualization on a selected user/day
  - global + local feature importance (permutation importance + model coefficients)
  - personalized behavioral recommendations derived from feature contributions

## Quickstart

### 1) Create environment & install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train the model

```bash
python -m src.train --generate-data --n-users 200 --days-per-user 10
```

Artifacts are saved to `artifacts/`:
- `artifacts/model.joblib`
- `artifacts/feature_schema.json`
- `artifacts/metrics.json`

### 3) Run the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

## Project layout

```
app/
  streamlit_app.py
artifacts/
data/
  raw/
  processed/
src/
  __init__.py
  data_generation.py
  features.py
  modeling.py
  train.py
```

## Notes

- The training script can generate a synthetic dataset for demonstration. If you have real wearable exports, drop them into `data/raw/` and adapt `src/train.py` to read them.
- Target label is binary: `sleep_quality` in {0 (low), 1 (high)}.

