# DPS Web Application — README

## Overview
A Flask-based web application that demonstrates and compares two approaches to computing the Developer Productivity Score (DPS):

1. **Statistical DPS Model** (White-Box) — Hierarchical entropy-weighted composite index
2. **XGBoost ML Model** (Black-Box) — Gradient boosted trees trained on the same features

## Project Structure
```
webapp/
├── app.py                          # Flask entry point
├── README.md                       # This file
├── models/
│   ├── __init__.py
│   ├── statistical_model.py        # White-box DPS computation
│   └── xgboost_model.py            # XGBoost model wrapper
├── trained_models/
│   └── dps_xgboost_model.json      # Pre-trained XGBoost model
└── templates/
    ├── index.html                  # Main prediction page
    └── about.html                  # Model explanation page
```

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
cd webapp
pip install flask numpy pandas xgboost scikit-learn
```

### Running
```bash
python app.py
```
Open http://127.0.0.1:5000 in your browser.

## How It Works

### Input
A single form collects 15 PR characteristics grouped by dimension (TES, CQS, CLS, AIS, RCS).

### Processing
The same inputs are passed to **both** models simultaneously:
- **Statistical Model** applies the exact entropy-weighted formula from `dps_pipeline_v2.py`
- **XGBoost Model** loads the pre-trained model and runs inference

### Output
Side-by-side comparison showing:
- DPS score from both models
- Productivity classification (High / Moderate / Low)
- Dimension score breakdown (statistical model only)
- Feature contribution chart
- Feature importance chart (XGBoost)
- Radar chart of dimension scores
- Full explainability summary
- Model comparison table

## Model Details

### Statistical Model
- **Type**: Unsupervised composite index
- **Weighting**: Shannon entropy (within-dimension) + equal (across dimensions)
- **Normalization**: RobustScaler
- **Reference**: OECD/JRC Handbook on Composite Indicators (2008)

### XGBoost Model
- **Type**: Supervised regression (gradient boosted trees)
- **Training Data**: 22,258 PRs from GitHub
- **Performance**: R² = 0.998, RMSE = 0.0036
- **Features**: 29 (14 pipeline + 15 behavioral)
