# 🏦 TrustFin Bank – Loan Default Risk Analysis

A production-grade ML system that predicts loan default risk using customer financial behaviour, repayment history, credit information, employment stability, and regional economic factors.

---

## Problem Statement

TrustFin Bank has observed a rise in loan defaults, particularly among new borrowers. The risk management team needs a data-driven system that can proactively identify high-risk applicants before a loan is approved, enabling targeted intervention.

## Business Objective

Build a reliable binary classifier that predicts whether a customer will default on their loan (`TARGET = 1`) or repay it (`TARGET = 0`), and surface interpretable risk factors to support credit officer decisions.

**Primary metric**: ROC-AUC (robust to class imbalance)  
**Secondary metrics**: Precision, Recall, F1

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
│  Core Banking  │  Credit Bureau  │  App Form  │  Econ APIs  │
└───────┬──────────────┬─────────────────┬──────────┬─────────┘
        │              │                 │          │
        ▼              ▼                 ▼          ▼
┌──────────────────────────────────────────────────────────────┐
│               DATA LAYER  (data/)                            │
│  application_train.csv  bureau.csv  installments.csv  ...   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│           PREPROCESSING  (src/preprocessing.py)              │
│  • Sentinel value handling   • High-missing column drop      │
│  • Duplicate removal         • History-flag creation         │
│  • Median / zero imputation  • Label encoding                │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│        FEATURE ENGINEERING  (src/feature_engineering.py)     │
│  Credit features  │  Repayment features  │  Financial feats  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              TRAINING  (src/train.py)                        │
│  Logistic Regression  │  XGBoost  │  LightGBM               │
│  MLflow tracking → best model → models/                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────┴──────────────┐
              ▼                           ▼
┌─────────────────────────┐  ┌───────────────────────────────┐
│  EVALUATION             │  │  PREDICTION SERVICE           │
│  (src/evaluate.py)      │  │  (src/predict.py)             │
│  metrics.json           │  │  JSON risk output             │
│  plots → reports/       │  │  risk_level / probability     │
└─────────────────────────┘  └───────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │  STREAMLIT DASHBOARD         │
                              │  (app/streamlit_app.py)      │
                              │  Upload CSV · Risk scores    │
                              │  Portfolio summary · Factors │
                              └──────────────────────────────┘
```

---

## Project Structure

```
Home-Credit-Default-Risk/
├── data/                        # Raw and processed data files
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   └── Modeling.ipynb
├── src/
│   ├── preprocessing.py         # PreprocessingPipeline class
│   ├── feature_engineering.py   # FeatureEngineer class
│   ├── train.py                 # Training script (MLflow)
│   ├── evaluate.py              # Metrics + plots
│   └── predict.py               # LoanRiskPredictor class
├── models/                      # Saved artefacts (model, preprocessor, features)
├── reports/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── project_analysis.md
├── mlruns/                      # MLflow tracking directory
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the data
# Place Home Credit CSV files in data/:
#   application_train.csv, bureau.csv, bureau_balance.csv,
#   previous_application.csv, POS_CASH_balance.csv,
#   installments_payments.csv, credit_card_balance.csv
# Dataset: https://www.kaggle.com/c/home-credit-default-risk/data
```

---

## Running Training

```bash
cd Home-Credit-Default-Risk
python src/train.py --data-dir data/ --models-dir models/
```

This will:
1. Load and merge all data sources
2. Preprocess and engineer features
3. Train Logistic Regression, XGBoost, and LightGBM
4. Track all runs in MLflow (`./mlruns`)
5. Save the best model artefacts to `models/`

**View MLflow UI:**
```bash
mlflow ui
# Open http://127.0.0.1:5000
```

---

## Running Evaluation

```bash
python src/evaluate.py \
  --models-dir models/ \
  --reports-dir reports/ \
  --data-path data/processed_test.csv
```

Outputs saved to `reports/`:
- `metrics.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`

---

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Features:**
- Upload a CSV of customer records
- Run batch predictions
- View portfolio risk distribution
- Inspect per-customer risk scores, risk factors, and recommendations

---

## Running Predictions Programmatically

```python
from src.predict import LoanRiskPredictor

predictor = LoanRiskPredictor(models_dir="models/", threshold=0.5)
result = predictor.predict_single({
    "AMT_CREDIT": 500000,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_ANNUITY": 25000,
    "DAYS_EMPLOYED": -1200,
    # ... other fields
})
print(result)
# {
#   "risk_level": "HIGH",
#   "default_probability": 0.82,
#   "recommendation": ["Reduce loan amount", "Additional verification required"],
#   "risk_factors": ["High debt ratio", "Low repayment score"]
# }
```

---

## Model Explanation

### Why not just use accuracy?

The target variable is severely imbalanced (~8 % defaults). A model that always predicts "no default" achieves ~92 % accuracy yet catches zero actual defaulters — catastrophically bad for a bank. **ROC-AUC** is the primary metric because it measures ranking quality across all decision thresholds. Precision and Recall balance the asymmetric costs of false negatives (missed defaults, the more costly error) and false positives (declined creditworthy customers).

### Model selection

Three models are trained and compared:

| Model | Strengths | Imbalance handling |
|---|---|---|
| Logistic Regression | Fast, interpretable baseline | `class_weight='balanced'` |
| XGBoost | Handles missing values, high accuracy | `scale_pos_weight` |
| LightGBM | Fastest on large data, best AUC on Home Credit | `class_weight='balanced'` |

The model with the highest test ROC-AUC is automatically selected and saved.

### Key features (by importance)

1. `EXT_SOURCE_2`, `EXT_SOURCE_3` – external credit bureau scores
2. `FIN_DEBT_TO_INCOME` – loan amount / annual income
3. `REPAY_BEHAVIOUR_SCORE` – composite late-payment measure
4. `DAYS_EMPLOYED` – employment tenure
5. `BUREAU_TOTAL_OVERDUE` – outstanding overdue at bureaux

---

## Docker Deployment

```bash
# Build
docker build -t trustfin-risk .

# Run the Streamlit dashboard
docker run -p 8501:8501 trustfin-risk

# Run training inside Docker
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  trustfin-risk python src/train.py
```

---

## Future Improvements

- **SHAP explanations**: integrate SHAP to provide loan-officer-friendly feature contributions per applicant, satisfying adverse-action notice requirements
- **Threshold optimisation**: use Youden's J statistic or business-cost matrix to select the optimal decision threshold rather than a fixed 0.5
- **Time-series features**: add rolling 3/6/12-month payment behaviour windows to capture trend (improving vs. deteriorating)
- **AutoML / hyperparameter tuning**: integrate Optuna or Ray Tune for automated model search
- **Real-time scoring API**: wrap `predict.py` in a FastAPI service with async endpoints and Redis caching for sub-100 ms latency
- **Model monitoring**: integrate Evidently AI or NannyML for drift detection and automated retraining triggers
- **Graph features**: model relationships between guarantors and applicants using Graph Neural Networks (shared addresses, phone numbers, employers)

---

## Dataset

Home Credit Default Risk – [Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)

## License

For educational and portfolio use. All data is sourced from the public Kaggle competition.