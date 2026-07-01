"""
predict.py
----------
Prediction service for TrustFin Bank Loan Default Risk project.

Loads trained artefacts from models/ and produces structured risk assessments.

Usage (CLI)
-----------
    python src/predict.py --input data/sample_customer.json
    python src/predict.py --input data/customers.csv --output reports/predictions.json

Usage (programmatic)
--------------------
    from predict import LoanRiskPredictor
    predictor = LoanRiskPredictor()
    result = predictor.predict_single(customer_dict)
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Risk thresholds and recommendation logic
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {
    "LOW":    (0.00, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH":   (0.60, 1.01),
}

RISK_RECOMMENDATIONS: Dict[str, List[str]] = {
    "LOW": [
        "Standard loan terms applicable",
        "No additional documentation required",
        "Eligible for preferred interest rate",
    ],
    "MEDIUM": [
        "Consider reducing loan amount by 20–30 %",
        "Request additional income verification",
        "Apply standard interest rate with quarterly review",
    ],
    "HIGH": [
        "Reduce loan amount significantly or decline",
        "Require co-signer or collateral",
        "Additional identity and employment verification required",
        "Escalate to senior credit officer for manual review",
    ],
}

# Human-readable names for the engineered features used in risk-factor reporting
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "FIN_DEBT_TO_INCOME":          "High debt-to-income ratio",
    "REPAY_BEHAVIOUR_SCORE":       "Poor repayment behaviour",
    "FIN_FINANCIAL_STRESS_INDEX":  "Elevated financial stress",
    "CREDIT_DEFAULT_FLAG":         "Previous default on record",
    "CREDIT_RISK_SCORE":           "High credit risk score",
    "REPAY_LATE_PAYMENT_RATIO":    "High late-payment rate",
    "FIN_ANNUITY_TO_INCOME":       "High annuity-to-income ratio",
    "CREDIT_UTILISATION":          "High credit utilisation",
    "FIN_INCOME_STABILITY_SCORE":  "Low income stability",
    "EXT_SOURCE_2":                "Low external credit score (EXT_SOURCE_2)",
    "EXT_SOURCE_3":                "Low external credit score (EXT_SOURCE_3)",
}


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class LoanRiskPredictor:
    """
    Load saved artefacts and produce structured loan risk assessments.

    Parameters
    ----------
    models_dir : path to directory containing model.pkl, preprocessing.pkl,
                 feature_list.pkl
    threshold  : probability threshold above which a customer is predicted
                 to default (default: 0.5)
    """

    def __init__(self, models_dir: Union[str, Path] = Path("models"),
                 threshold: float = 0.5):
        self.models_dir = Path(models_dir)
        
        meta_path = self.models_dir / "model_meta.json"
        self._meta = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self._meta = json.load(f)
        
        # Load threshold from metadata if available, otherwise use passed threshold
        self.threshold = threshold
        if "optimal_threshold" in self._meta:
            self.threshold = self._meta["optimal_threshold"]
            logger.info("Using optimal threshold from model metadata: %.2f", self.threshold)
            
        self._model        = None
        self._preprocessor = None
        self._feature_list: List[str] = []
        self._load_artefacts()

    # ------------------------------------------------------------------
    def _load_artefacts(self) -> None:
        model_lr_path = self.models_dir / "model_lr.pkl"
        model_xgb_path = self.models_dir / "model_xgb.pkl"
        model_lgb_path = self.models_dir / "model_lgb.pkl"
        knn_path       = self.models_dir / "knn_transformer.pkl"
        prep_path      = self.models_dir / "preprocessing.pkl"
        feat_path      = self.models_dir / "feature_list.pkl"

        for p in [prep_path, feat_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required artefact not found: {p}\n"
                    "Run src/train.py first to generate model artefacts."
                )

        with open(prep_path,   "rb") as f: self._preprocessor = pickle.load(f)
        with open(feat_path,   "rb") as f: self._feature_list = pickle.load(f)
        
        self.ensemble = False
        if model_lr_path.exists() and model_xgb_path.exists() and model_lgb_path.exists():
            with open(model_lr_path, "rb") as f: self._model_lr = pickle.load(f)
            with open(model_xgb_path, "rb") as f: self._model_xgb = pickle.load(f)
            with open(model_lgb_path, "rb") as f: self._model_lgb = pickle.load(f)
            self.ensemble = True
            logger.info("Loaded ensemble models (LR, XGB, LGB).")
        else:
            model_path = self.models_dir / "model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Champion model not found at {model_path}")
            with open(model_path,  "rb") as f: self._model = pickle.load(f)
            logger.info("Loaded single champion model.")

        if knn_path.exists():
            with open(knn_path, "rb") as f: self._knn_transformer = pickle.load(f)
            logger.info("Loaded KNN neighborhood target mean transformer.")
        else:
            self._knn_transformer = None
        logger.info("Artefacts loaded from %s", self.models_dir)

    # ------------------------------------------------------------------
    def _classify_risk(self, probability: float) -> str:
        for level, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= probability < hi:
                return level
        return "HIGH"

    # ------------------------------------------------------------------
    def _identify_risk_factors(self, df_engineered: pd.DataFrame) -> List[str]:
        """
        Return up to 5 human-readable risk factors based on feature values.
        """
        factors: List[str] = []
        row = df_engineered.iloc[0]

        checks = [
            ("FIN_DEBT_TO_INCOME",          lambda v: v > 5,   "High debt-to-income ratio"),
            ("REPAY_BEHAVIOUR_SCORE",        lambda v: v > 0.5, "Poor repayment behaviour"),
            ("FIN_FINANCIAL_STRESS_INDEX",   lambda v: v > 0.6, "Elevated financial stress index"),
            ("CREDIT_DEFAULT_FLAG",          lambda v: v == 1,  "Previous default on record"),
            ("REPAY_LATE_PAYMENT_RATIO",     lambda v: v > 0.3, "High proportion of late payments"),
            ("FIN_ANNUITY_TO_INCOME",        lambda v: v > 0.5, "Monthly repayment burden is high"),
            ("CREDIT_UTILISATION",           lambda v: v > 0.8, "Credit utilisation near limit"),
            ("FIN_INCOME_STABILITY_SCORE",   lambda v: v < 0.3, "Low income stability"),
            ("EXT_SOURCE_2",                 lambda v: v < 0.3, "Low external credit score (source 2)"),
            ("EXT_SOURCE_3",                 lambda v: v < 0.3, "Low external credit score (source 3)"),
        ]

        for col, condition, label in checks:
            if col in df_engineered.columns:
                try:
                    if condition(row[col]):
                        factors.append(label)
                        if len(factors) == 5:
                            break
                except Exception:
                    pass

        return factors if factors else ["Insufficient data for detailed risk factors"]

    # ------------------------------------------------------------------
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict default risk for a single customer.

        Parameters
        ----------
        customer_data : dict of feature_name → value

        Returns
        -------
        dict with keys: risk_level, default_probability, recommendation, risk_factors
        """
        df = pd.DataFrame([customer_data])
        return self._run_prediction(df)[0]

    # ------------------------------------------------------------------
    def predict_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Predict default risk for a DataFrame of customers.

        Parameters
        ----------
        df : DataFrame where each row is one customer

        Returns
        -------
        list of result dicts (same format as predict_single)
        """
        return self._run_prediction(df)

    # ------------------------------------------------------------------
    def _run_prediction(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        # Preprocessing (transform only – do not re-fit)
        df_clean = self._preprocessor.transform(df)

        # Feature engineering
        from feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        df_fe = fe.transform(df_clean)

        # Compute neighbor target mean feature if transformer is present
        if self._knn_transformer is not None:
            df_fe['NEIGHBOR_TARGET_MEAN'] = self._knn_transformer.transform(df_fe, is_train=False)

        # Align to training feature list
        X = df_fe.reindex(columns=self._feature_list, fill_value=0)

        if self.ensemble:
            probs_lr = self._model_lr.predict_proba(X)[:, 1]
            probs_xgb = self._model_xgb.predict_proba(X)[:, 1]
            probs_lgb = self._model_lgb.predict_proba(X)[:, 1]
            
            # Weighted average based on metadata
            weights = self._meta.get("ensemble_weights", [0.1, 0.4, 0.5])
            probs = weights[0] * probs_lr + weights[1] * probs_xgb + weights[2] * probs_lgb
        else:
            probs = self._model.predict_proba(X)[:, 1]

        results = []
        for i, prob in enumerate(probs):
            risk_level = self._classify_risk(float(prob))
            row_df = df_fe.iloc[[i]]

            result = {
                "risk_level":         risk_level,
                "default_probability": round(float(prob), 4),
                "recommendation":     RISK_RECOMMENDATIONS[risk_level],
                "risk_factors":       self._identify_risk_factors(row_df),
            }
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustFin loan default risk prediction.")
    parser.add_argument("--input",      type=Path, required=True,  help="Input JSON or CSV file")
    parser.add_argument("--output",     type=Path, default=None,   help="Output JSON path (optional)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--threshold",  type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predictor = LoanRiskPredictor(models_dir=args.models_dir, threshold=args.threshold)

    if args.input.suffix == ".json":
        with open(args.input) as f:
            data = json.load(f)
        if isinstance(data, list):
            df_input = pd.DataFrame(data)
        else:
            df_input = pd.DataFrame([data])
    else:
        df_input = pd.read_csv(args.input)

    predictions = predictor.predict_batch(df_input)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(predictions, f, indent=2)
        logger.info("Saved predictions → %s", args.output)
    else:
        print(json.dumps(predictions, indent=2))