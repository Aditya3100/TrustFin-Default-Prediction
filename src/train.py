"""
train.py
--------
Training system for TrustFin Bank Loan Default Risk project.

Workflow
--------
1. Load and merge all data sources.
2. Run preprocessing pipeline.
3. Run feature engineering.
4. Split train / test (stratified).
5. Train baseline (Logistic Regression) + advanced models (XGBoost, LightGBM).
6. Track all experiments with MLflow.
7. Persist best model artefacts to models/.

Usage
-----
    python src/train.py --data-dir data/ --models-dir models/

Environment
-----------
Set MLFLOW_TRACKING_URI to point at a remote server, or leave it as the
default local ./mlruns directory.
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier

from preprocessing import PreprocessingPipeline
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
TEST_SIZE = 0.20
RANDOM_STATE = 42
MODEL_DIR = Path("models")
MLFLOW_EXPERIMENT = "TrustFin-LoanDefault"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge all Home Credit data files.

    Expected files (subset is fine — missing files are skipped with a warning)
    ----------
    application_train.csv   – main application table (required)
    bureau.csv              – credit bureau records
    bureau_balance.csv      – monthly bureau balances
    previous_application.csv
    POS_CASH_balance.csv
    installments_payments.csv
    credit_card_balance.csv

    Returns
    -------
    pd.DataFrame  merged dataset
    """
    # Check if a pre-merged dataset exists
    premerged_path = data_dir / "final_df.csv.xls"
    if premerged_path.exists():
        logger.info("Loading pre-merged final_df.csv.xls …")
        df = pd.read_csv(premerged_path)
        logger.info("final_df.csv.xls shape: %s", df.shape)
        return df

    app_path = data_dir / "application_train.csv"
    if not app_path.exists():
        raise FileNotFoundError(f"Required file not found: {app_path} (or final_df.csv.xls)")

    logger.info("Loading application_train …")
    df = pd.read_csv(app_path)
    logger.info("application_train shape: %s", df.shape)

    # ---------- bureau -------------------------------------------------------
    bureau_path = data_dir / "bureau.csv"
    if bureau_path.exists():
        bureau = pd.read_csv(bureau_path)
        bureau_agg = bureau.groupby(ID_COL).agg(
            BUREAU_LOAN_COUNT=(     "SK_ID_BUREAU", "count"),
            BUREAU_ACTIVE_LOANS=(   "CREDIT_ACTIVE",   lambda x: (x == "Active").sum()),
            BUREAU_CLOSED_LOANS=(   "CREDIT_ACTIVE",   lambda x: (x == "Closed").sum()),
            BUREAU_AVG_CREDIT_AMT=( "AMT_CREDIT_SUM",  "mean"),
            BUREAU_TOTAL_DEBT=(     "AMT_CREDIT_SUM_DEBT", "sum"),
            BUREAU_AVG_DAYS_CREDIT=("DAYS_CREDIT",     "mean"),
            BUREAU_MAX_OVERDUE=(    "AMT_CREDIT_MAX_OVERDUE", "max"),
            BUREAU_TOTAL_OVERDUE=(  "AMT_CREDIT_SUM_OVERDUE", "sum"),
        ).reset_index()

        # bureau_balance
        bb_path = data_dir / "bureau_balance.csv"
        if bb_path.exists():
            bb = pd.read_csv(bb_path)
            bb_agg = bb.groupby("SK_ID_BUREAU").agg(
                BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
                BB_DPD_MEAN=(   "STATUS", lambda x: x.isin(["1","2","3","4","5"]).mean()),
            ).reset_index()
            bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
            bureau_agg2 = bureau.groupby(ID_COL).agg(
                BB_DPD_MEAN=("BB_DPD_MEAN", "mean"),
            ).reset_index()
            bureau_agg = bureau_agg.merge(bureau_agg2, on=ID_COL, how="left")

        df = df.merge(bureau_agg, on=ID_COL, how="left")
        logger.info("Merged bureau data.")

    # ---------- previous_application -----------------------------------------
    prev_path = data_dir / "previous_application.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        prev_agg = prev.groupby(ID_COL).agg(
            PREV_APP_COUNT=(       "SK_ID_PREV", "count"),
            PREV_APPROVED_COUNT=(  "NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
            PREV_REFUSED_COUNT=(   "NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
            PREV_AVG_CREDIT=(      "AMT_CREDIT", "mean"),
            PREV_AVG_ANNUITY=(     "AMT_ANNUITY", "mean"),
            PREV_AVG_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "mean"),
            PREV_APPROVAL_RATE=(   "NAME_CONTRACT_STATUS", lambda x: (x == "Approved").mean()),
        ).reset_index()

        # Add lag features
        from feature_engineering import compute_prev_application_lag_features
        lag_features = compute_prev_application_lag_features(prev)
        if not lag_features.empty:
            prev_agg = prev_agg.merge(lag_features, on=ID_COL, how="left")

        df = df.merge(prev_agg, on=ID_COL, how="left")
        logger.info("Merged previous_application data (with lag features).")

    # ---------- POS_CASH_balance ---------------------------------------------
    pos_path = data_dir / "POS_CASH_balance.csv"
    if pos_path.exists():
        pos = pd.read_csv(pos_path)
        pos_agg = pos.groupby(ID_COL).agg(
            POS_MONTHS_COUNT=(   "MONTHS_BALANCE", "count"),
            POS_COMPLETED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Completed").sum()),
            POS_AVG_DPD=(        "SK_DPD", "mean"),
            POS_MAX_DPD=(        "SK_DPD", "max"),
        ).reset_index()
        df = df.merge(pos_agg, on=ID_COL, how="left")
        logger.info("Merged POS_CASH_balance data.")

    # ---------- installments_payments ----------------------------------------
    inst_path = data_dir / "installments_payments.csv"
    if inst_path.exists():
        inst = pd.read_csv(inst_path)
        inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
        inst["DAYS_LATE"]    = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
        inst_agg = inst.groupby(ID_COL).agg(
            INSTALL_COUNT=(            "SK_ID_PREV", "count"),
            INSTALL_AVG_LATE_DAYS=(    "DAYS_LATE",     "mean"),
            INSTALL_MAX_LATE_DAYS=(    "DAYS_LATE",     "max"),
            INSTALL_MISSED_PAYMENTS=(  "PAYMENT_DIFF",  lambda x: (x > 0).sum()),
            INSTALL_AVG_PAYMENT_DIFF=( "PAYMENT_DIFF",  "mean"),
            INSTALL_MAX_AMT_INSTALMENT=("AMT_INSTALMENT", "max"),
        ).reset_index()

        # Add temporal features
        from feature_engineering import compute_installment_temporal_features
        temp_features = compute_installment_temporal_features(inst)
        if not temp_features.empty:
            inst_agg = inst_agg.merge(temp_features, on=ID_COL, how="left")

        df = df.merge(inst_agg, on=ID_COL, how="left")
        logger.info("Merged installments_payments data (with temporal and max installment).")

    # ---------- credit_card_balance ------------------------------------------
    cc_path = data_dir / "credit_card_balance.csv"
    if cc_path.exists():
        cc = pd.read_csv(cc_path)
        cc_agg = cc.groupby(ID_COL).agg(
            CC_MONTHS_COUNT=("MONTHS_BALANCE",         "count"),
            CC_AVG_BALANCE=( "AMT_BALANCE",            "mean"),
            CC_MAX_BALANCE=( "AMT_BALANCE",            "max"),
            CC_AVG_DRAWINGS=("AMT_DRAWINGS_CURRENT",   "mean"),
            CC_AVG_DPD=(     "SK_DPD",                 "mean"),
        ).reset_index()
        df = df.merge(cc_agg, on=ID_COL, how="left")
        logger.info("Merged credit_card_balance data.")

    logger.info("Final merged shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_artefact(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved → %s", path)


def compute_scale_pos_weight(y: pd.Series) -> float:
    counts = y.value_counts()
    return float(counts[0] / counts[1])


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    logger.info("Training Logistic Regression …")
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
            n_jobs=-1,
        )
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  scale_pos_weight: float) -> XGBClassifier:
    logger.info("Training XGBoost …")
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not installed – skipping.")
        return None

    logger.info("Training LightGBM …")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(data_dir: Path, models_dir: Path) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # 1. Load data
    df = load_data(data_dir)

    # 2. Preprocess
    pipe = PreprocessingPipeline()
    df_clean = pipe.fit_transform(df)

    # 3. Feature engineering
    fe = FeatureEngineer()
    df_fe = fe.fit_transform(df_clean)

    # 4. Split
    drop_cols = [c for c in [TARGET_COL, ID_COL] if c in df_fe.columns]
    X = df_fe.drop(columns=drop_cols)
    y = df_fe[TARGET_COL]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Initial Train: %s  Test: %s", X_train_full.shape, X_test.shape)

    # Feature Selection using a quick LightGBM
    import lightgbm as lgb
    logger.info("Running LightGBM feature selection ...")
    selector_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=RANDOM_STATE, verbose=-1)
    selector_model.fit(X_train_full, y_train_full)
    
    importance_df = pd.DataFrame({
        'feature': X_train_full.columns,
        'importance': selector_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    meaningful_features = importance_df[importance_df['importance'] >= 2]['feature'].tolist()
    logger.info("Selected %d features out of %d using LightGBM importance >= 2", len(meaningful_features), len(X_train_full.columns))
    
    X_train_full = X_train_full[meaningful_features]
    X_test = X_test[meaningful_features]
    feature_list = meaningful_features.copy()

    # Fit final KNN Target Mean Transformer on full train set
    from feature_engineering import NeighborTargetMeanTransformer
    logger.info("Fitting final KNN target mean transformer ...")
    knn_transformer = NeighborTargetMeanTransformer(n_neighbors=500)
    knn_transformer.fit(X_train_full, y_train_full)

    # Compute target mean feature for test set (is_train=False) and full train set (is_train=True)
    X_test = X_test.copy()
    X_test['NEIGHBOR_TARGET_MEAN'] = knn_transformer.transform(X_test, is_train=False)

    X_train_full_final = X_train_full.copy()
    X_train_full_final['NEIGHBOR_TARGET_MEAN'] = knn_transformer.transform(X_train_full, is_train=True)

    feature_list.append('NEIGHBOR_TARGET_MEAN')
    logger.info("Added NEIGHBOR_TARGET_MEAN feature.")

    # 5. Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds_lr = np.zeros(len(X_train_full))
    oof_preds_xgb = np.zeros(len(X_train_full))
    oof_preds_lgb = np.zeros(len(X_train_full))
    
    logger.info("Starting Stratified 5-Fold Cross Validation ...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        logger.info("Fold %d ...", fold + 1)
        X_tr, X_val = X_train_full.iloc[train_idx].copy(), X_train_full.iloc[val_idx].copy()
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        # Fit fold KNN target mean
        fold_knn = NeighborTargetMeanTransformer(n_neighbors=500)
        fold_knn.fit(X_tr, y_tr)
        
        X_tr['NEIGHBOR_TARGET_MEAN'] = fold_knn.transform(X_tr, is_train=True)
        X_val['NEIGHBOR_TARGET_MEAN'] = fold_knn.transform(X_val, is_train=False)
        
        # 1. LR
        lr_fold = train_logistic_regression(X_tr, y_tr)
        oof_preds_lr[val_idx] = lr_fold.predict_proba(X_val)[:, 1]
        
        # 2. XGBoost
        scale_pos_wt_fold = compute_scale_pos_weight(y_tr)
        xgb_fold = train_xgboost(X_tr, y_tr, scale_pos_wt_fold)
        oof_preds_xgb[val_idx] = xgb_fold.predict_proba(X_val)[:, 1]
        
        # 3. LightGBM
        lgb_fold = train_lightgbm(X_tr, y_tr)
        if lgb_fold is not None:
            oof_preds_lgb[val_idx] = lgb_fold.predict_proba(X_val)[:, 1]

    auc_lr = roc_auc_score(y_train_full, oof_preds_lr)
    auc_xgb = roc_auc_score(y_train_full, oof_preds_xgb)
    auc_lgb = roc_auc_score(y_train_full, oof_preds_lgb)
    
    logger.info("OOF ROC-AUC Scores -- LR: %.4f, XGB: %.4f, LGB: %.4f", auc_lr, auc_xgb, auc_lgb)

    # Blend OOF predictions
    w_lr, w_xgb, w_lgb = 0.1, 0.4, 0.5
    oof_preds_blended = w_lr * oof_preds_lr + w_xgb * oof_preds_xgb + w_lgb * oof_preds_lgb
    auc_blended = roc_auc_score(y_train_full, oof_preds_blended)
    logger.info("Blended OOF ROC-AUC Score: %.4f", auc_blended)

    # Threshold Optimization based on Blended OOF F1-macro score
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.1, 0.9, 0.01)
    for t in thresholds:
        preds = (oof_preds_blended >= t).astype(int)
        score = f1_score(y_train_full, preds, average='macro')
        if score > best_f1:
            best_f1 = score
            best_threshold = t
    logger.info("Optimal threshold on OOF: %.2f, F1-macro: %.4f", best_threshold, best_f1)

    # Train final models on full train set
    logger.info("Fitting final models on the entire training set ...")
    final_lr = train_logistic_regression(X_train_full_final, y_train_full)
    final_xgb = train_xgboost(X_train_full_final, y_train_full, compute_scale_pos_weight(y_train_full))
    final_lgb = train_lightgbm(X_train_full_final, y_train_full)

    # Blend test set predictions
    probs_lr = final_lr.predict_proba(X_test)[:, 1]
    probs_xgb = final_xgb.predict_proba(X_test)[:, 1]
    probs_lgb = final_lgb.predict_proba(X_test)[:, 1]
    
    y_prob = w_lr * probs_lr + w_xgb * probs_xgb + w_lgb * probs_lgb
    auc = roc_auc_score(y_test, y_prob)
    logger.info("Ensembled Test ROC-AUC: %.4f", auc)

    # Log to MLflow
    with mlflow.start_run(run_name="ensemble"):
        mlflow.log_params({
            "model_type": "ensemble_blend",
            "train_size": len(X_train_full_final),
            "test_size": len(X_test),
            "n_features": len(feature_list),
            "lr_weight": w_lr,
            "xgb_weight": w_xgb,
            "lgb_weight": w_lgb,
            "optimal_threshold": round(best_threshold, 2),
        })
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("oof_roc_auc", auc_blended)
        mlflow.sklearn.log_model(final_lgb, artifact_path="model", serialization_format="cloudpickle")

    # 6. Save models and preprocessing pipeline
    models_dir.mkdir(parents=True, exist_ok=True)
    save_artefact(final_lgb, models_dir / "model.pkl")  # production champion fallback
    save_artefact(final_lr,  models_dir / "model_lr.pkl")
    save_artefact(final_xgb, models_dir / "model_xgb.pkl")
    save_artefact(final_lgb, models_dir / "model_lgb.pkl")
    save_artefact(knn_transformer, models_dir / "knn_transformer.pkl")
    save_artefact(pipe,        models_dir / "preprocessing.pkl")
    save_artefact(feature_list,models_dir / "feature_list.pkl")

    # Save test set for evaluation
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    if ID_COL in df_fe.columns:
        test_df[ID_COL] = df_fe.loc[X_test.index, ID_COL]
    test_csv_path = data_dir / "processed_test.csv"
    test_df.to_csv(test_csv_path, index=False)
    logger.info("Saved test set for evaluation → %s", test_csv_path)

    # Save a metadata JSON for the prediction service
    meta = {
        "best_model": "ensemble",
        "roc_auc": auc,
        "n_features": len(feature_list),
        "optimal_threshold": round(best_threshold, 2),
        "ensemble_weights": [w_lr, w_xgb, w_lgb],
        "all_results": {
            "logistic_regression": {"auc": auc_lr},
            "xgboost": {"auc": auc_xgb},
            "lightgbm": {"auc": auc_lgb},
            "ensemble": {"auc": auc}
        }
    }
    with open(models_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training complete. Artefacts saved to %s", models_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TrustFin loan-default models.")
    parser.add_argument("--data-dir",   type=Path, default=Path("data"),   help="Path to data directory")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Path to models directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training(data_dir=args.data_dir, models_dir=args.models_dir)