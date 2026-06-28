"""
feature_engineering.py
-----------------------
Reusable feature-engineering module for TrustFin Bank Loan Default Risk project.

Feature families
----------------
1. Credit features        – bureau / external credit bureau data
2. Repayment features     – instalment-payment behaviour
3. Financial features     – income, debt, financial stress

All public functions accept a DataFrame and return a DataFrame with
new columns added.  A FeatureEngineer class wraps them all for
pipeline use.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============================================================
# 1. CREDIT FEATURES
# ============================================================

def build_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer credit-bureau-derived features.

    New columns
    -----------
    CREDIT_ACTIVE_LOAN_RATIO   – fraction of bureau loans that are still active
    CREDIT_HISTORY_YEARS       – length of credit history in years
    CREDIT_DEFAULT_FLAG        – 1 if any bureau overdue amount exists
    CREDIT_UTILISATION         – total debt / total credit amount
    CREDIT_RISK_SCORE          – composite credit risk score (0–1, higher = riskier)

    Parameters
    ----------
    df : pd.DataFrame  (post-merge, must contain BUREAU_* columns)

    Returns
    -------
    pd.DataFrame  with new columns appended
    """
    df = df.copy()

    # Active-loan ratio
    if "BUREAU_ACTIVE_LOANS" in df.columns and "BUREAU_LOAN_COUNT" in df.columns:
        df["CREDIT_ACTIVE_LOAN_RATIO"] = np.where(
            df["BUREAU_LOAN_COUNT"] > 0,
            df["BUREAU_ACTIVE_LOANS"] / df["BUREAU_LOAN_COUNT"].replace(0, np.nan),
            0,
        )
        logger.info("Created CREDIT_ACTIVE_LOAN_RATIO")

    # Credit history length (days → years; DAYS_CREDIT is negative)
    if "BUREAU_AVG_DAYS_CREDIT" in df.columns:
        df["CREDIT_HISTORY_YEARS"] = np.abs(df["BUREAU_AVG_DAYS_CREDIT"]) / 365
        logger.info("Created CREDIT_HISTORY_YEARS")

    # Previous default flag
    if "BUREAU_TOTAL_OVERDUE" in df.columns:
        df["CREDIT_DEFAULT_FLAG"] = (df["BUREAU_TOTAL_OVERDUE"] > 0).astype(int)
        logger.info("Created CREDIT_DEFAULT_FLAG")

    # Credit utilisation
    if "BUREAU_TOTAL_DEBT" in df.columns and "BUREAU_AVG_CREDIT_AMT" in df.columns:
        total_credit = df["BUREAU_AVG_CREDIT_AMT"] * df.get("BUREAU_LOAN_COUNT", 1)
        df["CREDIT_UTILISATION"] = np.where(
            total_credit > 0,
            df["BUREAU_TOTAL_DEBT"] / total_credit.replace(0, np.nan),
            0,
        ).clip(0, 1)
        logger.info("Created CREDIT_UTILISATION")

    # Composite credit risk score (normalised weighted sum)
    components = []
    if "CREDIT_ACTIVE_LOAN_RATIO" in df.columns:
        components.append(df["CREDIT_ACTIVE_LOAN_RATIO"] * 0.3)
    if "CREDIT_DEFAULT_FLAG" in df.columns:
        components.append(df["CREDIT_DEFAULT_FLAG"] * 0.4)
    if "CREDIT_UTILISATION" in df.columns:
        components.append(df["CREDIT_UTILISATION"] * 0.3)

    if components:
        df["CREDIT_RISK_SCORE"] = sum(components)
        logger.info("Created CREDIT_RISK_SCORE")

    return df


# ============================================================
# 2. REPAYMENT FEATURES
# ============================================================

def build_repayment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer repayment-behaviour features from instalment-payment data.

    New columns
    -----------
    REPAY_LATE_PAYMENT_RATIO     – fraction of instalments paid late
    REPAY_AVG_DAYS_LATE          – average days late (0 if no history)
    REPAY_BEHAVIOUR_SCORE        – composite score (0–1, higher = worse)
    REPAY_PAYMENT_DEFICIT_RATIO  – missed payments / total instalments

    Parameters
    ----------
    df : pd.DataFrame  (must contain INSTALL_* columns)

    Returns
    -------
    pd.DataFrame  with new columns appended
    """
    df = df.copy()

    # Late-payment ratio
    if "INSTALL_MISSED_PAYMENTS" in df.columns and "INSTALL_COUNT" in df.columns:
        df["REPAY_LATE_PAYMENT_RATIO"] = np.where(
            df["INSTALL_COUNT"] > 0,
            df["INSTALL_MISSED_PAYMENTS"] / df["INSTALL_COUNT"].replace(0, np.nan),
            0,
        ).clip(0, 1)
        logger.info("Created REPAY_LATE_PAYMENT_RATIO")

    # Average days late (clip negatives → 0 meaning paid early)
    if "INSTALL_AVG_LATE_DAYS" in df.columns:
        df["REPAY_AVG_DAYS_LATE"] = df["INSTALL_AVG_LATE_DAYS"].clip(lower=0).fillna(0)
        logger.info("Created REPAY_AVG_DAYS_LATE")

    # Payment deficit ratio
    if "INSTALL_AVG_PAYMENT_DIFF" in df.columns and "INSTALL_COUNT" in df.columns:
        df["REPAY_PAYMENT_DEFICIT_RATIO"] = np.where(
            df["INSTALL_AVG_PAYMENT_DIFF"] > 0, 1, 0
        )
        logger.info("Created REPAY_PAYMENT_DEFICIT_RATIO")

    # POS DPD contribution
    pos_component = pd.Series(np.zeros(len(df)), index=df.index)
    if "POS_AVG_DPD" in df.columns:
        pos_component = (df["POS_AVG_DPD"].fillna(0) / 30).clip(0, 1)

    # Composite repayment behaviour score (0 = perfect, 1 = terrible)
    components = []
    if "REPAY_LATE_PAYMENT_RATIO" in df.columns:
        components.append(df["REPAY_LATE_PAYMENT_RATIO"] * 0.4)
    if "REPAY_AVG_DAYS_LATE" in df.columns:
        normalised_days = (df["REPAY_AVG_DAYS_LATE"] / 90).clip(0, 1)
        components.append(normalised_days * 0.4)
    components.append(pos_component * 0.2)

    if components:
        df["REPAY_BEHAVIOUR_SCORE"] = sum(components).clip(0, 1)
        logger.info("Created REPAY_BEHAVIOUR_SCORE")

    return df


# ============================================================
# 3. FINANCIAL FEATURES
# ============================================================

def build_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer financial-health features from application-level data.

    New columns
    -----------
    FIN_DEBT_TO_INCOME         – AMT_CREDIT / AMT_INCOME_TOTAL
    FIN_ANNUITY_TO_INCOME      – AMT_ANNUITY / AMT_INCOME_TOTAL
    FIN_INCOME_STABILITY_SCORE – proxy for employment / income stability
    FIN_FINANCIAL_STRESS_INDEX – composite stress score (0–1, higher = more stressed)
    FIN_CREDIT_TO_GOODS        – AMT_CREDIT / AMT_GOODS_PRICE

    Parameters
    ----------
    df : pd.DataFrame  (must contain AMT_INCOME_TOTAL, AMT_CREDIT, etc.)

    Returns
    -------
    pd.DataFrame  with new columns appended
    """
    df = df.copy()
    eps = 1e-6  # small constant to avoid division by zero

    # Debt-to-income ratio
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["FIN_DEBT_TO_INCOME"] = (
            df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
        ).clip(0, 50)
        logger.info("Created FIN_DEBT_TO_INCOME")

    # Annuity-to-income ratio (monthly burden)
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        monthly_income = df["AMT_INCOME_TOTAL"] / 12
        df["FIN_ANNUITY_TO_INCOME"] = (
            df["AMT_ANNUITY"] / (monthly_income + eps)
        ).clip(0, 5)
        logger.info("Created FIN_ANNUITY_TO_INCOME")

    # Credit-to-goods ratio
    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        df["FIN_CREDIT_TO_GOODS"] = (
            df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + eps)
        ).clip(0, 5)
        logger.info("Created FIN_CREDIT_TO_GOODS")

    # Income stability score
    # Proxy: longer employment + higher income = more stable (0–1, higher = more stable)
    stability_parts = []

    if "DAYS_EMPLOYED" in df.columns:
        # DAYS_EMPLOYED is negative (days before application); larger absolute = more stable
        emp_years = (np.abs(df["DAYS_EMPLOYED"].fillna(0)) / 365).clip(0, 20)
        stability_parts.append((emp_years / 20) * 0.5)   # normalise to 0–0.5

    if "AMT_INCOME_TOTAL" in df.columns:
        # Percentile-rank income → 0–0.5
        income_rank = df["AMT_INCOME_TOTAL"].rank(pct=True) * 0.5
        stability_parts.append(income_rank)

    if stability_parts:
        df["FIN_INCOME_STABILITY_SCORE"] = sum(stability_parts).clip(0, 1)
        logger.info("Created FIN_INCOME_STABILITY_SCORE")

    # Financial stress index
    # Combines debt burden + repayment strain + credit risk
    stress_parts = []
    if "FIN_DEBT_TO_INCOME" in df.columns:
        stress_parts.append((df["FIN_DEBT_TO_INCOME"] / 10).clip(0, 1) * 0.4)
    if "FIN_ANNUITY_TO_INCOME" in df.columns:
        stress_parts.append((df["FIN_ANNUITY_TO_INCOME"] / 2).clip(0, 1) * 0.3)
    if "REPAY_BEHAVIOUR_SCORE" in df.columns:
        stress_parts.append(df["REPAY_BEHAVIOUR_SCORE"] * 0.3)

    if stress_parts:
        df["FIN_FINANCIAL_STRESS_INDEX"] = sum(stress_parts).clip(0, 1)
        logger.info("Created FIN_FINANCIAL_STRESS_INDEX")

    return df


# ============================================================
# Pipeline class
# ============================================================

class FeatureEngineer:
    """
    Orchestrates all feature-engineering steps.

    Usage
    -----
    >>> fe = FeatureEngineer()
    >>> df_enriched = fe.transform(df_preprocessed)

    The same instance can be applied to both train and test sets; the
    feature-engineering transformations are stateless (no fitting required).
    """

    def __init__(self, build_credit: bool = True,
                 build_repayment: bool = True,
                 build_financial: bool = True):
        self.build_credit = build_credit
        self.build_repayment = build_repayment
        self.build_financial = build_financial

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all enabled feature-engineering steps and return enriched DataFrame."""
        logger.info("=== FeatureEngineer: transform ===  Input shape: %s", df.shape)

        if self.build_credit:
            df = build_credit_features(df)

        if self.build_repayment:
            df = build_repayment_features(df)

        if self.build_financial:
            df = build_financial_features(df)

        # Ensure all engineered columns are finite (replace inf → NaN → 0)
        df = df.replace([np.inf, -np.inf], np.nan)
        eng_cols = [c for c in df.columns if c.startswith(("CREDIT_", "REPAY_", "FIN_"))]
        df[eng_cols] = df[eng_cols].fillna(0)

        logger.info("FeatureEngineer transform complete. Output shape: %s", df.shape)
        return df

    # Alias: sklearn-style fit_transform (no fitting needed here)
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)