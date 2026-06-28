"""
preprocessing.py
----------------
Reusable preprocessing pipeline for TrustFin Bank Loan Default Risk project.

Handles:
- Missing value imputation (strategy-aware)
- Categorical encoding (LabelEncoder with persistence)
- Numerical scaling (StandardScaler / MinMaxScaler)
- Duplicate detection and removal
- Inconsistent / sentinel value handling
- Feature-union style pipeline via PreprocessingPipeline class
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MERGED_PREFIXES = ["BUREAU_", "PREV_", "POS_", "INSTALL_", "CC_", "BB_"]

SENTINEL_MAP: Dict[str, float] = {
    "DAYS_EMPLOYED": 365243,   # placeholder used when applicant has no employment
}

# EXT_SOURCE columns carry strong predictive signal; leave them NaN so
# tree-based models can exploit missingness rather than flatten it.
EXT_SOURCE_PATTERN = "EXT_SOURCE"

HIGH_MISSING_THRESHOLD = 0.80  # drop columns with >80 % missing


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------

def detect_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Detect and drop duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
    subset : list of column names to consider for identifying duplicates.
              If None, all columns are used.

    Returns
    -------
    pd.DataFrame  deduplicated copy
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d duplicate row(s).", n_dropped)
    return df.reset_index(drop=True)


def handle_sentinel_values(df: pd.DataFrame,
                            sentinel_map: Dict[str, float] = SENTINEL_MAP) -> pd.DataFrame:
    """
    Replace known sentinel / placeholder values with NaN.

    For each (column, sentinel_value) pair the sentinel is replaced with NaN
    and a binary flag column ``<col>_ANOM`` is added (1 = was sentinel).

    Parameters
    ----------
    df : pd.DataFrame
    sentinel_map : dict  {column_name: sentinel_value}

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col, val in sentinel_map.items():
        if col in df.columns:
            flag_col = f"{col}_ANOM"
            df[flag_col] = (df[col] == val).astype(int)
            df[col] = df[col].replace(val, np.nan)
            logger.info("Replaced sentinel %s in '%s'; flag column '%s' added.", val, col, flag_col)
    return df


def drop_high_missing(df: pd.DataFrame,
                       threshold: float = HIGH_MISSING_THRESHOLD) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns whose missing-value fraction exceeds *threshold*.

    Parameters
    ----------
    df        : pd.DataFrame
    threshold : float  (0–1)  e.g. 0.80 drops columns with >80 % NaN

    Returns
    -------
    (cleaned_df, list_of_dropped_columns)
    """
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac >= threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    if drop_cols:
        logger.info("Dropped %d high-missing columns (>= %.0f%%).", len(drop_cols), threshold * 100)
    return df, drop_cols


def add_history_flags(df: pd.DataFrame,
                       prefixes: List[str] = MERGED_PREFIXES) -> pd.DataFrame:
    """
    Add binary ``HAS_<PREFIX>HISTORY`` flags *before* zero-filling merged tables.

    After zero-filling it is impossible to distinguish "no history at all"
    from a genuine zero aggregation value.

    Parameters
    ----------
    df       : pd.DataFrame
    prefixes : list of column-name prefixes that come from merged tables

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            flag = f"HAS_{prefix}HISTORY"
            df[flag] = df[cols[0]].notna().astype(int)
    return df


def impute_merged_columns(df: pd.DataFrame,
                           prefixes: List[str] = MERGED_PREFIXES) -> pd.DataFrame:
    """Fill aggregated / merged columns with 0 (absence = 0 activity)."""
    merged_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    df[merged_cols] = df[merged_cols].fillna(0)
    return df


def impute_numeric_columns(df: pd.DataFrame,
                            exclude_patterns: Optional[List[str]] = None,
                            train_medians: Optional[pd.Series] = None,
                            prefixes: List[str] = MERGED_PREFIXES
                            ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fill original numeric columns with the column median.

    Columns matching *exclude_patterns* (e.g. EXT_SOURCE) are intentionally
    left NaN so that tree models can exploit the missingness signal.

    Parameters
    ----------
    df              : pd.DataFrame
    exclude_patterns: list of column-name substrings to skip
    train_medians   : pre-computed medians from the training set (pass during
                      test-set preprocessing to avoid data leakage)
    prefixes        : merged-table prefixes already handled elsewhere

    Returns
    -------
    (imputed_df, medians_series)
    """
    exclude_patterns = exclude_patterns or [EXT_SOURCE_PATTERN]
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    merged_cols = [c for c in num_cols if any(c.startswith(p) for p in prefixes)]
    target_cols = [
        c for c in num_cols
        if c not in merged_cols
        and not any(pat in c for pat in exclude_patterns)
    ]

    if train_medians is None:
        train_medians = df[target_cols].median()

    df[target_cols] = df[target_cols].fillna(train_medians)
    return df, train_medians


def impute_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill categorical (object) columns with the string ``'Unknown'``."""
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    return df


def encode_categoricals(df: pd.DataFrame,
                         encoders: Optional[Dict[str, LabelEncoder]] = None
                         ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode all object / string columns.

    Parameters
    ----------
    df       : pd.DataFrame
    encoders : dict of pre-fitted LabelEncoders (pass during test-set encoding)

    Returns
    -------
    (encoded_df, encoders_dict)
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = encoders or {}

    for col in cat_cols:
        le = encoders.get(col, LabelEncoder())
        if col not in encoders:
            le.fit(df[col].astype(str))
            encoders[col] = le
        # Handle unseen labels during inference
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
        df[col] = le.transform(df[col])

    return df, encoders


def scale_numeric_features(df: pd.DataFrame,
                             exclude_cols: Optional[List[str]] = None,
                             scaler: Optional[StandardScaler] = None
                             ) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standard-scale numeric columns (mean=0, std=1).

    Parameters
    ----------
    df           : pd.DataFrame  (post-encoding, all numeric)
    exclude_cols : columns to leave unscaled (IDs, binary flags, target)
    scaler       : pre-fitted scaler for test-set transformation

    Returns
    -------
    (scaled_df, scaler)
    """
    df = df.copy()
    exclude_cols = set(exclude_cols or [])
    num_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in exclude_cols
    ]

    scaler = scaler or StandardScaler()
    if not hasattr(scaler, "mean_"):
        scaler.fit(df[num_cols])

    df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for Home Credit Default Risk data.

    Usage
    -----
    >>> pipe = PreprocessingPipeline()
    >>> X_train_clean = pipe.fit_transform(df_train)
    >>> X_test_clean  = pipe.transform(df_test)

    Attributes (populated after fit_transform)
    ------------------------------------------
    encoders_      : dict[str, LabelEncoder]
    train_medians_ : pd.Series
    scaler_        : StandardScaler
    dropped_cols_  : list[str]
    """

    def __init__(self,
                 sentinel_map: Dict[str, float] = SENTINEL_MAP,
                 high_missing_threshold: float = HIGH_MISSING_THRESHOLD,
                 merged_prefixes: List[str] = MERGED_PREFIXES,
                 scale: bool = False,
                 id_col: str = "SK_ID_CURR",
                 target_col: str = "TARGET"):
        self.sentinel_map = sentinel_map
        self.high_missing_threshold = high_missing_threshold
        self.merged_prefixes = merged_prefixes
        self.scale = scale
        self.id_col = id_col
        self.target_col = target_col

        # Fitted state
        self.encoders_: Dict[str, LabelEncoder] = {}
        self.train_medians_: Optional[pd.Series] = None
        self.scaler_: Optional[StandardScaler] = None
        self.dropped_cols_: List[str] = []

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* and return the cleaned DataFrame."""
        logger.info("=== PreprocessingPipeline: fit_transform ===")

        df = df.copy()
        df = df.dropna(axis=1, how="all")

        df, self.dropped_cols_ = drop_high_missing(df, self.high_missing_threshold)
        df = detect_duplicates(df)
        df = handle_sentinel_values(df, self.sentinel_map)
        df = add_history_flags(df, self.merged_prefixes)
        df = impute_merged_columns(df, self.merged_prefixes)
        df, self.train_medians_ = impute_numeric_columns(df, prefixes=self.merged_prefixes)
        df = impute_categorical_columns(df)
        df, self.encoders_ = encode_categoricals(df)

        if self.scale:
            exclude = [self.id_col, self.target_col]
            df, self.scaler_ = scale_numeric_features(df, exclude_cols=exclude)

        logger.info("fit_transform complete. Shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformations to a new DataFrame (e.g. test set)."""
        logger.info("=== PreprocessingPipeline: transform ===")

        df = df.copy()
        df = df.dropna(axis=1, how="all")
        df = df.drop(columns=[c for c in self.dropped_cols_ if c in df.columns], errors="ignore")
        df = handle_sentinel_values(df, self.sentinel_map)
        df = add_history_flags(df, self.merged_prefixes)
        df = impute_merged_columns(df, self.merged_prefixes)
        df, _ = impute_numeric_columns(df,
                                        train_medians=self.train_medians_,
                                        prefixes=self.merged_prefixes)
        df = impute_categorical_columns(df)
        df, _ = encode_categoricals(df, encoders=self.encoders_)

        if self.scale and self.scaler_ is not None:
            exclude = [self.id_col, self.target_col]
            df, _ = scale_numeric_features(df, exclude_cols=exclude, scaler=self.scaler_)

        logger.info("transform complete. Shape: %s", df.shape)
        return df