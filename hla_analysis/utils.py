"""
Shared utilities: logging, FDR correction, encoding helpers.
"""

import logging
import sys
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logger with console and optional file handler.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str, optional
        Path to log file. If None, logs only to stderr.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    root = logging.getLogger("hla_analysis")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of p-values. NaN values are preserved.
    alpha : float
        FDR significance threshold (not used in computation, just for reference).

    Returns
    -------
    np.ndarray
        Array of FDR-adjusted p-values (q-values).
    """
    pvals = np.asarray(pvalues, dtype=np.float64)
    n = pvals.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    # Handle NaN
    valid = ~np.isnan(pvals)
    n_valid = valid.sum()
    if n_valid == 0:
        return pvals.copy()

    qvals = np.full(n, np.nan, dtype=np.float64)
    valid_p = pvals[valid]

    # Sort
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    # BH adjustment
    rank = np.arange(1, n_valid + 1, dtype=np.float64)
    adjusted = sorted_p * n_valid / rank

    # Enforce monotonicity (from largest to smallest)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Unsort
    result = np.empty(n_valid, dtype=np.float64)
    result[order] = adjusted
    qvals[valid] = result

    return qvals


def encode_sex(series: pd.Series) -> pd.Series:
    """Encode sex: M=1, F=0. Handles string and numeric inputs.

    Parameters
    ----------
    series : pd.Series
        Sex column with values like 'M', 'F', 1, 0, 'male', 'female'.

    Returns
    -------
    pd.Series
        Numeric series with M=1, F=0, missing=NaN.
    """
    s = series.copy()
    if s.dtype == object or s.dtype.name == "category":
        mapping = {
            "M": 1, "m": 1, "male": 1, "Male": 1, "MALE": 1,
            "F": 0, "f": 0, "female": 0, "Female": 0, "FEMALE": 0,
        }
        s = s.map(mapping)
    return pd.to_numeric(s, errors="coerce")


def encode_grade(series: pd.Series) -> pd.Series:
    """Encode grade: HGG=1, LGG=0.

    Parameters
    ----------
    series : pd.Series
        Grade column with values like 'HGG', 'LGG'.

    Returns
    -------
    pd.Series
        Numeric series with HGG=1, LGG=0, missing=NaN.
    """
    s = series.copy()
    if s.dtype == object or s.dtype.name == "category":
        mapping = {
            "HGG": 1, "hgg": 1, "High": 1, "high": 1,
            "LGG": 0, "lgg": 0, "Low": 0, "low": 0,
        }
        s = s.map(mapping)
    return pd.to_numeric(s, errors="coerce")


def classify_feature(name: str) -> Optional[str]:
    """Classify an HLA feature by its name.

    Parameters
    ----------
    name : str
        Feature column name.

    Returns
    -------
    str or None
        One of 'classical_2digit', 'classical_4digit', 'amino_acid', or None.
    """
    if name.startswith("AA_"):
        return "amino_acid"
    elif name.startswith("HLA_"):
        if ":" in name:
            return "classical_4digit"
        else:
            return "classical_2digit"
    return None


def classify_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """Classify a list of features by type.

    Parameters
    ----------
    feature_names : list of str
        Feature column names.

    Returns
    -------
    dict
        Mapping from feature type to list of feature names.
    """
    classified = {
        "classical_2digit": [],
        "classical_4digit": [],
        "amino_acid": [],
    }
    for name in feature_names:
        ftype = classify_feature(name)
        if ftype is not None:
            classified[ftype].append(name)
    return classified


def extract_gene_from_feature(feature_name: str) -> Optional[str]:
    """Extract HLA gene name from a feature name.

    Parameters
    ----------
    feature_name : str
        Feature column name, e.g. 'HLA_A_01:01' or 'AA_DRB1_9_V'.

    Returns
    -------
    str or None
        Gene name (e.g., 'A', 'B', 'DRB1') or None.
    """
    if feature_name.startswith("HLA_"):
        parts = feature_name[4:].split("_")
        if parts:
            return parts[0]
    elif feature_name.startswith("AA_"):
        parts = feature_name[3:].split("_")
        if parts:
            return parts[0]
    return None


def compute_concordance(time: np.ndarray, event: np.ndarray, risk_score: np.ndarray) -> float:
    """Compute Harrell's concordance index.

    Parameters
    ----------
    time : np.ndarray
        Survival times.
    event : np.ndarray
        Event indicators (1=event, 0=censored).
    risk_score : np.ndarray
        Predicted risk scores (higher = more risk).

    Returns
    -------
    float
        Concordance index in [0, 1]. Returns 0.5 if no valid pairs.
    """
    n = len(time)
    concordant = 0
    discordant = 0
    tied_risk = 0

    event_idx = np.where(event == 1)[0]

    for i in event_idx:
        # Compare with all individuals who survived longer
        for j in range(n):
            if time[j] > time[i]:
                if risk_score[i] > risk_score[j]:
                    concordant += 1
                elif risk_score[i] < risk_score[j]:
                    discordant += 1
                else:
                    tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


def safe_exp(x: float, max_val: float = 500.0) -> float:
    """Safe exponentiation to avoid overflow.

    Parameters
    ----------
    x : float
        Exponent.
    max_val : float
        Maximum absolute value of x.

    Returns
    -------
    float
        exp(clipped x).
    """
    return float(np.exp(np.clip(x, -max_val, max_val)))


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of result dicts to a DataFrame.

    Parameters
    ----------
    results : list of dict
        Each dict is one model result.

    Returns
    -------
    pd.DataFrame
    """
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)
