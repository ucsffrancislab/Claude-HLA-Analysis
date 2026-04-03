"""
Sensitivity analysis: compare results across covariate strategies.

When ``--sensitivity-analysis`` is enabled the pipeline runs each feature
with several covariate sets (all, drop-one, none). This module creates the
side-by-side comparison tables.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_sensitivity_comparison(
    results_df: pd.DataFrame,
    analysis_type: str = "risk",
) -> pd.DataFrame:
    """Pivot per-feature results across sensitivity strategies.

    For each *feature × stratum × dataset*, shows the effect size and
    p-value under every covariate strategy side by side, so users can
    instantly see whether adding / removing a covariate changes the
    association signal.

    Parameters
    ----------
    results_df : pd.DataFrame
        Combined per-dataset results containing at least:
        ``feature``, ``stratum``, ``dataset``, ``strategy``,
        ``beta``, ``pvalue``, ``se``.
        The ``strategy`` column must contain the sensitivity strategy
        names (e.g. ``all_covariates``, ``drop_age``, …).
    analysis_type : str
        ``'risk'`` or ``'survival'``.  Used to pick the right effect
        column name (``or_val`` vs ``hr``).

    Returns
    -------
    pd.DataFrame
        Wide-format table with one row per feature × stratum × dataset
        and columns ``{strategy}_beta``, ``{strategy}_pvalue``,
        ``{strategy}_effect`` (OR or HR) for each strategy.
    """
    if results_df.empty:
        return pd.DataFrame()

    effect_col = "or_val" if analysis_type == "risk" else "hr"
    index_cols = ["feature", "stratum", "dataset"]

    # Ensure all index cols exist
    for c in index_cols:
        if c not in results_df.columns:
            logger.warning("Missing column '%s' in results for sensitivity comparison", c)
            return pd.DataFrame()

    strategies = sorted(results_df["strategy"].unique())
    if len(strategies) < 2:
        logger.warning("Need ≥2 strategies for sensitivity comparison; found %d", len(strategies))
        return pd.DataFrame()

    pieces: List[pd.DataFrame] = []
    for strat in strategies:
        subset = results_df[results_df["strategy"] == strat][
            index_cols + ["beta", "se", "pvalue"]
            + ([effect_col] if effect_col in results_df.columns else [])
        ].copy()
        rename_map = {
            "beta": f"{strat}_beta",
            "se": f"{strat}_se",
            "pvalue": f"{strat}_pvalue",
        }
        if effect_col in subset.columns:
            rename_map[effect_col] = f"{strat}_effect"
        subset = subset.rename(columns=rename_map)
        pieces.append(subset)

    # Merge all strategies on the index columns
    merged = pieces[0]
    for p in pieces[1:]:
        merged = merged.merge(p, on=index_cols, how="outer")

    # Add convenience columns: max absolute change in beta across strategies
    beta_cols = [f"{s}_beta" for s in strategies if f"{s}_beta" in merged.columns]
    if len(beta_cols) >= 2:
        betas = merged[beta_cols].values
        merged["max_beta_range"] = np.nanmax(betas, axis=1) - np.nanmin(betas, axis=1)
    else:
        merged["max_beta_range"] = np.nan

    # Sort by the all_covariates p-value (if present)
    sort_col = "all_covariates_pvalue"
    if sort_col in merged.columns:
        merged = merged.sort_values(sort_col).reset_index(drop=True)

    return merged


def summarise_sensitivity(
    comparison_df: pd.DataFrame,
    strategies: List[str],
    pvalue_threshold: float = 0.05,
) -> pd.DataFrame:
    """Summarise sensitivity comparison: flag features whose significance
    status changes across strategies.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of :func:`create_sensitivity_comparison`.
    strategies : list of str
        Strategy names present in the comparison.
    pvalue_threshold : float
        Nominal p-value threshold for calling significance.

    Returns
    -------
    pd.DataFrame
        Subset of *comparison_df* where the significance call flips
        between at least two strategies, with an extra column
        ``sig_strategies`` listing which strategies reached significance.
    """
    if comparison_df.empty:
        return pd.DataFrame()

    p_cols = [f"{s}_pvalue" for s in strategies if f"{s}_pvalue" in comparison_df.columns]
    if len(p_cols) < 2:
        return pd.DataFrame()

    # For each row, check if significance status changes
    sig_matrix = comparison_df[p_cols] < pvalue_threshold
    n_sig = sig_matrix.sum(axis=1)
    flips = (n_sig > 0) & (n_sig < len(p_cols))

    result = comparison_df.loc[flips].copy()
    # List which strategies are significant
    sig_lists = []
    for _, row in result[p_cols].iterrows():
        sig_strats = [col.replace("_pvalue", "") for col in p_cols if row[col] < pvalue_threshold]
        sig_lists.append(",".join(sig_strats))
    result["sig_strategies"] = sig_lists

    return result.reset_index(drop=True)
