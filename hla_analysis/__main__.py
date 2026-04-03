"""
Main entry point: orchestrates the full HLA analysis pipeline.

Usage:
    python -m hla_analysis --dosage-files ... --covariate-files ...
"""

import logging
import os
import sys
import time
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from hla_analysis.cli import parse_args
from hla_analysis.config import AnalysisConfig, SensitivityStrategy
from hla_analysis.data_loader import DataLoader, find_common_features
from hla_analysis.risk_analysis import RiskAnalyzer
from hla_analysis.survival_analysis import SurvivalAnalyzer
from hla_analysis.meta_analysis import MetaAnalyzer, create_summary_tables
from hla_analysis.sensitivity import create_sensitivity_comparison
from hla_analysis.visualization import Visualizer
from hla_analysis.utils import setup_logging

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _resolve_strategies(
    config: AnalysisConfig,
    loader: DataLoader,
    covariates: pd.DataFrame,
    sample_mask: np.ndarray,
    analysis_type: str,
) -> List[Tuple[str, List[str]]]:
    """Return a list of (strategy_name, covariate_list) pairs.

    When *sensitivity_analysis* is off the list comes from the fixed
    ``covariate_strategies`` setting (``full`` / ``reduced``).

    When it is on, we inspect per-covariate missingness and generate
    ``all_covariates``, ``drop_<cov>``, and ``no_covariates`` strategies.
    """
    if analysis_type == "risk":
        base_covariates = config.risk_covariates
    else:
        base_covariates = config.survival_covariates

    available = [c for c in base_covariates if c in covariates.columns]

    if config.sensitivity_analysis:
        miss = loader.compute_missingness(covariates, sample_mask, available)
        sens_strategies = config.generate_sensitivity_strategies(available, miss)
        return [(s.name, s.covariates) for s in sens_strategies]
    else:
        # Traditional FULL / REDUCED
        result: List[Tuple[str, List[str]]] = []
        for strat in config.covariate_strategies:
            if strat == "full":
                result.append(("full", available))
            else:  # reduced
                kept = [
                    c for c in available
                    if covariates.loc[sample_mask, c].isna().mean()
                    <= config.missingness_threshold
                ]
                result.append(("reduced", kept))
        return result


# ── pipeline ───────────────────────────────────────────────────────────────

def run_pipeline(config: AnalysisConfig) -> Dict[str, pd.DataFrame]:
    """Run the full HLA analysis pipeline.

    Parameters
    ----------
    config : AnalysisConfig
        Pipeline configuration.

    Returns
    -------
    dict
        Keys: 'risk_results', 'survival_results', 'risk_meta', 'survival_meta',
              'summary_tables', 'sensitivity_risk', 'sensitivity_survival'.
    """
    t_start = time.time()
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)

    logger.info("=" * 60)
    logger.info("HLA Analysis Pipeline v1.2.1")
    logger.info("=" * 60)
    logger.info("Workers: %d, Memory limit: %.1f GB, Chunk size: %d",
                config.workers, config.memory_limit, config.chunk_size)
    logger.info("Analyses: %s", config.analyses)
    logger.info("Strata: %s", config.strata)
    if config.sensitivity_analysis:
        logger.info("Sensitivity analysis: ENABLED (overrides covariate strategies)")
    else:
        logger.info("Covariate strategies: %s", config.covariate_strategies)
    logger.info("Datasets: %s", config.dataset_names)

    # ── Load data ──
    loader = DataLoader(config)
    datasets = loader.load_all_datasets()
    logger.info("Loaded %d datasets", len(datasets))

    # ── Initialize analyzers ──
    risk_analyzer = RiskAnalyzer(config) if "risk" in config.analyses else None
    surv_analyzer = SurvivalAnalyzer(config) if "survival" in config.analyses else None
    meta_analyzer = MetaAnalyzer(config)

    all_risk_results = []
    all_survival_results = []

    # ── Per-dataset analysis ──
    for ds in datasets:
        ds_name = ds["dataset_name"]
        cov = ds["covariates"]
        dosage = ds["dosage"]
        features = ds["feature_names"]

        # Memory estimate
        mem_est = config.estimate_memory_per_dataset(dosage.shape[0], dosage.shape[1])
        logger.info("Dataset %s: estimated %.2f GB memory", ds_name, mem_est)
        if mem_est > config.memory_limit * 0.8:
            logger.warning("Dataset %s may exceed memory limit (%.1f GB > %.1f GB * 0.8)",
                           ds_name, mem_est, config.memory_limit)

        for stratum in config.strata:

            # ── Risk analysis ──
            if risk_analyzer is not None:
                try:
                    case_idx, ctrl_idx = loader.get_stratum_indices(cov, stratum)
                    combined_idx = case_idx | ctrl_idx

                    if case_idx.sum() < 2 or ctrl_idx.sum() < 2:
                        logger.warning("Skipping risk: %s/%s — insufficient samples "
                                       "(cases=%d, controls=%d)",
                                       ds_name, stratum, case_idx.sum(), ctrl_idx.sum())
                    else:
                        strategies = _resolve_strategies(
                            config, loader, cov, combined_idx, "risk",
                        )
                        for strategy_name, strategy_covs in strategies:
                            X_cov, used_covs, final_mask = loader.prepare_covariate_matrix(
                                cov, combined_idx, strategy_covs, "full",
                            )
                            y = cov.loc[final_mask, "case"].values.astype(np.float64)
                            dosage_sub = dosage[final_mask]

                            risk_df = risk_analyzer.analyze_stratum(
                                dosage_sub, y, X_cov, features,
                                ds_name, stratum, strategy_name, used_covs,
                            )
                            if not risk_df.empty:
                                all_risk_results.append(risk_df)

                except Exception as e:
                    logger.error("Risk analysis failed: %s/%s: %s",
                                 ds_name, stratum, e, exc_info=True)

            # ── Survival analysis ──
            if surv_analyzer is not None:
                try:
                    surv_idx = loader.get_survival_indices(cov, stratum)

                    if surv_idx.sum() < 5:
                        logger.warning("Skipping survival: %s/%s — insufficient samples (%d)",
                                       ds_name, stratum, surv_idx.sum())
                    else:
                        strategies = _resolve_strategies(
                            config, loader, cov, surv_idx, "survival",
                        )
                        for strategy_name, strategy_covs in strategies:
                            X_cov_s, used_covs_s, final_mask_s = loader.prepare_covariate_matrix(
                                cov, surv_idx, strategy_covs, "full",
                            )

                            time_vals = cov.loc[final_mask_s, "survdays"].values.astype(np.float64)
                            event_vals = cov.loc[final_mask_s, "vstatus"].values.astype(np.float64)
                            dosage_sub_s = dosage[final_mask_s]

                            n_events = int(event_vals.sum())
                            if n_events < 2:
                                logger.warning("Skipping survival: %s/%s/%s — <2 events",
                                               ds_name, stratum, strategy_name)
                                continue

                            surv_df = surv_analyzer.analyze_stratum(
                                dosage_sub_s, time_vals, event_vals,
                                X_cov_s, features, ds_name, stratum,
                                strategy_name, used_covs_s,
                            )
                            if not surv_df.empty:
                                all_survival_results.append(surv_df)

                except Exception as e:
                    logger.error("Survival analysis failed: %s/%s: %s",
                                 ds_name, stratum, e, exc_info=True)

    # ── Combine results ──
    risk_combined = pd.concat(all_risk_results, ignore_index=True) if all_risk_results else pd.DataFrame()
    surv_combined = pd.concat(all_survival_results, ignore_index=True) if all_survival_results else pd.DataFrame()

    # Save per-dataset results
    if not risk_combined.empty:
        path = os.path.join(config.output_dir, "risk_per_dataset.csv")
        risk_combined.to_csv(path, index=False)
        logger.info("Saved per-dataset risk results: %s (%d rows)", path, len(risk_combined))

    if not surv_combined.empty:
        path = os.path.join(config.output_dir, "survival_per_dataset.csv")
        surv_combined.to_csv(path, index=False)
        logger.info("Saved per-dataset survival results: %s (%d rows)", path, len(surv_combined))

    # ── Sensitivity comparison ──
    sens_risk = pd.DataFrame()
    sens_surv = pd.DataFrame()

    if config.sensitivity_analysis:
        if not risk_combined.empty:
            sens_risk = create_sensitivity_comparison(risk_combined, analysis_type="risk")
            if not sens_risk.empty:
                path = os.path.join(config.output_dir, "sensitivity_comparison_risk.csv")
                sens_risk.to_csv(path, index=False)
                logger.info("Saved risk sensitivity comparison: %s (%d rows)", path, len(sens_risk))

        if not surv_combined.empty:
            sens_surv = create_sensitivity_comparison(surv_combined, analysis_type="survival")
            if not sens_surv.empty:
                path = os.path.join(config.output_dir, "sensitivity_comparison_survival.csv")
                sens_surv.to_csv(path, index=False)
                logger.info("Saved survival sensitivity comparison: %s (%d rows)", path, len(sens_surv))

    # ── Meta-analysis ──
    risk_meta = pd.DataFrame()
    surv_meta = pd.DataFrame()

    if not risk_combined.empty and len(datasets) >= config.meta_min_datasets:
        risk_meta = meta_analyzer.run_meta_analysis(risk_combined, analysis_type="risk")
        if not risk_meta.empty:
            path = os.path.join(config.output_dir, "risk_meta_analysis.csv")
            risk_meta.to_csv(path, index=False)
            logger.info("Saved risk meta-analysis: %s (%d rows)", path, len(risk_meta))

    if not surv_combined.empty and len(datasets) >= config.meta_min_datasets:
        surv_meta = meta_analyzer.run_meta_analysis(
            surv_combined, analysis_type="survival"
        )
        if not surv_meta.empty:
            path = os.path.join(config.output_dir, "survival_meta_analysis.csv")
            surv_meta.to_csv(path, index=False)
            logger.info("Saved survival meta-analysis: %s (%d rows)", path, len(surv_meta))

    # ── Summary tables ──
    summary = create_summary_tables(risk_meta, surv_meta, config.fdr_threshold)
    for name, table in summary.items():
        if not table.empty:
            path = os.path.join(config.output_dir, f"summary_{name}.csv")
            table.to_csv(path, index=False)
            logger.info("Saved summary table: %s (%d rows)", path, len(table))

    # ── Visualization ──
    if config.plots:
        viz = Visualizer(
            output_dir=os.path.join(config.output_dir, "plots"),
            fdr_threshold=config.fdr_threshold,
        )

        # Determine unique strategies present in results
        risk_strategies = sorted(risk_combined["strategy"].unique()) if not risk_combined.empty and "strategy" in risk_combined.columns else []
        surv_strategies = sorted(surv_combined["strategy"].unique()) if not surv_combined.empty and "strategy" in surv_combined.columns else []

        for stratum in config.strata:
            for strategy in (risk_strategies or ["full"]):
                if "manhattan" in config.plots:
                    if not risk_meta.empty:
                        viz.manhattan_plot(risk_meta, "risk", stratum, strategy)
                if "forest" in config.plots:
                    if not risk_meta.empty and not risk_combined.empty:
                        viz.forest_plot(risk_meta, risk_combined, "risk", stratum, strategy)

            for strategy in (surv_strategies or ["full"]):
                if "manhattan" in config.plots:
                    if not surv_meta.empty:
                        viz.manhattan_plot(surv_meta, "survival", stratum, strategy)
                if "forest" in config.plots:
                    if not surv_meta.empty and not surv_combined.empty:
                        viz.forest_plot(surv_meta, surv_combined, "survival", stratum, strategy)

            if "heatmap" in config.plots:
                if not risk_meta.empty:
                    viz.heatmap_plot(risk_meta, "risk", strategy="full" if not config.sensitivity_analysis else "all_covariates")
                if not surv_meta.empty:
                    viz.heatmap_plot(surv_meta, "survival", strategy="full" if not config.sensitivity_analysis else "all_covariates")

        if "comparison" in config.plots:
            if not risk_meta.empty:
                viz.comparison_plot(risk_meta, "risk")
            if not surv_meta.empty:
                viz.comparison_plot(surv_meta, "survival")

        if "sensitivity" in config.plots and config.sensitivity_analysis:
            if not sens_risk.empty:
                viz.sensitivity_plot(sens_risk, "risk")
            if not sens_surv.empty:
                viz.sensitivity_plot(sens_surv, "survival")

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("=" * 60)

    return {
        "risk_results": risk_combined,
        "survival_results": surv_combined,
        "risk_meta": risk_meta,
        "survival_meta": surv_meta,
        "summary_tables": summary,
        "sensitivity_risk": sens_risk,
        "sensitivity_survival": sens_surv,
    }


def main(argv=None):
    """CLI entry point."""
    config = parse_args(argv)
    os.makedirs(config.output_dir, exist_ok=True)
    setup_logging(config.log_level, os.path.join(config.output_dir, "pipeline.log"))
    run_pipeline(config)


if __name__ == "__main__":
    main()
