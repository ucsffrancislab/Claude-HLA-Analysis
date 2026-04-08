"""
Main entry point: orchestrates the full HLA analysis pipeline.

Usage:
    python -m hla_analysis --dosage-files ... --covariate-files ...
"""

import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hla_analysis import __version__
from hla_analysis.cli import parse_args
from hla_analysis.config import AnalysisConfig, SensitivityStrategy
from hla_analysis.data_loader import DataLoader, find_common_features
from hla_analysis.risk_analysis import RiskAnalyzer
from hla_analysis.survival_analysis import SurvivalAnalyzer
from hla_analysis.meta_analysis import MetaAnalyzer, create_summary_tables
from hla_analysis.sensitivity import create_sensitivity_comparison
from hla_analysis.visualization import Visualizer
from hla_analysis.utils import setup_logging, classify_feature

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


def _filter_features_by_type(
    features: List[str],
    feature_type: str,
) -> List[str]:
    """Filter features to only those matching a specific feature type.

    Parameters
    ----------
    features : list of str
        All feature names.
    feature_type : str
        'alleles' — keep HLA_ features that contain ':' (4-digit classical alleles)
        'amino_acids' — keep AA_ features

    Returns
    -------
    list of str
        Filtered feature names.
    """
    if feature_type == "alleles":
        return [f for f in features if f.startswith("HLA_") and ":" in f]
    elif feature_type == "amino_acids":
        return [f for f in features if f.startswith("AA_")]
    else:
        return features


def _run_single_analysis(
    config: AnalysisConfig,
    datasets: List[Dict],
    feature_type_label: str = "all",
) -> Dict[str, pd.DataFrame]:
    """Run the analysis pipeline for a specific feature type subset.

    Parameters
    ----------
    config : AnalysisConfig
        Pipeline configuration.
    datasets : list of dict
        Loaded datasets.
    feature_type_label : str
        'alleles', 'amino_acids', or 'all'.

    Returns
    -------
    dict
        Keys: risk_results, survival_results, risk_meta, survival_meta, etc.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    loader = DataLoader(config)

    # ── Initialize analyzers ──
    risk_analyzer = RiskAnalyzer(config) if "risk" in config.analyses else None
    surv_analyzer = SurvivalAnalyzer(config) if "survival" in config.analyses else None
    meta_analyzer = MetaAnalyzer(config)

    all_risk_results = []
    all_survival_results = []

    # Skip main analysis if --conditional-only
    _run_associations = not getattr(config, "conditional_only", False)

    # ── Per-dataset analysis ──
    for ds in (_run_associations and datasets or []):
        ds_name = ds["dataset_name"]
        cov = ds["covariates"]
        dosage = ds["dosage"]
        features = ds["feature_names"]

        # Filter features by type
        if feature_type_label != "all":
            selected_features = _filter_features_by_type(features, feature_type_label)
            if not selected_features:
                logger.warning(
                    "Dataset %s: no %s features found, skipping",
                    ds_name, feature_type_label,
                )
                continue
            # Get column indices for the selected features
            feat_indices = [features.index(f) for f in selected_features]
            dosage_sub_type = dosage[:, feat_indices]
            features_sub = selected_features
        else:
            dosage_sub_type = dosage
            features_sub = features

        # Memory estimate
        mem_est = config.estimate_memory_per_dataset(dosage_sub_type.shape[0], dosage_sub_type.shape[1])
        logger.info("Dataset %s [%s]: estimated %.2f GB memory, %d features",
                     ds_name, feature_type_label, mem_est, len(features_sub))

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
                            dosage_sub = dosage_sub_type[final_mask]

                            # Fix 1: early skip if insufficient cases/controls after covariate filtering
                            n_cases_final = int((y == 1).sum())
                            n_ctrls_final = int((y == 0).sum())
                            if n_cases_final < config.min_carriers or n_ctrls_final < config.min_carriers:
                                logger.warning(
                                    "Skipping risk: %s/%s/%s — after covariate filtering: "
                                    "cases=%d, controls=%d < min_carriers=%d",
                                    ds_name, stratum, strategy_name,
                                    n_cases_final, n_ctrls_final, config.min_carriers,
                                )
                                continue

                            risk_df = risk_analyzer.analyze_stratum(
                                dosage_sub, y, X_cov, features_sub,
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
                            dosage_sub_s = dosage_sub_type[final_mask_s]

                            # Fix 1: early skip if insufficient events after covariate filtering
                            n_events = int(event_vals.sum())
                            if n_events < config.min_events:
                                logger.warning(
                                    "Skipping survival: %s/%s/%s — after covariate "
                                    "filtering: events=%d < min_events=%d",
                                    ds_name, stratum, strategy_name,
                                    n_events, config.min_events,
                                )
                                continue
                            if n_events < 2:
                                logger.warning("Skipping survival: %s/%s/%s — <2 events",
                                               ds_name, stratum, strategy_name)
                                continue

                            surv_df = surv_analyzer.analyze_stratum(
                                dosage_sub_s, time_vals, event_vals,
                                X_cov_s, features_sub, ds_name, stratum,
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

    if _run_associations and not risk_combined.empty and len(datasets) >= config.meta_min_datasets:
        risk_meta = meta_analyzer.run_meta_analysis(risk_combined, analysis_type="risk")
        if not risk_meta.empty:
            path = os.path.join(config.output_dir, "risk_meta_analysis.csv")
            risk_meta.to_csv(path, index=False)
            logger.info("Saved risk meta-analysis: %s (%d rows)", path, len(risk_meta))

    if _run_associations and not surv_combined.empty and len(datasets) >= config.meta_min_datasets:
        surv_meta = meta_analyzer.run_meta_analysis(
            surv_combined, analysis_type="survival"
        )
        if not surv_meta.empty:
            path = os.path.join(config.output_dir, "survival_meta_analysis.csv")
            surv_meta.to_csv(path, index=False)
            logger.info("Saved survival meta-analysis: %s (%d rows)", path, len(surv_meta))

    # ── Best-adjusted meta-analysis ──
    risk_meta_best = pd.DataFrame()
    surv_meta_best = pd.DataFrame()

    if config.best_adjusted_meta:
        from hla_analysis.meta_analysis import create_best_adjusted_results

        if not risk_combined.empty and len(datasets) >= config.meta_min_datasets:
            best_risk = create_best_adjusted_results(risk_combined)
            if not best_risk.empty:
                risk_meta_best = meta_analyzer.run_meta_analysis(
                    best_risk, analysis_type="risk"
                )
                if not risk_meta_best.empty:
                    path = os.path.join(config.output_dir, "risk_meta_analysis_best_adjusted.csv")
                    risk_meta_best.to_csv(path, index=False)
                    logger.info("Saved best-adjusted risk meta: %s (%d rows)", path, len(risk_meta_best))

        if not surv_combined.empty and len(datasets) >= config.meta_min_datasets:
            best_surv = create_best_adjusted_results(surv_combined)
            if not best_surv.empty:
                surv_meta_best = meta_analyzer.run_meta_analysis(
                    best_surv, analysis_type="survival"
                )
                if not surv_meta_best.empty:
                    path = os.path.join(config.output_dir, "survival_meta_analysis_best_adjusted.csv")
                    surv_meta_best.to_csv(path, index=False)
                    logger.info("Saved best-adjusted survival meta: %s (%d rows)", path, len(surv_meta_best))

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
            max_forest_signals=config.max_forest_signals,
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
                if "qq" in config.plots:
                    if not risk_meta.empty:
                        viz.qq_plot(risk_meta, "risk", stratum, strategy)

            for strategy in (surv_strategies or ["full"]):
                if "manhattan" in config.plots:
                    if not surv_meta.empty:
                        viz.manhattan_plot(surv_meta, "survival", stratum, strategy)
                if "forest" in config.plots:
                    if not surv_meta.empty and not surv_combined.empty:
                        viz.forest_plot(surv_meta, surv_combined, "survival", stratum, strategy)
                if "qq" in config.plots:
                    if not surv_meta.empty:
                        viz.qq_plot(surv_meta, "survival", stratum, strategy)

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

    # ── Conditional analysis ──
    conditional_results = pd.DataFrame()
    if config.conditional_target:
        from hla_analysis.conditional import conditional_analysis

        logger.info("Running conditional analysis for target: %s", config.conditional_target)
        cond_rows = []
        for ds in datasets:
            ds_name = ds["dataset_name"]
            cov = ds["covariates"]
            dosage = ds["dosage"]
            features = ds["feature_names"]

            for stratum in config.strata:
                try:
                    if "risk" in config.analyses:
                        case_idx, ctrl_idx = loader.get_stratum_indices(cov, stratum)
                        combined_idx = case_idx | ctrl_idx
                        y_cond = cov.loc[combined_idx, "case"].values.astype(np.float64)
                        dosage_cond = dosage[combined_idx]
                        cond_df = conditional_analysis(
                            dosage_cond, features, config.conditional_target,
                            y=y_cond, analysis_type="risk",
                        )
                        if not cond_df.empty:
                            cond_df["dataset"] = ds_name
                            cond_df["stratum"] = stratum
                            cond_df["analysis_type"] = "risk"
                            cond_rows.append(cond_df)

                    if "survival" in config.analyses:
                        surv_idx = loader.get_survival_indices(cov, stratum)
                        if surv_idx.sum() >= 5:
                            time_c = cov.loc[surv_idx, "survdays"].values.astype(np.float64)
                            event_c = cov.loc[surv_idx, "vstatus"].values.astype(np.float64)
                            dosage_c = dosage[surv_idx]
                            cond_df = conditional_analysis(
                                dosage_c, features, config.conditional_target,
                                time=time_c, event=event_c,
                                analysis_type="survival",
                            )
                            if not cond_df.empty:
                                cond_df["dataset"] = ds_name
                                cond_df["stratum"] = stratum
                                cond_df["analysis_type"] = "survival"
                                cond_rows.append(cond_df)
                except Exception as e:
                    logger.error("Conditional analysis failed: %s/%s: %s",
                                 ds_name, stratum, e, exc_info=True)

        if cond_rows:
            conditional_results = pd.concat(cond_rows, ignore_index=True)
            # Sanitise target name for filename
            safe_name = config.conditional_target.replace("*", "_").replace(":", "_")
            path = os.path.join(config.output_dir, f"conditional_analysis_{safe_name}.csv")
            conditional_results.to_csv(path, index=False)
            logger.info("Saved conditional analysis: %s (%d rows)", path, len(conditional_results))

    return {
        "risk_results": risk_combined,
        "survival_results": surv_combined,
        "risk_meta": risk_meta,
        "survival_meta": surv_meta,
        "risk_meta_best_adjusted": risk_meta_best,
        "survival_meta_best_adjusted": surv_meta_best,
        "summary_tables": summary,
        "sensitivity_risk": sens_risk,
        "sensitivity_survival": sens_surv,
        "conditional_results": conditional_results,
    }


def _create_combined_summary(
    allele_results: Dict[str, pd.DataFrame],
    aa_results: Dict[str, pd.DataFrame],
    output_dir: str,
    fdr_threshold: float = 0.05,
    haplotype_results: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Create a combined summary comparing allele, amino acid, and haplotype results.

    Parameters
    ----------
    allele_results : dict
        Results from allele analysis.
    aa_results : dict
        Results from amino acid analysis.
    output_dir : str
        Output directory for combined results.
    fdr_threshold : float
        FDR threshold for significance.
    haplotype_results : dict, optional
        Results from haplotype analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for analysis_type in ["risk", "survival"]:
        meta_key = f"{analysis_type}_meta"
        results_key = f"{analysis_type}_results"

        for label, res in [("alleles", allele_results), ("amino_acids", aa_results)]:
            meta_df = res.get(meta_key, pd.DataFrame())
            per_ds_df = res.get(results_key, pd.DataFrame())

            if meta_df.empty:
                continue

            for stratum in meta_df["stratum"].unique() if "stratum" in meta_df.columns else ["overall"]:
                stratum_df = meta_df[meta_df["stratum"] == stratum] if "stratum" in meta_df.columns else meta_df

                n_tested = len(stratum_df)
                n_sig_fe = (stratum_df["fe_fdr"] < fdr_threshold).sum() if "fe_fdr" in stratum_df.columns else 0
                n_sig_re = (stratum_df["re_fdr"] < fdr_threshold).sum() if "re_fdr" in stratum_df.columns else 0
                min_p = stratum_df["fe_pvalue"].min() if "fe_pvalue" in stratum_df.columns and not stratum_df["fe_pvalue"].isna().all() else np.nan

                rows.append({
                    "feature_type": label,
                    "analysis_type": analysis_type,
                    "stratum": stratum,
                    "n_features_tested": n_tested,
                    "n_FE_significant": int(n_sig_fe),
                    "n_RE_significant": int(n_sig_re),
                    "min_FE_pvalue": min_p,
                })

    if rows:
        comparison_df = pd.DataFrame(rows)
        path = os.path.join(output_dir, "feature_type_comparison.csv")
        comparison_df.to_csv(path, index=False)
        logger.info("Saved feature type comparison: %s (%d rows)", path, len(comparison_df))

    # Top signals comparison
    top_signals = []
    for label, res in [("alleles", allele_results), ("amino_acids", aa_results)]:
        for analysis_type in ["risk", "survival"]:
            meta_key = f"{analysis_type}_meta"
            meta_df = res.get(meta_key, pd.DataFrame())
            if meta_df.empty or "fe_pvalue" not in meta_df.columns:
                continue
            top = meta_df.nsmallest(10, "fe_pvalue")[
                ["feature", "stratum", "strategy", "fe_pvalue", "fe_beta", "fe_se"]
            ].copy()
            top["feature_type"] = label
            top["analysis_type"] = analysis_type
            top_signals.append(top)

    if top_signals:
        top_df = pd.concat(top_signals, ignore_index=True)
        path = os.path.join(output_dir, "top_signals_comparison.csv")
        top_df.to_csv(path, index=False)
        logger.info("Saved top signals comparison: %s (%d rows)", path, len(top_df))


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
    logger.info("HLA Analysis Pipeline v%s", __version__)
    logger.info("=" * 60)
    logger.info("Workers: %d, Memory limit: %.1f GB, Chunk size: %d",
                config.workers, config.memory_limit, config.chunk_size)
    logger.info("Analyses: %s", config.analyses)
    logger.info("Strata: %s", config.strata)
    logger.info("MAF thresholds: allele=%.3f, AA=%.4f", config.maf_threshold_allele, config.maf_threshold_aa)
    logger.info("Min imputation R²: %.2f", config.min_imputation_r2)
    logger.info("Firth penalization: %s", config.use_firth)
    logger.info("Cox solver: %s (penalizer=%.3f)", config.cox_solver, config.cox_penalizer)
    if config.sensitivity_analysis:
        logger.info("Sensitivity analysis: ENABLED (overrides covariate strategies)")
    else:
        logger.info("Covariate strategies: %s", config.covariate_strategies)
    logger.info("Datasets: %s", config.dataset_names)
    logger.info("Split by feature type: %s", config.split_by_feature_type)

    # ── Load data ──
    loader = DataLoader(config)
    datasets = loader.load_all_datasets()
    logger.info("Loaded %d datasets", len(datasets))

    if config.split_by_feature_type:
        # ── Run separate analyses for alleles and amino acids ──
        logger.info("=" * 60)
        logger.info("Running ALLELE analysis")
        logger.info("=" * 60)

        allele_config = AnalysisConfig(
            **{k: v for k, v in config.__dict__.items()
               if k not in ("output_dir",)}
        )
        # We don't re-validate since we're just changing output_dir
        object.__setattr__(allele_config, "output_dir",
                           os.path.join(config.output_dir, "alleles"))
        allele_results = _run_single_analysis(allele_config, datasets, "alleles")

        logger.info("=" * 60)
        logger.info("Running AMINO ACID analysis")
        logger.info("=" * 60)

        aa_config = AnalysisConfig(
            **{k: v for k, v in config.__dict__.items()
               if k not in ("output_dir",)}
        )
        object.__setattr__(aa_config, "output_dir",
                           os.path.join(config.output_dir, "amino_acids"))
        aa_results = _run_single_analysis(aa_config, datasets, "amino_acids")

        # ── Haplotype analysis (optional) ──
        haplotype_results = None
        if config.haplotype_analysis:
            logger.info("=" * 60)
            logger.info("Running HAPLOTYPE analysis")
            logger.info("=" * 60)

            from hla_analysis.haplotype import build_haplotype_dosage_matrix

            hap_datasets = []
            for i, ds in enumerate(datasets):
                vcf_path = config.dosage_files[i]
                hap_dosage_df = build_haplotype_dosage_matrix(
                    vcf_path,
                    loci=config.haplotype_loci,
                    resolution=config.haplotype_resolution,
                    min_freq=config.min_haplotype_freq,
                )
                hap_features = [c for c in hap_dosage_df.columns if c != "sample_id"]
                if not hap_features:
                    logger.warning("Dataset %s: no haplotypes survived filtering", ds["dataset_name"])
                    continue

                # Merge with covariates on sample_id
                cov = ds["covariates"]
                merged = hap_dosage_df.merge(cov, on="sample_id", how="inner")
                if len(merged) == 0:
                    logger.warning("Dataset %s: no samples matched between haplotypes and covariates",
                                   ds["dataset_name"])
                    continue

                hap_dosage_matrix = merged[hap_features].values.astype(np.float32)
                cov_cols = [c for c in merged.columns if c not in hap_features and c != "sample_id"]
                hap_cov = merged[["sample_id"] + cov_cols].copy()

                hap_datasets.append({
                    "dataset_name": ds["dataset_name"],
                    "covariates": hap_cov,
                    "dosage": hap_dosage_matrix,
                    "feature_names": hap_features,
                    "sample_ids": merged["sample_id"].tolist(),
                })

            if hap_datasets:
                hap_config = AnalysisConfig(
                    **{k: v for k, v in config.__dict__.items()
                       if k not in ("output_dir",)}
                )
                object.__setattr__(hap_config, "output_dir",
                                   os.path.join(config.output_dir, "haplotypes"))
                haplotype_results = _run_single_analysis(hap_config, hap_datasets, "all")
            else:
                logger.warning("No haplotype datasets available for analysis")

        # ── Create combined summary ──
        combined_dir = os.path.join(config.output_dir, "combined")
        _create_combined_summary(
            allele_results, aa_results, combined_dir, config.fdr_threshold,
            haplotype_results=haplotype_results,
        )

        elapsed = time.time() - t_start
        logger.info("=" * 60)
        logger.info("Pipeline complete in %.1f seconds", elapsed)
        logger.info("=" * 60)

        # Collect all result sets
        all_result_sets = [allele_results, aa_results]
        if haplotype_results is not None:
            all_result_sets.append(haplotype_results)

        # Return merged results
        return {
            "risk_results": pd.concat(
                [r.get("risk_results", pd.DataFrame()) for r in all_result_sets],
                ignore_index=True,
            ),
            "survival_results": pd.concat(
                [r.get("survival_results", pd.DataFrame()) for r in all_result_sets],
                ignore_index=True,
            ),
            "risk_meta": pd.concat(
                [r.get("risk_meta", pd.DataFrame()) for r in all_result_sets],
                ignore_index=True,
            ),
            "survival_meta": pd.concat(
                [r.get("survival_meta", pd.DataFrame()) for r in all_result_sets],
                ignore_index=True,
            ),
            "summary_tables": {
                k: v
                for r in all_result_sets
                for k, v in r.get("summary_tables", {}).items()
            },
            "sensitivity_risk": pd.concat(
                [r.get("sensitivity_risk", pd.DataFrame()) for r in all_result_sets],
                ignore_index=True,
            ),
            "sensitivity_survival": pd.concat(
                [r.get("sensitivity_survival", pd.DataFrame()) for r in all_result_sets],
                ignore_index=True,
            ),
        }

    else:
        # ── Run single combined analysis ──
        results = _run_single_analysis(config, datasets, "all")

        elapsed = time.time() - t_start
        logger.info("=" * 60)
        logger.info("Pipeline complete in %.1f seconds", elapsed)
        logger.info("=" * 60)

        return results


def main(argv=None):
    """CLI entry point."""
    config = parse_args(argv)
    os.makedirs(config.output_dir, exist_ok=True)
    setup_logging(config.log_level, os.path.join(config.output_dir, "pipeline.log"))
    run_pipeline(config)


if __name__ == "__main__":
    main()
