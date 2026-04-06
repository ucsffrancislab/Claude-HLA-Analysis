"""
Command-line interface for the HLA analysis pipeline.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

from hla_analysis.config import (
    AnalysisConfig,
    DEFAULT_RISK_COVARIATES,
    DEFAULT_SURVIVAL_COVARIATES,
    STRATA_DEFINITIONS,
    FEATURE_TYPES,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the HLA analysis CLI.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="hla-analysis",
        description="HLA Allele Risk & Survival Analysis Pipeline with Meta-Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with 4 datasets
  python -m hla_analysis \
    --dosage-files cidr.csv i370.csv onco.csv tcga.csv \
    --covariate-files cidr_cov.csv i370_cov.csv onco_cov.csv tcga_cov.csv \
    --dataset-names CIDR I370 ONCO TCGA \
    --output-dir results/

  # Risk only, specific strata
  python -m hla_analysis \
    --dosage-files data/*.csv \
    --covariate-files cov/*.csv \
    --analyses risk \
    --strata overall idh_wt \
    --workers 16
        """,
    )

    # ── Input/Output ──
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--dosage-files", nargs="+", required=True,
        help="Paths to HLA dosage CSV files (one per dataset).",
    )
    io_group.add_argument(
        "--covariate-files", nargs="+", required=True,
        help="Paths to covariate CSV files (one per dataset).",
    )
    io_group.add_argument(
        "--dataset-names", nargs="+", default=None,
        help="Short names for each dataset. If omitted, derived from filenames.",
    )
    io_group.add_argument(
        "--output-dir", default="results",
        help="Output directory (default: results/).",
    )

    # ── VCF Options ──
    vcf_group = parser.add_argument_group("VCF Options")
    vcf_group.add_argument(
        "--dosage-format", default="auto", choices=["auto", "csv", "vcf"],
        help="Input dosage format: 'auto' detects from extension, 'csv' for CSV, "
             "'vcf' for VCF/VCF.GZ (default: auto).",
    )
    vcf_group.add_argument(
        "--vcf-field", default="DS",
        help="FORMAT sub-field to extract from VCF files (default: DS).",
    )
    vcf_group.add_argument(
        "--vcf-filter-prefixes", nargs="+", default=["HLA_", "AA_"],
        help="Variant-ID prefixes to keep when parsing VCF (default: HLA_ AA_).",
    )
    vcf_group.add_argument(
        "--include-snps", action="store_true", default=False,
        help="Also keep SNP_* variants from VCF files.",
    )

    # ── Analysis Selection ──
    analysis_group = parser.add_argument_group("Analysis Selection")
    analysis_group.add_argument(
        "--analyses", nargs="+", default=["risk", "survival"],
        choices=["risk", "survival"],
        help="Analyses to run (default: risk survival).",
    )
    analysis_group.add_argument(
        "--strata", nargs="+",
        default=list(STRATA_DEFINITIONS.keys()),
        choices=list(STRATA_DEFINITIONS.keys()),
        help="Strata to analyse (default: all).",
    )
    analysis_group.add_argument(
        "--feature-types", nargs="+",
        default=list(FEATURE_TYPES.keys()),
        choices=list(FEATURE_TYPES.keys()),
        help="HLA feature types to include (default: all).",
    )
    analysis_group.add_argument(
        "--split-by-feature-type", action="store_true", default=True,
        help="Run separate analyses for alleles and amino acid positions (default: True).",
    )
    analysis_group.add_argument(
        "--no-split-by-feature-type", dest="split_by_feature_type", action="store_false",
        help="Run a single combined analysis for all feature types.",
    )

    # ── Covariates ──
    cov_group = parser.add_argument_group("Covariates")
    cov_group.add_argument(
        "--risk-covariates", nargs="+",
        default=list(DEFAULT_RISK_COVARIATES),
        help="Covariates for risk analysis (default: age sex PC1-PC8).",
    )
    cov_group.add_argument(
        "--survival-covariates", nargs="+",
        default=list(DEFAULT_SURVIVAL_COVARIATES),
        help="Covariates for survival analysis (default: age sex grade rad chemo).",
    )
    cov_group.add_argument(
        "--covariate-strategies", nargs="+",
        default=["full", "reduced"], choices=["full", "reduced"],
        help="Covariate strategies (default: full reduced). Ignored when --sensitivity-analysis is set.",
    )
    cov_group.add_argument(
        "--sensitivity-analysis", "--sensitivity", action="store_true",
        default=False,
        help="Enable per-covariate sensitivity analysis. Overrides --covariate-strategies. "
             "Runs models with all covariates, dropping each missing covariate individually, "
             "and with no covariates, so you can compare the impact of each covariate.",
    )

    # ── Thresholds ──
    thresh_group = parser.add_argument_group("Thresholds")
    thresh_group.add_argument(
        "--min-carriers", type=int, default=10,
        help="Min carriers per group for risk analysis (default: 10).",
    )
    thresh_group.add_argument(
        "--min-events", type=int, default=5,
        help="Min events among carriers for survival (default: 5).",
    )
    thresh_group.add_argument(
        "--missingness-threshold", type=float, default=0.3,
        help="Max missingness fraction for reduced strategy (default: 0.3).",
    )
    thresh_group.add_argument(
        "--fdr-threshold", type=float, default=0.05,
        help="FDR significance threshold (default: 0.05).",
    )
    thresh_group.add_argument(
        "--meta-min-datasets", type=int, default=2,
        help="Min datasets for meta-analysis (default: 2).",
    )
    thresh_group.add_argument(
        "--maf-threshold-allele", type=float, default=0.01,
        help="MAF threshold for classical HLA alleles (default: 0.01 = 1%%).",
    )
    thresh_group.add_argument(
        "--maf-threshold-aa", type=float, default=0.005,
        help="MAF threshold for amino acid positions (default: 0.005 = 0.5%%).",
    )
    thresh_group.add_argument(
        "--min-imputation-r2", type=float, default=0.3,
        help="Minimum imputation R² for feature inclusion (default: 0.3).",
    )

    # ── Firth Penalization ──
    firth_group = parser.add_argument_group("Firth Penalization")
    firth_group.add_argument(
        "--use-firth", action="store_true", default=True,
        help="Use Firth's penalized logistic regression (default: True).",
    )
    firth_group.add_argument(
        "--no-firth", dest="use_firth", action="store_false",
        help="Disable Firth's penalized logistic regression.",
    )

    # ── HPC / Parallelism ──
    hpc_group = parser.add_argument_group("HPC / Parallelism")
    hpc_group.add_argument(
        "--workers", type=int, default=-1,
        help="Number of parallel workers (-1 = auto-detect, default: -1).",
    )
    hpc_group.add_argument(
        "--memory-limit", type=float, default=-1,
        help="Memory limit in GB (-1 = auto-detect, default: -1).",
    )
    hpc_group.add_argument(
        "--chunk-size", type=int, default=500,
        help="Features per processing chunk (default: 500).",
    )

    # ── Cox Solver ──
    cox_group = parser.add_argument_group("Cox Solver")
    cox_group.add_argument(
        "--cox-solver", default="lifelines", choices=["custom", "lifelines"],
        help="Cox PH solver to use (default: lifelines).",
    )
    cox_group.add_argument(
        "--cox-penalizer", type=float, default=0.01,
        help="Penalizer for lifelines CoxPHFitter (default: 0.01).",
    )
    cox_group.add_argument(
        "--cox-max-iter", type=int, default=50,
        help="Max iterations for custom Cox solver (default: 50).",
    )
    cox_group.add_argument(
        "--cox-tol", type=float, default=1e-9,
        help="Convergence tolerance for custom Cox solver (default: 1e-9).",
    )

    # ── Visualization ──
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument(
        "--plots", nargs="+", default=["manhattan", "forest", "heatmap", "qq"],
        choices=["manhattan", "forest", "heatmap", "comparison", "sensitivity", "qq"],
        help="Plot types to generate (default: manhattan forest heatmap qq).",
    )
    viz_group.add_argument(
        "--max-forest-signals", type=int, default=15,
        help="Maximum signals per panel in forest plots (default: 15).",
    )

    # ── Miscellaneous ──
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    misc_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    return parser


def parse_args(argv: Optional[List[str]] = None) -> AnalysisConfig:
    """Parse command-line arguments and return an AnalysisConfig.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. If None, uses sys.argv.

    Returns
    -------
    AnalysisConfig
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Derive dataset names from filenames if not provided
    if args.dataset_names is None:
        args.dataset_names = [
            os.path.splitext(os.path.basename(f))[0]
            for f in args.dosage_files
        ]

    config = AnalysisConfig(
        dosage_files=args.dosage_files,
        covariate_files=args.covariate_files,
        dataset_names=args.dataset_names,
        output_dir=args.output_dir,
        analyses=args.analyses,
        strata=args.strata,
        risk_covariates=args.risk_covariates,
        survival_covariates=args.survival_covariates,
        covariate_strategies=args.covariate_strategies,
        sensitivity_analysis=args.sensitivity_analysis,
        dosage_format=args.dosage_format,
        vcf_field=args.vcf_field,
        vcf_filter_prefixes=args.vcf_filter_prefixes,
        include_snps=args.include_snps,
        min_carriers=args.min_carriers,
        min_events=args.min_events,
        missingness_threshold=args.missingness_threshold,
        fdr_threshold=args.fdr_threshold,
        meta_min_datasets=args.meta_min_datasets,
        maf_threshold_allele=args.maf_threshold_allele,
        maf_threshold_aa=args.maf_threshold_aa,
        min_imputation_r2=args.min_imputation_r2,
        use_firth=args.use_firth,
        workers=args.workers,
        memory_limit=args.memory_limit,
        chunk_size=args.chunk_size,
        feature_types=args.feature_types,
        plots=args.plots,
        max_forest_signals=args.max_forest_signals,
        split_by_feature_type=args.split_by_feature_type,
        log_level=args.log_level,
        seed=args.seed,
        cox_solver=args.cox_solver,
        cox_penalizer=args.cox_penalizer,
        cox_max_iter=args.cox_max_iter,
        cox_tol=args.cox_tol,
    )

    return config
