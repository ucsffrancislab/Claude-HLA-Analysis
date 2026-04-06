"""
hla_analysis - HLA Allele Risk & Survival Analysis Pipeline
============================================================

A production-grade Python package for HLA allele association studies,
implementing logistic regression (risk), Cox proportional hazards (survival),
and inverse-variance weighted meta-analysis across multiple cohorts.

Supports both CSV and VCF (.vcf.gz) dosage input formats, including
haplotype construction from phased HDS data.
"""

from importlib.metadata import version as _version
try:
    __version__ = _version("hla-analysis")
except Exception:
    __version__ = "unknown"  # not installed as a package
__author__ = "HLA Analysis Team"
__license__ = "MIT"

from hla_analysis.config import AnalysisConfig, SensitivityStrategy
from hla_analysis.data_loader import DataLoader, compute_maf
from hla_analysis.risk_analysis import RiskAnalyzer
from hla_analysis.survival_analysis import SurvivalAnalyzer
from hla_analysis.meta_analysis import MetaAnalyzer
from hla_analysis.visualization import Visualizer
from hla_analysis.sensitivity import create_sensitivity_comparison
from hla_analysis.vcf_parser import parse_vcf_dosage, parse_vcf_to_dosage_df
from hla_analysis.haplotype import build_haplotype_dosage_matrix, call_haplotypes

__all__ = [
    "AnalysisConfig",
    "SensitivityStrategy",
    "DataLoader",
    "compute_maf",
    "RiskAnalyzer",
    "SurvivalAnalyzer",
    "MetaAnalyzer",
    "Visualizer",
    "create_sensitivity_comparison",
    "parse_vcf_dosage",
    "parse_vcf_to_dosage_df",
    "build_haplotype_dosage_matrix",
    "call_haplotypes",
]
