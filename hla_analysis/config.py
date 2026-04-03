"""
Configuration and default parameters for HLA analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import multiprocessing
import logging

logger = logging.getLogger(__name__)

# Default strata definitions
STRATA_DEFINITIONS = {
    "overall": {
        "description": "All cases vs all controls",
        "case_filter": None,  # No additional filter on cases
    },
    "idh_wt": {
        "description": "IDH wild-type cases vs controls",
        "case_filter": {"idh": 0},
    },
    "idh_mut": {
        "description": "IDH mutant cases vs controls",
        "case_filter": {"idh": 1},
    },
    "idh_mut_intact": {
        "description": "IDH mutant, 1p/19q intact cases vs controls",
        "case_filter": {"idh": 1, "pq": 0},
    },
    "idh_mut_codel": {
        "description": "IDH mutant, 1p/19q co-deleted cases vs controls",
        "case_filter": {"idh": 1, "pq": 1},
    },
}

# Default covariates
DEFAULT_RISK_COVARIATES = ["age", "sex", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]
DEFAULT_SURVIVAL_COVARIATES = ["age", "sex", "grade", "treated"]

# Feature type classification
FEATURE_TYPES = {
    "classical_2digit": "Two-digit classical HLA alleles (e.g., HLA_A_01)",
    "classical_4digit": "Four-digit classical HLA alleles (e.g., HLA_A_01:01)",
    "amino_acid": "Amino acid variants (e.g., AA_A_9_V)",
}


@dataclass
class SensitivityStrategy:
    """A single covariate strategy for sensitivity analysis.

    Attributes
    ----------
    name : str
        Strategy label, e.g. 'all_covariates', 'drop_age', 'no_covariates'.
    covariates : list of str
        Covariates to include in the model.
    description : str
        Human-readable description.
    """

    name: str
    covariates: List[str]
    description: str = ""


@dataclass
class AnalysisConfig:
    """Central configuration for the HLA analysis pipeline.

    Attributes
    ----------
    dosage_files : list of str
        Paths to HLA dosage CSV files (one per dataset).
    covariate_files : list of str
        Paths to covariate CSV files (one per dataset).
    dataset_names : list of str
        Short names for each dataset.
    output_dir : str
        Directory for output files.
    analyses : list of str
        Which analyses to run: 'risk', 'survival', or both.
    strata : list of str
        Which strata to analyse.
    risk_covariates : list of str
        Covariates for logistic regression (risk analysis).
    survival_covariates : list of str
        Covariates for Cox PH (survival analysis).
    covariate_strategies : list of str
        'full' and/or 'reduced'.
    sensitivity_analysis : bool
        When True, generate per-covariate drop strategies instead of FULL/REDUCED.
    min_carriers : int
        Minimum carrier count in each group for risk analysis.
    min_events : int
        Minimum events among carriers for survival analysis.
    missingness_threshold : float
        Max fraction of missing values before a covariate is dropped (reduced strategy).
    fdr_threshold : float
        FDR threshold for significance.
    meta_min_datasets : int
        Minimum number of datasets for meta-analysis.
    workers : int
        Number of parallel workers.
    memory_limit : float
        Memory limit in GB.
    chunk_size : int
        Number of features per processing chunk.
    feature_types : list of str
        Which feature types to include.
    plots : list of str
        Which plots to generate.
    log_level : str
        Logging level.
    seed : int
        Random seed for reproducibility.
    dosage_format : str
        Input format: 'auto' (detect from extension), 'csv', or 'vcf'.
    vcf_field : str
        FORMAT sub-field to extract from VCF (default 'DS').
    vcf_filter_prefixes : list of str
        Variant-ID prefixes to keep when parsing VCF.
    include_snps : bool
        If True, also keep SNP_* variants from VCF.
    cox_solver : str
        Cox solver to use: 'custom' or 'lifelines'.
    cox_penalizer : float
        Penalizer for lifelines CoxPHFitter.
    cox_max_iter : int
        Maximum iterations for custom Cox solver.
    cox_tol : float
        Convergence tolerance for custom Cox solver.
    """

    # Input/output
    dosage_files: List[str] = field(default_factory=list)
    covariate_files: List[str] = field(default_factory=list)
    dataset_names: List[str] = field(default_factory=list)
    output_dir: str = "results"

    # Analysis selection
    analyses: List[str] = field(default_factory=lambda: ["risk", "survival"])
    strata: List[str] = field(
        default_factory=lambda: ["overall", "idh_wt", "idh_mut", "idh_mut_intact", "idh_mut_codel"]
    )

    # Covariates
    risk_covariates: List[str] = field(default_factory=lambda: list(DEFAULT_RISK_COVARIATES))
    survival_covariates: List[str] = field(default_factory=lambda: list(DEFAULT_SURVIVAL_COVARIATES))
    covariate_strategies: List[str] = field(default_factory=lambda: ["full", "reduced"])

    # Sensitivity analysis
    sensitivity_analysis: bool = False

    # VCF input options
    dosage_format: str = "auto"          # "auto", "csv", or "vcf"
    vcf_field: str = "DS"                # FORMAT sub-field to extract
    vcf_filter_prefixes: List[str] = field(default_factory=lambda: ["HLA_", "AA_"])
    include_snps: bool = False           # also keep SNP_* variants from VCF

    # Thresholds
    min_carriers: int = 10
    min_events: int = 5
    missingness_threshold: float = 0.3
    fdr_threshold: float = 0.05
    meta_min_datasets: int = 2

    # HPC / parallelism
    workers: int = -1  # -1 = auto-detect
    memory_limit: float = -1.0  # -1 = auto-detect
    chunk_size: int = 500

    # Feature selection
    feature_types: List[str] = field(
        default_factory=lambda: ["classical_2digit", "classical_4digit", "amino_acid"]
    )

    # Visualization
    plots: List[str] = field(default_factory=lambda: ["manhattan", "forest", "heatmap"])

    # Logging & reproducibility
    log_level: str = "INFO"
    seed: int = 42

    # Cox solver settings
    cox_solver: str = "custom"
    cox_penalizer: float = 0.01
    cox_max_iter: int = 50
    cox_tol: float = 1e-9

    def __post_init__(self):
        """Validate and resolve auto-detect settings."""
        # Resolve workers
        if self.workers <= 0:
            self.workers = max(1, multiprocessing.cpu_count() - 1)

        # Resolve memory limit
        if self.memory_limit <= 0:
            try:
                import psutil
                self.memory_limit = psutil.virtual_memory().total / (1024 ** 3)
            except ImportError:
                self.memory_limit = 16.0  # conservative default
                logger.warning(
                    "psutil not installed; defaulting memory_limit to %.0f GB", self.memory_limit
                )

        self.validate()

    def validate(self):
        """Check configuration consistency."""
        if len(self.dosage_files) != len(self.covariate_files):
            raise ValueError(
                f"Number of dosage files ({len(self.dosage_files)}) must match "
                f"covariate files ({len(self.covariate_files)})"
            )
        if self.dataset_names and len(self.dataset_names) != len(self.dosage_files):
            raise ValueError(
                f"Number of dataset names ({len(self.dataset_names)}) must match "
                f"dosage files ({len(self.dosage_files)})"
            )
        for a in self.analyses:
            if a not in ("risk", "survival"):
                raise ValueError(f"Unknown analysis type: {a!r}. Must be 'risk' or 'survival'.")
        for s in self.strata:
            if s not in STRATA_DEFINITIONS:
                raise ValueError(
                    f"Unknown stratum: {s!r}. Available: {list(STRATA_DEFINITIONS.keys())}"
                )
        if not self.sensitivity_analysis:
            for cs in self.covariate_strategies:
                if cs not in ("full", "reduced"):
                    raise ValueError(f"Unknown covariate strategy: {cs!r}. Must be 'full' or 'reduced'.")
        for ft in self.feature_types:
            if ft not in FEATURE_TYPES:
                raise ValueError(
                    f"Unknown feature type: {ft!r}. Available: {list(FEATURE_TYPES.keys())}"
                )
        for p in self.plots:
            if p not in ("manhattan", "forest", "heatmap", "comparison", "sensitivity"):
                raise ValueError(
                    f"Unknown plot type: {p!r}. Available: manhattan, forest, heatmap, comparison, sensitivity"
                )
        if self.min_carriers < 1:
            raise ValueError("min_carriers must be >= 1")
        if self.min_events < 1:
            raise ValueError("min_events must be >= 1")
        if not 0 < self.missingness_threshold <= 1:
            raise ValueError("missingness_threshold must be in (0, 1]")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.dosage_format not in ("auto", "csv", "vcf"):
            raise ValueError(
                f"Unknown dosage_format: {self.dosage_format!r}. "
                "Must be 'auto', 'csv', or 'vcf'."
            )

    def generate_sensitivity_strategies(
        self,
        covariate_names: List[str],
        missingness_info: Dict[str, float],
    ) -> List["SensitivityStrategy"]:
        """Generate covariate strategies for sensitivity analysis.

        Creates:
        - ``all_covariates``: include all specified covariates (like FULL)
        - ``drop_{name}``: for every covariate with **any** missingness, a
          strategy that excludes that single covariate
        - ``no_covariates``: unadjusted model (maximum sample size)

        Parameters
        ----------
        covariate_names : list of str
            Requested covariate column names that are available.
        missingness_info : dict of str -> float
            Mapping from covariate name to fraction of missingness among
            relevant samples (0.0 – 1.0).

        Returns
        -------
        list of SensitivityStrategy
        """
        strategies: List[SensitivityStrategy] = []

        # 1. All covariates (baseline, like FULL)
        strategies.append(SensitivityStrategy(
            name="all_covariates",
            covariates=list(covariate_names),
            description="All specified covariates; samples with missing values dropped",
        ))

        # 2. Drop each covariate that has ANY missingness
        for cov, miss_frac in sorted(missingness_info.items()):
            if miss_frac > 0 and cov in covariate_names:
                remaining = [c for c in covariate_names if c != cov]
                strategies.append(SensitivityStrategy(
                    name=f"drop_{cov}",
                    covariates=remaining,
                    description=(
                        f"Excluding {cov} ({miss_frac*100:.1f}% missing); "
                        f"retains more samples"
                    ),
                ))

        # 3. No covariates (unadjusted, maximum sample size)
        strategies.append(SensitivityStrategy(
            name="no_covariates",
            covariates=[],
            description="Unadjusted model; no covariates, maximum sample size",
        ))

        return strategies

    def estimate_memory_per_dataset(self, n_samples: int, n_features: int) -> float:
        """Estimate memory in GB for one dataset's dosage matrix (float32)."""
        bytes_per_element = 4  # float32
        matrix_bytes = n_samples * n_features * bytes_per_element
        overhead_factor = 2.5  # copies during computation
        return (matrix_bytes * overhead_factor) / (1024 ** 3)
