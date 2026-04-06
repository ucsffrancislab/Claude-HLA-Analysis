"""
Haplotype construction from phased VCF data.

Parses HDS (haploid dosage) fields from imputed VCF files, calls
per-chromosome alleles at each HLA locus, joins them into multi-locus
haplotypes, and returns a dosage matrix suitable for the existing
risk/survival analysis pipeline.
"""

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hla_analysis.vcf_parser import _open_vcf, _find_field_index

logger = logging.getLogger(__name__)

# Default gene loci for haplotype construction
DEFAULT_HAPLOTYPE_LOCI: List[str] = ["A", "B", "C", "DRB1", "DQB1"]

# Regex to extract gene name and allele digits from a classical HLA variant ID
# Matches e.g. HLA_A*02:01, HLA_DRB1*15:01, HLA_A*01
_HLA_PATTERN = re.compile(r"^HLA_([A-Za-z0-9]+)\*(.+)$")


def _parse_gene_allele(variant_id: str) -> Optional[Tuple[str, str]]:
    """Extract gene name and allele designation from a classical HLA variant ID.

    Parameters
    ----------
    variant_id : str
        Raw VCF variant ID, e.g. ``HLA_A*02:01`` or ``HLA_DRB1*15:01``.

    Returns
    -------
    tuple of (str, str) or None
        ``(gene, allele)`` e.g. ``("A", "02:01")``, or *None* if the ID
        does not match the expected pattern.
    """
    m = _HLA_PATTERN.match(variant_id)
    if m is None:
        return None
    return m.group(1), m.group(2)


def _truncate_to_2digit(allele: str) -> str:
    """Truncate a 4-digit allele designation to 2-digit.

    Parameters
    ----------
    allele : str
        e.g. ``"02:01"`` or ``"02"``.

    Returns
    -------
    str
        e.g. ``"02"``.
    """
    return allele.split(":")[0]


def extract_haploid_dosages(
    vcf_path: str,
    filter_prefixes: Sequence[str] = ("HLA_",),
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Parse a VCF and extract HDS (haploid dosage) fields for classical HLA alleles.

    The HDS field contains two comma-separated values per sample: the
    haploid dosage for chromosome 1 and chromosome 2.

    Parameters
    ----------
    vcf_path : str
        Path to a ``.vcf`` or ``.vcf.gz`` file.
    filter_prefixes : sequence of str
        Variant-ID prefixes to keep (default: only ``HLA_``).

    Returns
    -------
    sample_ids : list of str
        Sample identifiers from the VCF header.
    variant_ids : list of str
        Variant IDs that passed the prefix filter.
    hds_hap1 : np.ndarray
        Shape ``(n_variants, n_samples)`` — haploid dosage for chromosome 1.
    hds_hap2 : np.ndarray
        Shape ``(n_variants, n_samples)`` — haploid dosage for chromosome 2.
    """
    prefixes = tuple(filter_prefixes)
    sample_ids: List[str] = []
    variant_ids: List[str] = []
    hap1_rows: List[np.ndarray] = []
    hap2_rows: List[np.ndarray] = []

    hds_index: Optional[int] = None
    cached_format: Optional[str] = None
    n_total = 0
    n_kept = 0

    with _open_vcf(vcf_path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue

            if line.startswith("#CHROM") or line.startswith("#chrom"):
                cols = line.rstrip("\n\r").split("\t")
                sample_ids = cols[9:]
                logger.info("HDS parsing: %d samples in VCF header", len(sample_ids))
                continue

            cols = line.rstrip("\n\r").split("\t")
            n_total += 1
            variant_id = cols[2]

            if not variant_id.startswith(prefixes):
                continue

            # Only keep classical alleles (HLA_GENE*ALLELE pattern)
            if _parse_gene_allele(variant_id) is None:
                continue

            n_kept += 1

            fmt = cols[8]
            if fmt != cached_format:
                hds_index = _find_field_index(fmt, "HDS")
                cached_format = fmt

            n_samples = len(sample_ids)
            h1 = np.empty(n_samples, dtype=np.float64)
            h2 = np.empty(n_samples, dtype=np.float64)

            for i, gt_str in enumerate(cols[9:]):
                try:
                    hds_str = gt_str.split(":")[hds_index]
                    parts = hds_str.split(",")
                    h1[i] = float(parts[0])
                    h2[i] = float(parts[1])
                except (IndexError, ValueError):
                    h1[i] = np.nan
                    h2[i] = np.nan

            variant_ids.append(variant_id)
            hap1_rows.append(h1)
            hap2_rows.append(h2)

    logger.info(
        "HDS parsing complete: %d variants total, %d classical HLA alleles kept (%d samples)",
        n_total, n_kept, len(sample_ids),
    )

    if not variant_ids:
        empty = np.empty((0, len(sample_ids)), dtype=np.float64)
        return sample_ids, variant_ids, empty, empty

    hds_hap1 = np.vstack(hap1_rows)  # (n_variants, n_samples)
    hds_hap2 = np.vstack(hap2_rows)
    return sample_ids, variant_ids, hds_hap1, hds_hap2


def call_haplotypes(
    sample_ids: List[str],
    variant_ids: List[str],
    hds_hap1: np.ndarray,
    hds_hap2: np.ndarray,
    loci: Optional[List[str]] = None,
    resolution: str = "4digit",
) -> pd.DataFrame:
    """Construct multi-locus haplotypes from per-chromosome allele calls.

    For each sample and each chromosome, the allele with the highest
    haploid dosage at each requested locus is selected.  The per-locus
    calls are joined with ``~`` to form a haplotype string, and dosages
    (0, 1, or 2 copies) are counted per individual.

    Parameters
    ----------
    sample_ids : list of str
        Sample identifiers.
    variant_ids : list of str
        Classical HLA allele IDs from the VCF (e.g. ``HLA_A*02:01``).
    hds_hap1 : np.ndarray
        Shape ``(n_variants, n_samples)`` — haploid dosages for chromosome 1.
    hds_hap2 : np.ndarray
        Shape ``(n_variants, n_samples)`` — haploid dosages for chromosome 2.
    loci : list of str, optional
        Gene loci to include in the haplotype (default: ``A, B, C, DRB1, DQB1``).
    resolution : str
        ``"2digit"``, ``"4digit"``, or ``"both"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``sample_id`` + one column per unique haplotype.
        Values are integer dosages (0, 1, or 2).
    """
    if loci is None:
        loci = list(DEFAULT_HAPLOTYPE_LOCI)

    n_variants = len(variant_ids)
    n_samples = len(sample_ids)

    if n_variants == 0 or n_samples == 0:
        return pd.DataFrame({"sample_id": sample_ids})

    # ── Group variants by gene ──
    gene_variants: Dict[str, List[Tuple[int, str]]] = {}  # gene -> [(var_index, allele_str)]
    for vi, vid in enumerate(variant_ids):
        parsed = _parse_gene_allele(vid)
        if parsed is None:
            continue
        gene, allele = parsed
        if gene not in gene_variants:
            gene_variants[gene] = []
        gene_variants[gene].append((vi, allele))

    # Check which requested loci are available
    missing_loci = [loc for loc in loci if loc not in gene_variants]
    if missing_loci:
        logger.warning(
            "Loci not found in VCF data: %s. Available: %s",
            missing_loci, sorted(gene_variants.keys()),
        )
    active_loci = [loc for loc in loci if loc in gene_variants]

    if not active_loci:
        logger.warning("No requested loci available — returning empty haplotype matrix")
        return pd.DataFrame({"sample_id": sample_ids})

    # ── Determine which resolutions to produce ──
    if resolution == "both":
        resolutions = ["2digit", "4digit"]
    else:
        resolutions = [resolution]

    all_haplotype_dfs = []

    for res in resolutions:
        # ── Call best allele per locus per chromosome per sample ──
        hap1_calls: List[List[str]] = []  # per sample, list of locus calls
        hap2_calls: List[List[str]] = []

        for sample_j in range(n_samples):
            h1_parts = []
            h2_parts = []
            for loc in active_loci:
                var_list = gene_variants[loc]  # [(var_index, allele_str), ...]

                # Find best allele on chromosome 1
                best_h1_idx = -1
                best_h1_val = -1.0
                best_h1_allele = ""
                for vi, allele in var_list:
                    val = hds_hap1[vi, sample_j]
                    if np.isnan(val):
                        continue
                    if val > best_h1_val:
                        best_h1_val = val
                        best_h1_idx = vi
                        best_h1_allele = allele

                # Find best allele on chromosome 2
                best_h2_idx = -1
                best_h2_val = -1.0
                best_h2_allele = ""
                for vi, allele in var_list:
                    val = hds_hap2[vi, sample_j]
                    if np.isnan(val):
                        continue
                    if val > best_h2_val:
                        best_h2_val = val
                        best_h2_idx = vi
                        best_h2_allele = allele

                # Apply resolution truncation
                if res == "2digit":
                    best_h1_allele = _truncate_to_2digit(best_h1_allele)
                    best_h2_allele = _truncate_to_2digit(best_h2_allele)

                h1_parts.append(f"{loc}*{best_h1_allele}")
                h2_parts.append(f"{loc}*{best_h2_allele}")

            hap1_calls.append(h1_parts)
            hap2_calls.append(h2_parts)

        # ── Form haplotype strings and count dosages ──
        haplotype_counts: Dict[str, np.ndarray] = {}

        for sample_j in range(n_samples):
            hap1_str = "~".join(hap1_calls[sample_j])
            hap2_str = "~".join(hap2_calls[sample_j])

            for hap_str in [hap1_str, hap2_str]:
                if hap_str not in haplotype_counts:
                    haplotype_counts[hap_str] = np.zeros(n_samples, dtype=np.int32)
                haplotype_counts[hap_str][sample_j] += 1

        # Build DataFrame
        if haplotype_counts:
            hap_df = pd.DataFrame(haplotype_counts)
            hap_df.insert(0, "sample_id", sample_ids)
        else:
            hap_df = pd.DataFrame({"sample_id": sample_ids})

        all_haplotype_dfs.append(hap_df)

    # ── Merge resolutions if "both" ──
    if len(all_haplotype_dfs) == 1:
        return all_haplotype_dfs[0]
    else:
        # Merge the two DataFrames on sample_id, avoiding duplicate columns
        merged = all_haplotype_dfs[0]
        right = all_haplotype_dfs[1]
        # Avoid re-adding columns that already exist
        new_cols = [c for c in right.columns if c != "sample_id" and c not in merged.columns]
        if new_cols:
            merged = merged.merge(right[["sample_id"] + new_cols], on="sample_id", how="outer")
        return merged


def build_haplotype_dosage_matrix(
    vcf_path: str,
    loci: Optional[List[str]] = None,
    resolution: str = "4digit",
    min_freq: float = 0.01,
) -> pd.DataFrame:
    """High-level pipeline: parse VCF → call haplotypes → filter by frequency.

    Parameters
    ----------
    vcf_path : str
        Path to a VCF or VCF.GZ file containing imputed HLA data.
    loci : list of str, optional
        Gene loci to include (default: ``A, B, C, DRB1, DQB1``).
    resolution : str
        ``"2digit"``, ``"4digit"``, or ``"both"``.
    min_freq : float
        Minimum haplotype frequency (fraction of total chromosomes).
        Haplotypes below this threshold are dropped.

    Returns
    -------
    pd.DataFrame
        Columns: ``sample_id`` + one column per haplotype.
        Values are integer dosages (0, 1, or 2).
    """
    if loci is None:
        loci = list(DEFAULT_HAPLOTYPE_LOCI)

    logger.info(
        "Building haplotype dosage matrix: loci=%s, resolution=%s, min_freq=%.3f",
        loci, resolution, min_freq,
    )

    # Step 1: Extract haploid dosages
    sample_ids, variant_ids, hds_hap1, hds_hap2 = extract_haploid_dosages(
        vcf_path, filter_prefixes=("HLA_",)
    )

    if not variant_ids:
        logger.warning("No classical HLA alleles found in VCF — returning empty matrix")
        return pd.DataFrame({"sample_id": sample_ids})

    logger.info("Extracted HDS for %d classical alleles across %d samples",
                len(variant_ids), len(sample_ids))

    # Step 2: Call haplotypes
    hap_df = call_haplotypes(
        sample_ids, variant_ids, hds_hap1, hds_hap2,
        loci=loci, resolution=resolution,
    )

    # Step 3: Filter by minimum frequency
    hap_cols = [c for c in hap_df.columns if c != "sample_id"]
    if hap_cols:
        n_chromosomes = 2 * len(sample_ids)
        to_drop = []
        for col in hap_cols:
            total_copies = hap_df[col].sum()
            freq = total_copies / n_chromosomes
            if freq < min_freq:
                to_drop.append(col)
                logger.debug("Dropping haplotype %s (freq=%.4f < %.4f)", col, freq, min_freq)

        if to_drop:
            logger.info(
                "Frequency filter: dropping %d/%d haplotypes below %.1f%% threshold",
                len(to_drop), len(hap_cols), min_freq * 100,
            )
            hap_df = hap_df.drop(columns=to_drop)

    remaining = [c for c in hap_df.columns if c != "sample_id"]
    logger.info("Final haplotype matrix: %d samples × %d haplotypes",
                len(sample_ids), len(remaining))

    return hap_df
