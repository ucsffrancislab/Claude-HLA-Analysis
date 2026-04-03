"""
VCF dosage parser for imputed HLA region files.

Reads ``*.chr6.dose.vcf.gz`` (or plain ``.vcf``) files from imputation
servers (e.g. Michigan Imputation Server) and extracts per-sample dosage
values for HLA classical alleles and amino-acid variants.

The parser is pure-Python (gzip + stdlib) — no ``pysam`` dependency.
"""

import gzip
import logging
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default variant-ID prefixes to keep
DEFAULT_PREFIXES: Tuple[str, ...] = ("HLA_", "AA_")
SNP_PREFIX: str = "SNP_"


def _open_vcf(path: str):
    """Return a line iterator for a VCF, handling .gz transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _find_field_index(format_str: str, field: str) -> int:
    """Return the 0-based index of *field* in a VCF FORMAT string.

    Parameters
    ----------
    format_str : str
        Colon-delimited FORMAT field, e.g. ``GT:HDS:GP:DS``.
    field : str
        Target field name, e.g. ``DS``.

    Returns
    -------
    int
        Index of the field.

    Raises
    ------
    ValueError
        If the field is not present in the FORMAT string.
    """
    parts = format_str.split(":")
    try:
        return parts.index(field)
    except ValueError:
        raise ValueError(
            f"Field '{field}' not found in FORMAT '{format_str}'. "
            f"Available fields: {parts}"
        )


def parse_vcf_dosage(
    vcf_path: str,
    field: str = "DS",
    filter_prefixes: Optional[Sequence[str]] = None,
    include_snps: bool = False,
    log_every: int = 5000,
) -> pd.DataFrame:
    """Parse a VCF file and extract per-sample dosage values.

    Parameters
    ----------
    vcf_path : str
        Path to a ``.vcf`` or ``.vcf.gz`` file.
    field : str
        FORMAT sub-field to extract (default ``DS`` for dosage).
    filter_prefixes : sequence of str, optional
        Variant-ID prefixes to *keep*.  ``None`` uses the default
        ``("HLA_", "AA_")``.
    include_snps : bool
        If ``True``, also keep variants whose ID starts with ``SNP_``.
    log_every : int
        Log progress every *n* variants.

    Returns
    -------
    pd.DataFrame
        Rows = samples (``sample_id`` as first column), remaining columns =
        variant IDs, values = dosage (float32).
    """
    if filter_prefixes is None:
        filter_prefixes = DEFAULT_PREFIXES
    prefixes: Tuple[str, ...] = tuple(filter_prefixes)
    if include_snps:
        prefixes = prefixes + (SNP_PREFIX,)

    logger.info("Parsing VCF: %s (field=%s, prefixes=%s, include_snps=%s)",
                vcf_path, field, prefixes, include_snps)

    sample_ids: List[str] = []
    variant_ids: List[str] = []
    # Collect dosage values column-wise (one list per variant)
    dosage_columns: List[np.ndarray] = []

    n_variants_total = 0
    n_variants_kept = 0
    ds_index: Optional[int] = None
    cached_format: Optional[str] = None

    with _open_vcf(vcf_path) as fh:
        for line in fh:
            # ── Header lines ──
            if line.startswith("##"):
                continue

            if line.startswith("#CHROM") or line.startswith("#chrom"):
                cols = line.rstrip("\n\r").split("\t")
                # Columns 0-8 are fixed; 9+ are sample IDs
                sample_ids = cols[9:]
                n_samples = len(sample_ids)
                logger.info("VCF header: %d samples detected", n_samples)
                continue

            # ── Data lines ──
            cols = line.rstrip("\n\r").split("\t")
            n_variants_total += 1

            # col 2 = ID
            variant_id = cols[2]

            # Filter by prefix
            if not variant_id.startswith(prefixes):
                continue

            n_variants_kept += 1

            # Parse FORMAT to locate the target field
            fmt = cols[8]
            if fmt != cached_format:
                ds_index = _find_field_index(fmt, field)
                cached_format = fmt

            # Extract dosage from each sample (cols 9+)
            n_samples = len(sample_ids)
            dosages = np.empty(n_samples, dtype=np.float32)
            for i, genotype_str in enumerate(cols[9:]):
                try:
                    val_str = genotype_str.split(":")[ds_index]
                    dosages[i] = float(val_str)
                except (IndexError, ValueError):
                    dosages[i] = np.nan

            variant_ids.append(variant_id)
            dosage_columns.append(dosages)

            if log_every > 0 and n_variants_kept % log_every == 0:
                logger.info(
                    "  … parsed %d/%d variants (kept %d)",
                    n_variants_total, n_variants_total, n_variants_kept,
                )

    logger.info(
        "VCF parsing complete: %d variants total, %d kept (%d samples)",
        n_variants_total, n_variants_kept, len(sample_ids),
    )

    if not variant_ids:
        logger.warning("No variants matched the filter prefixes in %s", vcf_path)
        return pd.DataFrame({"sample_id": sample_ids})

    # Build DataFrame: samples × variants
    matrix = np.column_stack(dosage_columns)  # (n_samples, n_variants)
    df = pd.DataFrame(matrix, columns=variant_ids)
    df.insert(0, "sample_id", sample_ids)

    return df


def detect_dosage_format(path: str) -> str:
    """Detect whether a dosage file is CSV or VCF.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    str
        ``'vcf'`` or ``'csv'``.
    """
    base = os.path.basename(path).lower()
    if base.endswith(".vcf.gz") or base.endswith(".vcf"):
        return "vcf"
    return "csv"


def normalize_variant_id(variant_id: str) -> str:
    """Normalise a VCF variant ID to the column-name convention used
    internally by the pipeline.

    VCF IDs use ``*`` as an allele separator (e.g. ``HLA_A*01:01``),
    while the CSV convention uses ``_`` (e.g. ``HLA_A_01:01``).  This
    function converts VCF-style to CSV-style so that the rest of the
    pipeline works identically.

    Parameters
    ----------
    variant_id : str
        Raw variant ID from the VCF.

    Returns
    -------
    str
        Normalised ID.
    """
    return variant_id.replace("*", "_")


def parse_vcf_to_dosage_df(
    vcf_path: str,
    field: str = "DS",
    filter_prefixes: Optional[Sequence[str]] = None,
    include_snps: bool = False,
    normalize_ids: bool = True,
) -> pd.DataFrame:
    """High-level convenience wrapper: parse VCF and optionally normalise IDs.

    Parameters
    ----------
    vcf_path : str
        Path to VCF file.
    field : str
        FORMAT sub-field to extract.
    filter_prefixes : sequence of str, optional
        Variant-ID prefixes to keep.
    include_snps : bool
        Also keep ``SNP_*`` variants.
    normalize_ids : bool
        If ``True``, replace ``*`` with ``_`` in variant IDs.

    Returns
    -------
    pd.DataFrame
        Dosage DataFrame ready for the analysis pipeline.
    """
    df = parse_vcf_dosage(
        vcf_path,
        field=field,
        filter_prefixes=filter_prefixes,
        include_snps=include_snps,
    )
    if normalize_ids and len(df.columns) > 1:
        rename = {c: normalize_variant_id(c) for c in df.columns if c != "sample_id"}
        df = df.rename(columns=rename)
    return df
