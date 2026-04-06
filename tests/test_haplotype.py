"""
Comprehensive tests for the haplotype analysis module.

Tests cover HDS parsing, gene extraction, haplotype calling,
dosage counting, resolution handling, frequency filtering,
custom loci, and end-to-end integration.
"""

import gzip
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

from hla_analysis.haplotype import (
    _parse_gene_allele,
    _truncate_to_2digit,
    extract_haploid_dosages,
    call_haplotypes,
    build_haplotype_dosage_matrix,
    DEFAULT_HAPLOTYPE_LOCI,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_hds_vcf_content(
    sample_ids=("S001", "S002", "S003", "S004"),
    variants=None,
    format_str="GT:HDS:GP:DS",
):
    """Build a VCF string with realistic HDS fields for testing.

    Each variant has explicit HDS values for precise test assertions.
    """
    if variants is None:
        # Default: two alleles each for genes A, B, C, DRB1, DQB1
        variants = [
            # (chrom, pos, id, ref, alt, hds_per_sample)
            # hds_per_sample: list of (h1, h2) per sample
            ("6", "100", "HLA_A*01:01", "A", "T",
             [(0.99, 0.01), (0.01, 0.99), (0.99, 0.99), (0.01, 0.01)]),
            ("6", "101", "HLA_A*02:01", "A", "T",
             [(0.01, 0.99), (0.99, 0.01), (0.01, 0.01), (0.99, 0.99)]),
            ("6", "200", "HLA_B*07:02", "A", "T",
             [(0.98, 0.02), (0.02, 0.98), (0.98, 0.98), (0.02, 0.02)]),
            ("6", "201", "HLA_B*08:01", "A", "T",
             [(0.02, 0.98), (0.98, 0.02), (0.02, 0.02), (0.98, 0.98)]),
            ("6", "300", "HLA_C*07:01", "A", "T",
             [(0.97, 0.03), (0.03, 0.97), (0.97, 0.97), (0.03, 0.03)]),
            ("6", "301", "HLA_C*07:02", "A", "T",
             [(0.03, 0.97), (0.97, 0.03), (0.03, 0.03), (0.97, 0.97)]),
            ("6", "400", "HLA_DRB1*15:01", "A", "T",
             [(0.96, 0.04), (0.04, 0.96), (0.96, 0.96), (0.04, 0.04)]),
            ("6", "401", "HLA_DRB1*03:01", "A", "T",
             [(0.04, 0.96), (0.96, 0.04), (0.04, 0.04), (0.96, 0.96)]),
            ("6", "500", "HLA_DQB1*06:02", "A", "T",
             [(0.95, 0.05), (0.05, 0.95), (0.95, 0.95), (0.05, 0.05)]),
            ("6", "501", "HLA_DQB1*02:01", "A", "T",
             [(0.05, 0.95), (0.95, 0.05), (0.05, 0.05), (0.95, 0.95)]),
            # Also include a non-classical variant (should be ignored by haplotype module)
            ("6", "600", "AA_A_9_V", "A", "T",
             [(0.50, 0.50), (0.50, 0.50), (0.50, 0.50), (0.50, 0.50)]),
        ]

    lines = []
    lines.append("##fileformat=VCFv4.1")
    lines.append('##FORMAT=<ID=HDS,Number=2,Type=Float,Description="Haploid Dosages">')
    lines.append('##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">')
    header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    header.extend(sample_ids)
    lines.append("\t".join(header))

    for var in variants:
        chrom, pos, vid, ref, alt, hds_list = var
        cols = [chrom, pos, vid, ref, alt, ".", "PASS", ".", format_str]
        for si in range(len(sample_ids)):
            h1, h2 = hds_list[si]
            ds = h1 + h2
            gt = "0|0" if ds < 0.5 else ("0|1" if ds < 1.5 else "1|1")
            entry = f"{gt}:{h1:.2f},{h2:.2f}:1,0,0:{ds:.2f}"
            cols.append(entry)
        lines.append("\t".join(cols))

    return "\n".join(lines) + "\n"


@pytest.fixture
def hds_vcf_plain(tmp_path):
    """Write a VCF with HDS fields to tmp_path."""
    content = _make_hds_vcf_content()
    path = tmp_path / "hds_test.vcf"
    path.write_text(content)
    return str(path)


@pytest.fixture
def hds_vcf_gzipped(tmp_path):
    """Write a gzipped VCF with HDS fields."""
    content = _make_hds_vcf_content()
    path = tmp_path / "hds_test.vcf.gz"
    with gzip.open(str(path), "wt") as fh:
        fh.write(content)
    return str(path)


# ── Test gene extraction ─────────────────────────────────────────────────


class TestParseGeneAllele:
    """Test _parse_gene_allele for extracting gene and allele from variant IDs."""

    def test_gene_a(self):
        result = _parse_gene_allele("HLA_A*02:01")
        assert result == ("A", "02:01")

    def test_gene_b(self):
        result = _parse_gene_allele("HLA_B*07:02")
        assert result == ("B", "07:02")

    def test_gene_c(self):
        result = _parse_gene_allele("HLA_C*07:01")
        assert result == ("C", "07:01")

    def test_gene_drb1(self):
        result = _parse_gene_allele("HLA_DRB1*15:01")
        assert result == ("DRB1", "15:01")

    def test_gene_dqb1(self):
        result = _parse_gene_allele("HLA_DQB1*06:02")
        assert result == ("DQB1", "06:02")

    def test_gene_dpa1(self):
        result = _parse_gene_allele("HLA_DPA1*01:03")
        assert result == ("DPA1", "01:03")

    def test_gene_dpb1(self):
        result = _parse_gene_allele("HLA_DPB1*04:01")
        assert result == ("DPB1", "04:01")

    def test_two_digit(self):
        result = _parse_gene_allele("HLA_A*02")
        assert result == ("A", "02")

    def test_amino_acid_returns_none(self):
        assert _parse_gene_allele("AA_A_9_V") is None

    def test_snp_returns_none(self):
        assert _parse_gene_allele("SNP_A_100") is None

    def test_random_string_returns_none(self):
        assert _parse_gene_allele("rs12345") is None

    def test_dqа1(self):
        result = _parse_gene_allele("HLA_DQA1*05:01")
        assert result == ("DQA1", "05:01")


class TestTruncateTo2digit:
    """Test allele resolution truncation."""

    def test_four_to_two(self):
        assert _truncate_to_2digit("02:01") == "02"

    def test_already_two(self):
        assert _truncate_to_2digit("02") == "02"

    def test_six_digit(self):
        assert _truncate_to_2digit("02:01:01") == "02"


# ── Test HDS parsing ────────────────────────────────────────────────────


class TestExtractHaploidDosages:
    """Test VCF HDS field extraction."""

    def test_basic_parsing(self, hds_vcf_plain):
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        assert len(sample_ids) == 4
        # Should only keep HLA_* classical alleles (10 of 11 variants)
        assert len(variant_ids) == 10
        assert hds1.shape == (10, 4)
        assert hds2.shape == (10, 4)

    def test_gzipped(self, hds_vcf_gzipped):
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_gzipped)
        assert len(variant_ids) == 10

    def test_hds_values_correct(self, hds_vcf_plain):
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        # First variant (HLA_A*01:01), first sample: h1=0.99, h2=0.01
        idx_a01 = variant_ids.index("HLA_A*01:01")
        assert hds1[idx_a01, 0] == pytest.approx(0.99, abs=0.01)
        assert hds2[idx_a01, 0] == pytest.approx(0.01, abs=0.01)

    def test_excludes_non_classical(self, hds_vcf_plain):
        _, variant_ids, _, _ = extract_haploid_dosages(hds_vcf_plain)
        assert "AA_A_9_V" not in variant_ids
        assert all(v.startswith("HLA_") for v in variant_ids)

    def test_sample_ids(self, hds_vcf_plain):
        sample_ids, _, _, _ = extract_haploid_dosages(hds_vcf_plain)
        assert sample_ids == ["S001", "S002", "S003", "S004"]

    def test_empty_vcf(self, tmp_path):
        """VCF with no HLA variants returns empty arrays."""
        content = _make_hds_vcf_content(
            variants=[("6", "600", "rs12345", "A", "T",
                       [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])]
        )
        path = str(tmp_path / "empty.vcf")
        with open(path, "w") as f:
            f.write(content)
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(path)
        assert len(variant_ids) == 0
        assert hds1.shape[0] == 0
        assert hds2.shape[0] == 0


# ── Test haplotype calling ───────────────────────────────────────────────


class TestCallHaplotypes:
    """Test haplotype construction from HDS data."""

    def test_basic_haplotype_calling(self, hds_vcf_plain):
        """Verify correct allele assignment per chromosome."""
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B"], resolution="4digit")

        assert "sample_id" in hap_df.columns
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        assert len(hap_cols) > 0

        # Sample S001: hap1 should have A*01:01, B*07:02; hap2 should have A*02:01, B*08:01
        s001 = hap_df[hap_df["sample_id"] == "S001"]
        assert s001["A*01:01~B*07:02"].values[0] == 1
        assert s001["A*02:01~B*08:01"].values[0] == 1

    def test_heterozygous_haplotype_dosage(self, hds_vcf_plain):
        """Individual with different haplotypes on each chromosome -> dosage 1 each."""
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B"], resolution="4digit")

        # S001 and S002 are heterozygous (different haplotypes per chromosome)
        s001 = hap_df[hap_df["sample_id"] == "S001"]
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        for col in hap_cols:
            val = s001[col].values[0]
            assert val in [0, 1], f"Expected 0 or 1 for heterozygous, got {val}"

    def test_homozygous_haplotype_dosage(self, hds_vcf_plain):
        """Same haplotype on both chromosomes -> dosage = 2."""
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B"], resolution="4digit")

        # S003: both chromosomes have high dosage for allele 1 of each gene
        # hap1: A*01:01, B*07:02; hap2: A*01:01, B*07:02
        s003 = hap_df[hap_df["sample_id"] == "S003"]
        assert s003["A*01:01~B*07:02"].values[0] == 2

    def test_dosage_sums_to_two(self, hds_vcf_plain):
        """Each individual should have exactly 2 total haplotype copies."""
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B"], resolution="4digit")

        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        row_sums = hap_df[hap_cols].sum(axis=1)
        np.testing.assert_array_equal(row_sums.values, np.full(len(hap_df), 2))

    def test_5locus_haplotypes(self, hds_vcf_plain):
        """Full 5-locus haplotype construction."""
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B", "C", "DRB1", "DQB1"],
                                 resolution="4digit")

        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        # Each haplotype should have 5 loci joined by ~
        for col in hap_cols:
            assert col.count("~") == 4, f"Expected 4 tildes in {col}"

    def test_empty_variants(self):
        """Empty variant list returns sample_id only."""
        hap_df = call_haplotypes(["S1", "S2"], [], np.empty((0, 2)), np.empty((0, 2)))
        assert list(hap_df.columns) == ["sample_id"]
        assert len(hap_df) == 2


# ── Test resolution handling ─────────────────────────────────────────────


class TestResolution:
    """Test 2-digit, 4-digit, and both resolution modes."""

    def test_2digit_resolution(self, hds_vcf_plain):
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B"], resolution="2digit")

        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        # 2-digit: no colon in allele names
        for col in hap_cols:
            parts = col.split("~")
            for part in parts:
                gene, allele = part.split("*")
                assert ":" not in allele, f"2-digit should not have colon: {col}"

    def test_4digit_resolution(self, hds_vcf_plain):
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B"], resolution="4digit")

        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        for col in hap_cols:
            parts = col.split("~")
            for part in parts:
                gene, allele = part.split("*")
                assert ":" in allele, f"4-digit should have colon: {col}"

    def test_both_resolution(self, hds_vcf_plain):
        """'both' should produce columns at both resolutions."""
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B"], resolution="both")

        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        has_2digit = any(":" not in col for col in hap_cols)
        has_4digit = any(":" in col for col in hap_cols)
        assert has_2digit, "Should have 2-digit haplotypes"
        assert has_4digit, "Should have 4-digit haplotypes"

    def test_2digit_merges_alleles(self, hds_vcf_plain):
        """2-digit resolution: different 4-digit alleles at same 2-digit should merge."""
        # Construct a VCF where two samples differ only at 4-digit level
        variants = [
            ("6", "100", "HLA_A*02:01", "A", "T",
             [(0.99, 0.01), (0.01, 0.01)]),
            ("6", "101", "HLA_A*02:02", "A", "T",
             [(0.01, 0.99), (0.99, 0.99)]),
        ]
        content = _make_hds_vcf_content(
            sample_ids=("S1", "S2"), variants=variants
        )
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            sids, vids, h1, h2 = extract_haploid_dosages(path)
            hap_df = call_haplotypes(sids, vids, h1, h2,
                                     loci=["A"], resolution="2digit")
            hap_cols = [c for c in hap_df.columns if c != "sample_id"]
            # Both chromosomes of S1 should call A*02 -> dosage 2
            # S2 also both A*02 -> dosage 2
            assert "A*02" in hap_cols
            assert hap_df["A*02"].tolist() == [2, 2]
        finally:
            os.unlink(path)


# ── Test frequency filtering ────────────────────────────────────────────


class TestFrequencyFiltering:
    """Test that rare haplotypes are filtered by minimum frequency."""

    def test_rare_haplotype_dropped(self, hds_vcf_plain):
        """With 4 samples (8 chromosomes), a haplotype appearing once = 12.5% freq.
        Setting min_freq=0.15 should drop it."""
        hap_df = build_haplotype_dosage_matrix(
            hds_vcf_plain, loci=["A", "B"], resolution="4digit", min_freq=0.15
        )
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        # Only haplotypes with freq >= 15% should survive
        for col in hap_cols:
            total = hap_df[col].sum()
            freq = total / (2 * len(hap_df))
            assert freq >= 0.15, f"Haplotype {col} has freq {freq:.3f} < 0.15"

    def test_no_filtering_at_zero(self, hds_vcf_plain):
        """min_freq=0 should keep all haplotypes."""
        hap_df = build_haplotype_dosage_matrix(
            hds_vcf_plain, loci=["A", "B"], resolution="4digit", min_freq=0.0
        )
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        assert len(hap_cols) > 0

    def test_high_threshold_drops_all(self, hds_vcf_plain):
        """Very high min_freq should drop all haplotypes."""
        hap_df = build_haplotype_dosage_matrix(
            hds_vcf_plain, loci=["A", "B"], resolution="4digit", min_freq=0.99
        )
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        # With only 4 samples, no haplotype can reach 99% freq
        # unless all 8 chromosomes carry it, so likely 0
        assert len(hap_cols) == 0


# ── Test custom loci ─────────────────────────────────────────────────────


class TestCustomLoci:
    """Test haplotype construction with different locus combinations."""

    def test_single_locus(self, hds_vcf_plain):
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A"], resolution="4digit")
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        for col in hap_cols:
            assert "~" not in col, "Single locus should have no tilde"

    def test_two_locus(self, hds_vcf_plain):
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B"], resolution="4digit")
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        for col in hap_cols:
            assert col.count("~") == 1

    def test_three_locus(self, hds_vcf_plain):
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "B", "C"], resolution="4digit")
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        for col in hap_cols:
            assert col.count("~") == 2

    def test_missing_locus_warning(self, hds_vcf_plain):
        """Requesting a locus not in VCF should warn and skip it."""
        sample_ids, variant_ids, hds1, hds2 = extract_haploid_dosages(hds_vcf_plain)
        hap_df = call_haplotypes(sample_ids, variant_ids, hds1, hds2,
                                 loci=["A", "NONEXIST"], resolution="4digit")
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        # Should only have A locus (NONEXIST skipped)
        for col in hap_cols:
            assert "~" not in col  # single locus


# ── Test integration ─────────────────────────────────────────────────────


class TestBuildHaplotypeDosageMatrix:
    """End-to-end integration tests for build_haplotype_dosage_matrix."""

    def test_output_structure(self, hds_vcf_plain):
        hap_df = build_haplotype_dosage_matrix(hds_vcf_plain, min_freq=0.0)
        assert "sample_id" in hap_df.columns
        assert len(hap_df) == 4  # 4 samples
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        assert len(hap_cols) > 0

    def test_dosage_values_valid(self, hds_vcf_plain):
        """All dosage values should be 0, 1, or 2."""
        hap_df = build_haplotype_dosage_matrix(hds_vcf_plain, min_freq=0.0)
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        for col in hap_cols:
            vals = hap_df[col].unique()
            assert all(v in [0, 1, 2] for v in vals), f"Invalid dosage in {col}: {vals}"

    def test_row_sums(self, hds_vcf_plain):
        """Each sample should have exactly 2 haplotype copies total."""
        hap_df = build_haplotype_dosage_matrix(hds_vcf_plain, min_freq=0.0)
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        row_sums = hap_df[hap_cols].sum(axis=1)
        np.testing.assert_array_equal(row_sums.values, np.full(len(hap_df), 2))

    def test_custom_loci_and_resolution(self, hds_vcf_plain):
        hap_df = build_haplotype_dosage_matrix(
            hds_vcf_plain, loci=["A", "B"], resolution="2digit", min_freq=0.0
        )
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        for col in hap_cols:
            assert col.count("~") == 1  # two loci
            for part in col.split("~"):
                _, allele = part.split("*")
                assert ":" not in allele  # 2-digit

    def test_compatible_with_analysis_pipeline(self, hds_vcf_plain):
        """Verify the output can be converted to the dataset dict format."""
        hap_df = build_haplotype_dosage_matrix(hds_vcf_plain, min_freq=0.0)
        hap_features = [c for c in hap_df.columns if c != "sample_id"]

        # Simulate creating a dataset dict as __main__.py would
        dosage_matrix = hap_df[hap_features].values.astype(np.float32)
        dataset = {
            "dataset_name": "test",
            "covariates": pd.DataFrame({"sample_id": hap_df["sample_id"]}),
            "dosage": dosage_matrix,
            "feature_names": hap_features,
            "sample_ids": hap_df["sample_id"].tolist(),
        }

        assert dataset["dosage"].shape == (4, len(hap_features))
        assert dataset["dosage"].dtype == np.float32

    def test_vcf_with_no_hla(self, tmp_path):
        """VCF with only non-HLA variants returns empty columns."""
        variants = [
            ("6", "100", "rs12345", "A", "T",
             [(0.5, 0.5), (0.5, 0.5)]),
        ]
        content = _make_hds_vcf_content(sample_ids=("S1", "S2"), variants=variants)
        path = str(tmp_path / "no_hla.vcf")
        with open(path, "w") as f:
            f.write(content)

        hap_df = build_haplotype_dosage_matrix(path)
        hap_cols = [c for c in hap_df.columns if c != "sample_id"]
        assert len(hap_cols) == 0
