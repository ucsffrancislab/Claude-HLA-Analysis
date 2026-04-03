"""Tests for vcf_parser module."""

import gzip
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

from hla_analysis.vcf_parser import (
    parse_vcf_dosage,
    parse_vcf_to_dosage_df,
    detect_dosage_format,
    normalize_variant_id,
    _find_field_index,
)
from hla_analysis.config import AnalysisConfig
from hla_analysis.data_loader import DataLoader


# ── Helpers to generate synthetic VCF files ──────────────────────────────

def _make_vcf_content(
    sample_ids=("S001", "S002", "S003", "S004"),
    variants=None,
    format_str="GT:HDS:GP:DS",
):
    """Build a minimal VCF string for testing.

    Each variant row has dosage values that increment by 0.1 per sample
    so we can verify correct parsing.
    """
    if variants is None:
        variants = [
            ("6", "29910338", "HLA_A*01", "A", "T"),
            ("6", "29910339", "HLA_A*02:01", "A", "T"),
            ("6", "29910500", "AA_A_9_V", "A", "T"),
            ("6", "29911000", "SNP_A_100", "A", "T"),
            ("6", "29912000", "rs12345", "A", "T"),
        ]

    lines = []
    lines.append("##fileformat=VCFv4.1")
    lines.append("##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Dosage\">")
    header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    header.extend(sample_ids)
    lines.append("\t".join(header))

    for vi, (chrom, pos, vid, ref, alt) in enumerate(variants):
        cols = [chrom, pos, vid, ref, alt, ".", "PASS", ".", format_str]
        for si in range(len(sample_ids)):
            ds = round((vi * 0.2 + si * 0.1) % 2.0, 4)
            gt = "0|0" if ds < 0.5 else ("0|1" if ds < 1.5 else "1|1")
            # Build a genotype string: GT:HDS:GP:DS
            # HDS and GP are placeholders
            entry = f"{gt}:0,0:1,0,0:{ds}"
            cols.append(entry)
        lines.append("\t".join(cols))

    return "\n".join(lines) + "\n"


@pytest.fixture
def vcf_plain(tmp_path):
    """Write a plain-text .vcf to tmp_path and return its path."""
    content = _make_vcf_content()
    path = tmp_path / "test.vcf"
    path.write_text(content)
    return str(path)


@pytest.fixture
def vcf_gzipped(tmp_path):
    """Write a gzipped .vcf.gz to tmp_path and return its path."""
    content = _make_vcf_content()
    path = tmp_path / "test.vcf.gz"
    with gzip.open(str(path), "wt") as fh:
        fh.write(content)
    return str(path)


@pytest.fixture
def vcf_and_csv(tmp_path):
    """Create a VCF and an equivalent CSV with identical dosage data.

    Returns (vcf_path, csv_path, expected_df).
    """
    sample_ids = ("S001", "S002", "S003")
    variants = [
        ("6", "100", "HLA_A*01", "A", "T"),
        ("6", "200", "HLA_B*07:02", "A", "T"),
        ("6", "300", "AA_DRB1_9_V", "A", "T"),
    ]

    # Build VCF
    vcf_content = _make_vcf_content(sample_ids=sample_ids, variants=variants)
    vcf_path = str(tmp_path / "data.vcf")
    with open(vcf_path, "w") as fh:
        fh.write(vcf_content)

    # Parse it to get the expected dosage values
    df_expected = parse_vcf_to_dosage_df(vcf_path, normalize_ids=True)

    # Write the same data as CSV
    csv_path = str(tmp_path / "data.csv")
    df_expected.to_csv(csv_path, index=False)

    return vcf_path, csv_path, df_expected


# ── Unit tests ───────────────────────────────────────────────────────────

class TestFindFieldIndex:
    def test_ds_found(self):
        assert _find_field_index("GT:HDS:GP:DS", "DS") == 3

    def test_gt_found(self):
        assert _find_field_index("GT:HDS:GP:DS", "GT") == 0

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="not found"):
            _find_field_index("GT:HDS:GP", "DS")


class TestDetectDosageFormat:
    def test_vcf_gz(self):
        assert detect_dosage_format("path/to/file.chr6.dose.vcf.gz") == "vcf"

    def test_vcf_plain(self):
        assert detect_dosage_format("data.vcf") == "vcf"

    def test_csv(self):
        assert detect_dosage_format("dosage.csv") == "csv"

    def test_tsv_as_csv(self):
        assert detect_dosage_format("dosage.tsv") == "csv"

    def test_uppercase(self):
        assert detect_dosage_format("FILE.VCF.GZ") == "vcf"


class TestNormalizeVariantId:
    def test_star_to_underscore(self):
        assert normalize_variant_id("HLA_A*01") == "HLA_A_01"

    def test_four_digit(self):
        assert normalize_variant_id("HLA_A*02:01") == "HLA_A_02:01"

    def test_aa_unchanged(self):
        assert normalize_variant_id("AA_A_9_V") == "AA_A_9_V"

    def test_snp_unchanged(self):
        assert normalize_variant_id("SNP_A_100") == "SNP_A_100"


class TestParseVcfDosage:
    """Tests for the low-level VCF parser."""

    def test_basic_plain(self, vcf_plain):
        df = parse_vcf_dosage(vcf_plain, field="DS")
        assert "sample_id" in df.columns
        # Default prefixes keep HLA_ and AA_ (3 of 5 variants)
        feature_cols = [c for c in df.columns if c != "sample_id"]
        assert len(feature_cols) == 3
        assert any(c.startswith("HLA_") for c in feature_cols)
        assert any(c.startswith("AA_") for c in feature_cols)
        # No SNP_ or rs
        assert not any(c.startswith("SNP_") for c in feature_cols)
        assert not any(c.startswith("rs") for c in feature_cols)

    def test_gzipped(self, vcf_gzipped):
        df = parse_vcf_dosage(vcf_gzipped, field="DS")
        feature_cols = [c for c in df.columns if c != "sample_id"]
        assert len(feature_cols) == 3

    def test_include_snps(self, vcf_plain):
        df = parse_vcf_dosage(vcf_plain, field="DS", include_snps=True)
        feature_cols = [c for c in df.columns if c != "sample_id"]
        assert len(feature_cols) == 4  # HLA_*2 + AA_*1 + SNP_*1

    def test_custom_prefixes(self, vcf_plain):
        df = parse_vcf_dosage(vcf_plain, field="DS",
                              filter_prefixes=["rs"])
        feature_cols = [c for c in df.columns if c != "sample_id"]
        assert len(feature_cols) == 1
        assert feature_cols[0].startswith("rs")

    def test_sample_ids_correct(self, vcf_plain):
        df = parse_vcf_dosage(vcf_plain, field="DS")
        assert list(df["sample_id"]) == ["S001", "S002", "S003", "S004"]

    def test_dosage_is_float32(self, vcf_plain):
        df = parse_vcf_dosage(vcf_plain, field="DS")
        feature_cols = [c for c in df.columns if c != "sample_id"]
        for c in feature_cols:
            assert df[c].dtype == np.float32

    def test_dosage_values_reasonable(self, vcf_plain):
        df = parse_vcf_dosage(vcf_plain, field="DS")
        feature_cols = [c for c in df.columns if c != "sample_id"]
        for c in feature_cols:
            assert df[c].between(0, 2).all()

    def test_no_matching_variants(self, vcf_plain):
        df = parse_vcf_dosage(vcf_plain, field="DS",
                              filter_prefixes=["NONEXIST_"])
        feature_cols = [c for c in df.columns if c != "sample_id"]
        assert len(feature_cols) == 0
        assert len(df) == 4  # samples still present


class TestParseVcfToDosageDf:
    """Tests for the high-level wrapper with ID normalization."""

    def test_normalize_ids(self, vcf_plain):
        df = parse_vcf_to_dosage_df(vcf_plain, normalize_ids=True)
        feature_cols = [c for c in df.columns if c != "sample_id"]
        for c in feature_cols:
            assert "*" not in c, f"Variant ID not normalised: {c}"

    def test_no_normalize(self, vcf_plain):
        df = parse_vcf_to_dosage_df(vcf_plain, normalize_ids=False)
        feature_cols = [c for c in df.columns if c != "sample_id"]
        hla_cols = [c for c in feature_cols if c.startswith("HLA_")]
        # Original VCF uses * separator
        assert any("*" in c for c in hla_cols)


class TestVcfCsvEquivalence:
    """Verify that VCF and CSV input produce identical downstream results."""

    def test_dosage_match(self, vcf_and_csv):
        """VCF-derived and CSV-derived DataFrames should match."""
        vcf_path, csv_path, expected = vcf_and_csv
        csv_df = pd.read_csv(csv_path)
        vcf_df = parse_vcf_to_dosage_df(vcf_path, normalize_ids=True)

        pd.testing.assert_frame_equal(
            vcf_df.sort_values("sample_id").reset_index(drop=True),
            csv_df.sort_values("sample_id").reset_index(drop=True),
            check_dtype=False,  # float32 vs float64 ok
        )

    def test_data_loader_auto_csv(self, vcf_and_csv, tmp_path):
        """DataLoader with format=auto picks CSV for .csv files."""
        _, csv_path, _ = vcf_and_csv
        # Make a dummy covariate file
        cov = pd.DataFrame({
            "IID": ["S001", "S002", "S003"],
            "case": [1, 0, 1],
            "age": [50, 60, 55],
            "sex": ["M", "F", "M"],
            "exclude": [0, 0, 0],
        })
        cov_path = str(tmp_path / "cov.csv")
        cov.to_csv(cov_path, index=False)

        config = AnalysisConfig(
            dosage_files=[csv_path],
            covariate_files=[cov_path],
            dataset_names=["test"],
            workers=1,
            dosage_format="auto",
        )
        loader = DataLoader(config)
        ds = loader.load_dataset(csv_path, cov_path, "test")
        assert ds["dosage"].shape[0] > 0

    def test_data_loader_auto_vcf(self, vcf_and_csv, tmp_path):
        """DataLoader with format=auto picks VCF for .vcf files."""
        vcf_path, _, _ = vcf_and_csv
        cov = pd.DataFrame({
            "IID": ["S001", "S002", "S003"],
            "case": [1, 0, 1],
            "age": [50, 60, 55],
            "sex": ["M", "F", "M"],
            "exclude": [0, 0, 0],
        })
        cov_path = str(tmp_path / "cov.csv")
        cov.to_csv(cov_path, index=False)

        config = AnalysisConfig(
            dosage_files=[vcf_path],
            covariate_files=[cov_path],
            dataset_names=["test"],
            workers=1,
            dosage_format="auto",
        )
        loader = DataLoader(config)
        ds = loader.load_dataset(vcf_path, cov_path, "test")
        assert ds["dosage"].shape[0] > 0
        assert len(ds["feature_names"]) > 0


class TestVcfFormatVariations:
    """Tests for robustness to FORMAT field variations."""

    def test_ds_not_last_field(self, tmp_path):
        """Test parsing when DS is not the last FORMAT field."""
        sample_ids = ("S1", "S2")
        lines = [
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
            # DS is at index 1 (not last)
            "6\t100\tHLA_A*01\tA\tT\t.\tPASS\t.\tGT:DS:GP\t0|0:0.5:1,0,0\t0|1:1.2:0,1,0",
        ]
        path = str(tmp_path / "custom_fmt.vcf")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

        df = parse_vcf_dosage(path, field="DS")
        assert len(df) == 2
        feature_cols = [c for c in df.columns if c != "sample_id"]
        assert len(feature_cols) == 1
        assert df[feature_cols[0]].iloc[0] == pytest.approx(0.5, abs=0.001)
        assert df[feature_cols[0]].iloc[1] == pytest.approx(1.2, abs=0.001)

    def test_varying_format_across_rows(self, tmp_path):
        """Test handling when FORMAT changes between variant rows."""
        lines = [
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
            "6\t100\tHLA_A*01\tA\tT\t.\tPASS\t.\tGT:DS\t0|0:0.5\t0|1:1.0",
            "6\t200\tHLA_B*08\tA\tT\t.\tPASS\t.\tGT:HDS:DS\t0|0:0,0:0.3\t0|1:0.5,0.5:1.5",
        ]
        path = str(tmp_path / "varying_fmt.vcf")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

        df = parse_vcf_dosage(path, field="DS")
        feature_cols = [c for c in df.columns if c != "sample_id"]
        assert len(feature_cols) == 2
        # First variant: DS at index 1
        assert df[feature_cols[0]].iloc[0] == pytest.approx(0.5, abs=0.001)
        # Second variant: DS at index 2
        assert df[feature_cols[1]].iloc[0] == pytest.approx(0.3, abs=0.001)
