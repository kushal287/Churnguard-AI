"""
Unit and Integration Tests for Artifact Integrity and Checksum Verification.
"""

from pathlib import Path
import json
import pytest

from config.config import ARTIFACTS_DIR, FINAL_RESULTS_PATH
from src.utils.integrity import EXPECTED_ARTIFACTS, verify_artifact_integrity


class TestArtifactIntegrity:
    """Test suite for validating the presence and format of authoritative experiment artifacts."""

    def test_all_expected_artifacts_exist_and_pass_validation(self):
        """Verify that all authoritative frozen experiment artifacts exist and pass verification."""
        all_valid, issues = verify_artifact_integrity()
        assert all_valid is True, f"Integrity check failed with issues: {issues}"
        assert len(issues) == 0

    def test_final_results_json_structure(self):
        """Verify that final_results.json contains all required top-level keys and valid benchmark metrics."""
        assert FINAL_RESULTS_PATH.exists()
        with open(FINAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "dataset_info" in data
        assert "split_protocol" in data
        assert "custom_numpy_model_test_metrics" in data
        assert "sklearn_benchmark_test_metrics" in data

        metrics = data["custom_numpy_model_test_metrics"]["at_default_threshold_0_50"]
        assert metrics["roc_auc"] == 0.8451590417140551
        assert metrics["illustrative_net_retention_savings"] == 64350.0

    def test_missing_artifact_detection(self, tmp_path, monkeypatch):
        """Verify that a missing artifact is detected with an informative error message."""
        dummy_artifacts = {
            "missing_file.npz": tmp_path / "non_existent.npz"
        }
        monkeypatch.setattr("src.utils.integrity.EXPECTED_ARTIFACTS", dummy_artifacts)

        all_valid, issues = verify_artifact_integrity()
        assert all_valid is False
        assert len(issues) == 1
        assert "Missing required artifact" in issues[0]
