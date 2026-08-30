"""
Artifact Integrity and Validation Utility for ChurnGuard AI.
Verifies existence and schema integrity of authoritative frozen artifacts.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import json
import logging

from config.config import (
    ARTIFACTS_DIR,
    BENCHMARK_RESULTS_PATH,
    CUSTOM_MODEL_PATH,
    FEATURE_NAMES_PATH,
    FINAL_RESULTS_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)

logger = logging.getLogger(__name__)

# Expected frozen artifacts
EXPECTED_ARTIFACTS = {
    "custom_logistic_model.npz": CUSTOM_MODEL_PATH,
    "preprocessor_pipeline.joblib": PREPROCESSOR_PATH,
    "final_results.json": FINAL_RESULTS_PATH,
    "benchmark_results.json": BENCHMARK_RESULTS_PATH,
    "feature_names.json": FEATURE_NAMES_PATH,
}


def verify_artifact_integrity() -> Tuple[bool, List[str]]:
    """
    Verify that all authoritative frozen experiment artifacts exist and are readable.
    
    Returns:
        Tuple of (all_valid: bool, issues: List[str])
    """
    issues = []

    for name, path in EXPECTED_ARTIFACTS.items():
        if not path.exists():
            issues.append(f"Missing required artifact: {name} at {path}")
            continue

        if path.stat().st_size == 0:
            issues.append(f"Artifact {name} is empty (0 bytes).")
            continue

        # Format-specific validation
        if name.endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, (dict, list)) or len(data) == 0:
                        issues.append(f"JSON artifact {name} does not contain valid non-empty data.")
            except Exception as e:
                issues.append(f"Corrupted JSON artifact {name}: {str(e)}")

        elif name.endswith(".npz"):
            try:
                import numpy as np
                loaded = np.load(path)
                if "weights" not in loaded or "bias" not in loaded:
                    issues.append(f"NPZ artifact {name} missing 'weights' or 'bias' keys.")
            except Exception as e:
                issues.append(f"Corrupted NPZ artifact {name}: {str(e)}")

    all_valid = len(issues) == 0
    return all_valid, issues
