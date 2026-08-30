"""
Comprehensive Unit Tests for Explainability, Log-Odds Reconstruction, and Retention Playbook.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config.config import CUSTOM_MODEL_PATH, FINAL_RESULTS_PATH, PREPROCESSOR_PATH
from src.data.preprocessor import DataPreprocessor
from src.explainability.feature_importance import GlobalExplainer
from src.explainability.individual_explainer import IndividualExplainer
from src.explainability.retention_playbook import RetentionPlaybook
from src.models.custom_logistic_regression import CustomLogisticRegression


class TestExplainabilityAndPlaybook:
    """Test suite validating explainability fidelity, non-causal formatting, and deterministic playbooks."""

    @pytest.fixture
    def loaded_system(self):
        """Load trained model, preprocessor, and explainers."""
        assert CUSTOM_MODEL_PATH.exists(), "Model artifact missing"
        assert PREPROCESSOR_PATH.exists(), "Preprocessor artifact missing"
        model = CustomLogisticRegression.load(CUSTOM_MODEL_PATH)
        preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
        feature_names = preprocessor.get_feature_names()
        global_explainer = GlobalExplainer(feature_names, model.weights, model.bias)
        individual_explainer = IndividualExplainer(feature_names, model.weights, model.bias)
        playbook = RetentionPlaybook(threshold=0.58)
        return model, preprocessor, global_explainer, individual_explainer, playbook

    def test_odds_ratio_mathematical_calculation(self, loaded_system):
        """Verify that Odds Ratio is strictly exp(w) for all features."""
        model, _, global_explainer, _, _ = loaded_system
        df_summary = global_explainer.get_summary_dataframe()

        # Check exp(w) for all rows
        for _, row in df_summary.iterrows():
            w = row["Weight (Log-Odds)"]
            or_val = row["Odds Ratio"]
            expected_or = np.exp(w)
            assert abs(or_val - expected_or) < 1e-3, f"Odds ratio mismatch for {row['Feature']}: {or_val} vs {expected_or}"

    def test_standardized_numerical_feature_interpretation(self, loaded_system):
        """Verify that standardized numerical features explicitly mention standard deviation and hold other inputs constant."""
        _, _, global_explainer, _, _ = loaded_system
        df_summary = global_explainer.get_summary_dataframe()
        df_total_charges = df_summary[df_summary["Feature"] == "TotalCharges"]
        assert not df_total_charges.empty

        text = df_total_charges.iloc[0]["Statistical Interpretation"]
        assert "1-standard-deviation increase" in text
        assert "holding other model inputs constant" in text
        assert "associated with" in text
        assert "causes" not in text.lower()

    def test_categorical_feature_reference_category_tracking(self, loaded_system):
        """Verify that one-hot categorical features reference their omitted baseline group."""
        _, _, global_explainer, _, _ = loaded_system
        df_summary = global_explainer.get_summary_dataframe()
        df_contract = df_summary[df_summary["Feature"] == "Contract_Two year"]
        assert not df_contract.empty

        text = df_contract.iloc[0]["Statistical Interpretation"]
        assert "Month-to-month Contract" in text
        assert "associated with" in text
        assert "holding other model inputs constant" in text

    def test_log_odds_and_probability_exact_reconstruction(self, loaded_system):
        """Verify that individual customer log-odds exactly reconstructs the model probability."""
        model, preprocessor, _, individual_explainer, _ = loaded_system

        sample_customer = {
            "gender": "Female",
            "SeniorCitizen": "0",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.50,
            "TotalCharges": 286.50,
        }

        X_vec = preprocessor.transform_single_record(sample_customer)
        direct_prob = float(model.predict_proba(X_vec)[0, 1])

        explanation = individual_explainer.explain_instance(X_vec)
        recon = explanation["mathematical_reconstruction"]

        # Check linear reconstruction: z = b + sum(w * x)
        assert abs(recon["direct_probability"] - direct_prob) < 1e-7
        assert abs(recon["reconstructed_probability"] - direct_prob) < 1e-7
        assert recon["discrepancy"] < 1e-7

    def test_risk_tier_classification_threshold_boundary(self, loaded_system):
        """Verify that risk classification strictly respects the 0.58 validation threshold."""
        _, _, _, _, playbook = loaded_system

        # High risk: >= 0.58
        high_risk = playbook.classify_risk(0.5800)
        assert high_risk["tier_code"] == "HIGH"

        high_risk_above = playbook.classify_risk(0.7500)
        assert high_risk_above["tier_code"] == "HIGH"

        # Medium risk: 0.40 <= p < 0.58
        med_risk = playbook.classify_risk(0.5799)
        assert med_risk["tier_code"] == "MEDIUM"

        med_risk_lower = playbook.classify_risk(0.4000)
        assert med_risk_lower["tier_code"] == "MEDIUM"

        # Low risk: < 0.40
        low_risk = playbook.classify_risk(0.3999)
        assert low_risk["tier_code"] == "LOW"

    def test_retention_playbook_deterministic_triggers(self, loaded_system):
        """Verify that retention actions fire deterministically on modeled risk triggers."""
        _, _, _, _, playbook = loaded_system

        # Customer on Month-to-month + Fiber optic + Electronic check
        customer_m2m = {
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 90.0,
            "tenure": 2,
        }
        recs = playbook.generate_recommendations(customer_m2m, churn_probability=0.72)
        actions = [r["action"] for r in recs]

        assert "Annual Loyalty Commitment Offer" in actions
        assert "Complimentary Tech & Security Guard Bundle" in actions
        assert "Automated Auto-Pay Migration Bonus" in actions
        assert "Dedicated Customer Success Concierge Check-in" in actions
        assert "Account Plan Optimization & Value Review" in actions

    def test_business_value_arithmetic_consistency(self):
        """Verify that the business value calculation matches 211 * 350 - 190 * 50 = $64,350."""
        from src.evaluation.metrics import compute_financial_utility

        # Mock test set confusion matrix: TP=211, FP=190, TN=586, FN=70 (Total=1057)
        y_true = np.concatenate([np.ones(211 + 70), np.zeros(586 + 190)])
        y_pred = np.concatenate([
            np.ones(211), np.zeros(70),  # TP=211, FN=70
            np.zeros(586), np.ones(190),  # TN=586, FP=190
        ])

        fin = compute_financial_utility(y_true, y_pred)
        assert fin["true_positives"] == 211.0
        assert fin["false_positives"] == 190.0
        assert fin["net_retention_savings"] == 64350.0
        assert abs(fin["roi_per_customer"] - (64350.0 / 1057.0)) < 1e-4
        assert round(fin["roi_per_customer"], 2) == 60.88

    def test_final_results_json_loaded_and_verified(self):
        """Verify that final_results.json exists, is valid JSON, and holds frozen values."""
        assert FINAL_RESULTS_PATH.exists(), "final_results.json missing"
        with open(FINAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            res = json.load(f)

        assert res["metadata"]["experiment_status"] == "FROZEN_FINAL_AUTHORITATIVE"
        c50 = res["custom_numpy_model_test_metrics"]["at_default_threshold_0_50"]
        assert c50["roc_auc"] == 0.8451590417140551
        assert c50["illustrative_net_retention_savings"] == 64350.0
        assert res["threshold_selection"]["selected_optimal_threshold"] == 0.5800000000000001
