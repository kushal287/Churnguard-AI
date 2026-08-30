"""
End-to-End Integration Test for Training and Inference Pipeline.
"""

from pathlib import Path
import pytest
from src.pipeline.train_pipeline import TrainingPipeline
from src.models.custom_logistic_regression import CustomLogisticRegression
from src.data.preprocessor import DataPreprocessor
from src.explainability.individual_explainer import IndividualExplainer
from src.explainability.retention_playbook import RetentionPlaybook


class TestEndToEndPipeline:
    """Validate full workflow from training to inference and explainability."""

    def test_complete_pipeline_execution(self):
        pipeline = TrainingPipeline(random_state=42)
        results = pipeline.run()

        # Assert results structure
        assert "benchmark_results" in results
        assert "optimal_threshold" in results
        assert "feature_importance_summary" in results

        bm = results["benchmark_results"]
        assert "custom_numpy_model" in bm
        assert "sklearn_benchmark_model" in bm

        custom_test_metrics = bm["custom_numpy_model"]["test_metrics"]
        assert custom_test_metrics["roc_auc"] > 0.80, f"ROC-AUC too low: {custom_test_metrics['roc_auc']}"
        assert custom_test_metrics["f1_score"] > 0.50, f"F1-score too low: {custom_test_metrics['f1_score']}"

        # Assert saved artifacts exist
        assert Path("artifacts/custom_logistic_model.npz").exists()
        assert Path("artifacts/preprocessor_pipeline.joblib").exists()
        assert Path("artifacts/benchmark_results.json").exists()

        # Assert generated figures exist
        assert Path("reports/figures/training_loss_curves.png").exists()
        assert Path("reports/figures/roc_pr_curves_comparison.png").exists()
        assert Path("reports/figures/confusion_matrices.png").exists()
        assert Path("reports/figures/odds_ratio_feature_importance.png").exists()

    def test_inference_and_playbook_integration(self):
        """Test loading artifacts and running individual customer explanation."""
        model = CustomLogisticRegression.load("artifacts/custom_logistic_model.npz")
        preprocessor = DataPreprocessor.load("artifacts/preprocessor_pipeline.joblib")

        sample_customer = {
            "customerID": "TEST-CUST-001",
            "gender": "Female",
            "SeniorCitizen": "0",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
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
            "MonthlyCharges": 89.5,
            "TotalCharges": 179.0,
        }

        X_cust = preprocessor.transform_single_record(sample_customer)
        proba = model.predict_proba(X_cust)
        churn_prob = float(proba[0, 1])

        assert 0.0 <= churn_prob <= 1.0

        explainer = IndividualExplainer(
            feature_names=preprocessor.get_feature_names(),
            weights=model.weights,
            bias=model.bias,
        )
        explanation = explainer.explain_instance(X_cust, top_n=5)

        assert "risk_drivers" in explanation
        assert "retention_anchors" in explanation
        assert len(explanation["waterfall_steps"]) > 0

        playbook = RetentionPlaybook()
        recommendations = playbook.generate_recommendations(
            sample_customer, churn_prob, explanation["risk_drivers"]
        )

        assert len(recommendations) > 0
        assert any(r["priority"] == "HIGH" for r in recommendations)
