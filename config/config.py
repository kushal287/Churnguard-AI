"""
Global Configuration for ChurnGuard AI
Explainable Customer Churn Prediction & Retention Intelligence Platform
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data File Paths
RAW_DATA_PATH = RAW_DATA_DIR / "telco_customer_churn.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
SCHEMA_METADATA_PATH = PROCESSED_DATA_DIR / "schema_metadata.json"

# Artifact Paths
CUSTOM_MODEL_PATH = ARTIFACTS_DIR / "custom_logistic_model.npz"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor_pipeline.joblib"
BENCHMARK_RESULTS_PATH = ARTIFACTS_DIR / "benchmark_results.json"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
FINAL_RESULTS_PATH = ARTIFACTS_DIR / "final_results.json"

# Reproducibility Seed
RANDOM_SEED = 42

# Data Splitting Parameters
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Dataset Column Definitions
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# Model Hyperparameters for Custom Logistic Regression
DEFAULT_HYPERPARAMS = {
    "learning_rate": 0.05,
    "max_iter": 1500,
    "l2_lambda": 0.01,
    "momentum": 0.9,
    "batch_size": 64,
    "early_stopping": True,
    "patience": 50,
    "tolerance": 1e-6,
    "use_class_weights": True,
    "fit_intercept": True
}

# Business Retention Cost-Utility Matrix (in USD)
BUSINESS_COST_MATRIX = {
    "cost_false_negative": 500.0,   # Lost Customer Lifetime Value (LTV) when churn is missed
    "cost_false_positive": 50.0,    # Unnecessary retention incentive / discount wasted on loyal customer
    "gain_true_positive": 350.0,    # Net recovered LTV through proactive retention (Value saved - retention cost)
    "gain_true_negative": 0.0       # Standard organic continuation
}

# Decision Threshold
DEFAULT_DECISION_THRESHOLD = 0.50
