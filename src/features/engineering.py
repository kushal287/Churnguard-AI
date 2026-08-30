"""
Domain Feature Engineering Module for ChurnGuard AI.
Constructs behavioral, contractual, and service-based interaction features.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for customer churn analytics."""

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply deterministic domain feature transformations.
        
        Engineered Features:
        1. tenure_cohort: Binned lifecycle stage
        2. charges_ratio: MonthlyCharges / (TotalCharges + 1.0)
        3. avg_monthly_diff: MonthlyCharges - (TotalCharges / (tenure + 1))
        4. total_services_count: Number of add-on products active
        5. protection_services_count: Security, Backup, DeviceProtection, TechSupport
        6. streaming_services_count: StreamingTV, StreamingMovies
        7. has_internet: Boolean indicator for DSL or Fiber Optic
        8. is_solo_senior: Senior Citizen with no Partner & no Dependents
        9. high_risk_fiber_m2m: Month-to-month contract with Fiber Optic
        """
        data = df.copy()

        # Safe local numeric series to guarantee float operations
        tc_numeric = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)
        tenure_num = pd.to_numeric(data["tenure"], errors="coerce").fillna(0).astype(int)
        mc_numeric = pd.to_numeric(data["MonthlyCharges"], errors="coerce").fillna(0.0)

        # 1. Tenure Cohort Categories
        # Bin tenure into standard business brackets: 0-12m (New), 13-24m (Early), 25-48m (Established), 49-72m (Loyal)
        data["tenure_cohort"] = pd.cut(
            tenure_num,
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"]
        ).astype(str)

        # 2. Monthly to Total Ratio (High ratio = new or recent price hike)
        data["monthly_to_total_ratio"] = mc_numeric / (tc_numeric + 1.0)

        # 3. Monthly charges difference from historical average
        historical_avg = tc_numeric / (tenure_num + 1.0)
        data["monthly_charge_discrepancy"] = mc_numeric - historical_avg

        # 4. Service counts
        service_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity", 
            "OnlineBackup", "DeviceProtection", "TechSupport", 
            "StreamingTV", "StreamingMovies"
        ]
        
        # Count affirmative services
        data["total_services_count"] = 0
        for col in service_cols:
            if col in data.columns:
                data["total_services_count"] += (data[col] == "Yes").astype(int)

        # 5. Protection services count
        prot_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        data["protection_services_count"] = 0
        for col in prot_cols:
            if col in data.columns:
                data["protection_services_count"] += (data[col] == "Yes").astype(int)

        # 6. Streaming services count
        stream_cols = ["StreamingTV", "StreamingMovies"]
        data["streaming_services_count"] = 0
        for col in stream_cols:
            if col in data.columns:
                data["streaming_services_count"] += (data[col] == "Yes").astype(int)

        # 7. Has internet
        if "InternetService" in data.columns:
            data["has_internet"] = (data["InternetService"] != "No").astype(int)
        else:
            data["has_internet"] = 0

        # 8. Solo Senior Citizen Flag
        if all(col in data.columns for col in ["SeniorCitizen", "Partner", "Dependents"]):
            is_senior = (data["SeniorCitizen"].astype(str).isin(["1", "Yes"])).astype(int)
            no_partner = (data["Partner"] == "No").astype(int)
            no_deps = (data["Dependents"] == "No").astype(int)
            data["is_solo_senior"] = (is_senior & no_partner & no_deps).astype(int)
        else:
            data["is_solo_senior"] = 0

        # 9. High Risk Contract + Fiber Interaction
        if all(col in data.columns for col in ["Contract", "InternetService"]):
            is_m2m = (data["Contract"] == "Month-to-month").astype(int)
            is_fiber = (data["InternetService"] == "Fiber optic").astype(int)
            data["high_risk_fiber_m2m"] = (is_m2m & is_fiber).astype(int)
        else:
            data["high_risk_fiber_m2m"] = 0

        logger.info(f"Engineered 9 domain features. New shape: {data.shape}")
        return data
