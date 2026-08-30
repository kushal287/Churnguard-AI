"""
Prescriptive Retention Playbook Engine for ChurnGuard AI.
Translates customer attributes and ML risk factors into prioritized, deterministic retention actions
using non-causal, compliant business decisioning logic locked to the validation threshold (0.58).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class RetentionPlaybook:
    """Generates tailored, deterministic retention action plans for high-risk customer profiles."""

    LOCKED_DECISION_THRESHOLD = 0.58

    def __init__(self, threshold: float = 0.58):
        self.threshold = threshold

    def classify_risk(self, churn_probability: float) -> Dict[str, str]:
        """
        Classify customer into standardized risk tiers based on the locked decision threshold (0.58).
        """
        if churn_probability >= self.threshold:
            return {
                "tier": "HIGH CHURN RISK",
                "tier_code": "HIGH",
                "badge_color": "#E53935",
                "status_text": f"Estimated churn probability ({churn_probability:.1%}) exceeds the decision threshold ({self.threshold:.2f}).",
                "urgency": "High Priority (Proactive Outreach Recommended)",
            }
        elif churn_probability >= 0.40:
            return {
                "tier": "MEDIUM RISK (ELEVATED)",
                "tier_code": "MEDIUM",
                "badge_color": "#FB8C00",
                "status_text": f"Estimated churn probability ({churn_probability:.1%}) is approaching the decision threshold ({self.threshold:.2f}).",
                "urgency": "Medium Priority (Targeted Loyalty Nurturing)",
            }
        else:
            return {
                "tier": "LOW RISK (HEALTHY)",
                "tier_code": "LOW",
                "badge_color": "#43A047",
                "status_text": f"Estimated churn probability ({churn_probability:.1%}) is well below the decision threshold ({self.threshold:.2f}).",
                "urgency": "Standard Lifecycle Engagement",
            }

    def generate_recommendations(
        self,
        customer_dict: Dict[str, Any],
        churn_probability: float,
        risk_factors: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Produce prioritized retention interventions deterministically mapped to customer attributes.
        """
        recommendations = []
        tenure = float(customer_dict.get("tenure", 0))
        monthly_charges = float(customer_dict.get("MonthlyCharges", 0.0))
        contract = str(customer_dict.get("Contract", "Month-to-month"))
        internet = str(customer_dict.get("InternetService", "Fiber optic"))
        payment = str(customer_dict.get("PaymentMethod", "Electronic check"))
        tech_support = str(customer_dict.get("TechSupport", "No"))
        online_sec = str(customer_dict.get("OnlineSecurity", "No"))
        paperless = str(customer_dict.get("PaperlessBilling", "Yes"))

        risk_info = self.classify_risk(churn_probability)
        is_elevated_or_high = churn_probability >= 0.40

        # 1. Contract Loyalty Strategy (Trigger: Month-to-month contract)
        if contract == "Month-to-month":
            priority = "HIGH" if is_elevated_or_high else "MEDIUM"
            recommendations.append({
                "category": "Contract & Commitment",
                "action": "Annual Loyalty Commitment Offer",
                "description": "Offer an illustrative 15% promotional credit contingent on committing to a 1-Year or 2-Year Contract.",
                "rationale": "Month-to-month contracts are the strongest statistical driver of churn in the dataset (OR = 4.09 relative to 2-year).",
                "expected_impact": "Designed to mitigate contract volatility by transitioning the subscriber to a multi-year term.",
                "financial_cost": "Illustrative $12/month discount",
                "priority": priority,
            })

        # 2. Fiber Optic Support Bundling (Trigger: Fiber Optic without TechSupport/OnlineSecurity)
        if internet == "Fiber optic" and (tech_support == "No" or online_sec == "No"):
            priority = "HIGH" if is_elevated_or_high else "MEDIUM"
            recommendations.append({
                "category": "Service & Experience Guard",
                "action": "Complimentary Tech & Security Guard Bundle",
                "description": "Provide 3 months of complimentary 24/7 dedicated Technical Support and Online Security.",
                "rationale": "Fiber optic subscribers lacking technical support exhibit higher modeled churn rates due to setup friction.",
                "expected_impact": "Addresses technical setup dissatisfaction and enhances perceived product value.",
                "financial_cost": "Illustrative $5/month service credit",
                "priority": priority,
            })

        # 3. High Monthly Charges Mitigation (Trigger: MonthlyCharges >= $75)
        if monthly_charges >= 75.0:
            recommendations.append({
                "category": "Price Sensitivity & Optimization",
                "action": "Account Plan Optimization & Value Review",
                "description": "Initiate an account review to adjust plan features or apply an automated $10 loyalty credit.",
                "rationale": "Elevated monthly charges and price discrepancies are associated with higher customer price sensitivity.",
                "expected_impact": "Reduces billing friction before the subscriber evaluates alternative market offerings.",
                "financial_cost": "Illustrative $10/month credit",
                "priority": "MEDIUM",
            })

        # 4. Payment Method Migration (Trigger: Electronic Check)
        if payment == "Electronic check":
            recommendations.append({
                "category": "Billing & Payment Friction",
                "action": "Automated Auto-Pay Migration Bonus",
                "description": "Offer a one-time $15 bill credit upon switching from Electronic Check to Bank ACH or Credit Card Auto-Pay.",
                "rationale": "Electronic check payments are associated with 41.4% higher modeled churn odds relative to automated bank transfer.",
                "expected_impact": "Reduces manual billing friction and involuntary payment failure attrition.",
                "financial_cost": "Illustrative $15 one-time credit",
                "priority": "MEDIUM",
            })

        # 5. Early Lifecycle Success (Trigger: Tenure <= 6 months)
        if tenure <= 6:
            priority = "HIGH" if is_elevated_or_high else "MEDIUM"
            recommendations.append({
                "category": "Early Onboarding Success",
                "action": "Dedicated Customer Success Concierge Check-in",
                "description": "Schedule a proactive welcome check-in from Customer Support to audit satisfaction and resolve early inquiries.",
                "rationale": "Subscribers with <= 6 months tenure exhibit the steepest drop-off rates across the customer lifecycle.",
                "expected_impact": "Stabilizes early customer adoption during the critical first 90-180 days.",
                "financial_cost": "$0 (Operational CS Time)",
                "priority": priority,
            })

        # Default healthy account maintenance if no specific trigger fired
        if not recommendations:
            recommendations.append({
                "category": "Standard Lifecycle",
                "action": "Maintain Standard Loyalty Engagement",
                "description": "Account indicators show healthy engagement. Continue regular quarterly product updates and appreciation perks.",
                "rationale": "Subscriber exhibits low churn risk with protective multi-year or established tenure factors.",
                "expected_impact": "Preserves organic subscriber satisfaction without unnecessary financial concessions.",
                "financial_cost": "$0.00",
                "priority": "LOW",
            })

        return recommendations
