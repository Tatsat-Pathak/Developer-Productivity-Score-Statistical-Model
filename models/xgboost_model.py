"""
XGBoost DPS Model — Black-Box ML Model
========================================
Loads the pre-trained XGBoost regressor from dps_xgboost_model.json
and provides a predict() method with the same interface as the statistical model.
"""

import os
import numpy as np
import xgboost as xgb


class XGBoostDPSModel:
    """
    XGBoost Regressor for DPS prediction.
    Trained on the same 14 pipeline features + extra behavioral features.
    """

    def __init__(self, model_path):
        """Load the pre-trained XGBoost model from JSON file."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        # Feature names — must match EXACTLY the order from the trained model
        # (38 features: 14 pipeline + 13 extra behavioral + 11 language dummies)
        self.feature_names = [
            "pr_completion_time_hours", "commit_count", "issue_resolution_count",
            "approval_status", "bug_fix_indicator",
            "high_context_changes", "timeline_interruptions", "pr_lifecycle_duration_hours",
            "ai_contribution_intensity", "ai_review_friction", "ai_rework_ratio",
            "repo_stars", "contributor_count", "repo_size_kb",
            "files_changed_count", "lines_added", "lines_deleted", "total_code_changes",
            "merge_success", "review_turnaround_time_hours", "pr_review_count",
            "rework_count", "bug_fix_commit_count", "large_pr_flag",
            "ai_contribution_ratio", "has_review", "is_claude_code",
            "lang_C#", "lang_Dart", "lang_Go", "lang_HTML",
            "lang_Java", "lang_JavaScript", "lang_Kotlin", "lang_Other",
            "lang_Python", "lang_Rust", "lang_TypeScript",
        ]

        # Features that need log1p transform (continuous features only)
        self.log_features = [
            "pr_completion_time_hours", "commit_count", "issue_resolution_count",
            "files_changed_count", "lines_added", "lines_deleted", "total_code_changes",
            "review_turnaround_time_hours", "pr_review_count", "rework_count",
            "bug_fix_commit_count", "ai_contribution_ratio",
            "repo_stars", "contributor_count", "repo_size_kb",
            "ai_contribution_intensity", "ai_review_friction", "ai_rework_ratio",
            "pr_lifecycle_duration_hours",
        ]

    def _engineer_features(self, raw):
        """Compute AIS engineered features from raw input."""
        EPS = 1e-6
        ai_ratio = raw.get("ai_contribution_ratio", 0)
        total_changes = raw.get("total_code_changes", 0)
        commit_count = raw.get("commit_count", 1)
        pr_review_count = raw.get("pr_review_count", 0)
        rework_count = raw.get("rework_count", 0)

        return {
            "ai_contribution_intensity": (ai_ratio * total_changes) / (commit_count + 1),
            "ai_review_friction": pr_review_count / (ai_ratio + EPS),
            "ai_rework_ratio": rework_count / (ai_ratio + EPS),
        }

    def predict(self, raw_input):
        """
        Run XGBoost prediction.
        Returns DPS score, productivity class, and feature importances.
        """
        # Engineer AIS features
        ais = self._engineer_features(raw_input)

        # Build feature vector with defaults for missing features
        defaults = {
            "files_changed_count": raw_input.get("total_code_changes", 150) * 0.1,
            "lines_added": raw_input.get("total_code_changes", 150) * 0.6,
            "lines_deleted": raw_input.get("total_code_changes", 150) * 0.4,
            "merge_success": 1,
            "review_turnaround_time_hours": raw_input.get("pr_completion_time_hours", 2) * 0.3,
            "bug_fix_commit_count": raw_input.get("bug_fix_indicator", 0),
            "large_pr_flag": 1 if raw_input.get("total_code_changes", 0) > 500 else 0,
            "has_review": 1 if raw_input.get("pr_review_count", 0) > 0 else 0,
            "is_claude_code": 0,
            # Language dummies — default to Python
            "lang_C#": 0, "lang_Dart": 0, "lang_Go": 0, "lang_HTML": 0,
            "lang_Java": 0, "lang_JavaScript": 0, "lang_Kotlin": 0, "lang_Other": 0,
            "lang_Python": 1, "lang_Rust": 0, "lang_TypeScript": 0,
        }

        feature_values = []
        for fname in self.feature_names:
            if fname in ais:
                val = ais[fname]
            elif fname in raw_input:
                val = raw_input[fname]
            elif fname in defaults:
                val = defaults[fname]
            else:
                val = 0.0

            # Apply log1p for continuous features
            if fname in self.log_features:
                val = float(np.log1p(max(0, val)))

            feature_values.append(val)

        # Predict
        X = np.array(feature_values).reshape(1, -1)
        dps_pred = float(self.model.predict(X)[0])
        dps_pred = max(0.0, min(1.0, dps_pred))

        # Productivity class
        if dps_pred >= 0.60:
            prod_class = "High"
        elif dps_pred >= 0.40:
            prod_class = "Moderate"
        else:
            prod_class = "Low"

        # Feature importances from the model
        importances = self.model.feature_importances_
        feat_imp = []
        for i, fname in enumerate(self.feature_names):
            if i < len(importances):
                feat_imp.append({
                    "feature": fname,
                    "importance": round(float(importances[i]), 6),
                })
        feat_imp.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "dps": round(dps_pred, 4),
            "productivity_class": prod_class,
            "feature_importances": feat_imp[:15],  # Top 15
            "model_info": {
                "model_type": "Black-Box ML Model",
                "algorithm": "XGBoost Gradient Boosted Trees",
                "training_r2": 0.9981,
                "training_rmse": 0.0036,
                "n_features": len(self.feature_names),
            },
        }
