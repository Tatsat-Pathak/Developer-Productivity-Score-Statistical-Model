"""
Statistical DPS Model — White-Box Explainable Model
=====================================================
Implements the hierarchical entropy-weighted composite index
from dps_pipeline_v2.py as a callable predict() method.

This is the SAME mathematical model — just packaged for inference.
"""

import numpy as np


class StatisticalDPSModel:
    """
    Hierarchical Entropy-Weighted Developer Productivity Score.

    Architecture:
        Level 1: Within each dimension, entropy weights combine features.
        Level 2: All 5 dimensions averaged equally (1/5 each).
        DPS = (TES + CQS + CLS + AIS + RCS) / 5

    All weights are pre-computed from the training data (22,258 PRs)
    and hardcoded here for inference speed.
    """

    def __init__(self):
        # ── Feature registry: name → (dimension, polarity, within-dim weight) ──
        # Weights from dps_pipeline_v2.py Stage 5 (Shannon entropy)
        self.features = {
            # TES — Task Efficiency
            "pr_completion_time_hours":    {"dim": "TES", "pol": -1, "w": 0.065273},
            "commit_count":                {"dim": "TES", "pol": -1, "w": 0.014566},
            "issue_resolution_count":      {"dim": "TES", "pol": +1, "w": 0.920161},
            # CQS — Code Quality
            "approval_status":             {"dim": "CQS", "pol": +1, "w": 0.927942},
            "bug_fix_indicator":           {"dim": "CQS", "pol": -1, "w": 0.072058},
            # CLS — Cognitive Load
            "high_context_changes":        {"dim": "CLS", "pol": -1, "w": 0.376022},
            "timeline_interruptions":      {"dim": "CLS", "pol": -1, "w": 0.055258},
            "pr_lifecycle_duration_hours":  {"dim": "CLS", "pol": -1, "w": 0.568720},
            # AIS — AI Assistance
            "ai_contribution_intensity":   {"dim": "AIS", "pol": +1, "w": 0.872079},
            "ai_review_friction":          {"dim": "AIS", "pol": -1, "w": 0.084590},
            "ai_rework_ratio":             {"dim": "AIS", "pol": -1, "w": 0.043330},
            # RCS — Repository Context
            "repo_stars":                  {"dim": "RCS", "pol": +1, "w": 0.295582},
            "contributor_count":           {"dim": "RCS", "pol": +1, "w": 0.274783},
            "repo_size_kb":                {"dim": "RCS", "pol": +1, "w": 0.429634},
        }

        self.dims = ["TES", "CQS", "CLS", "AIS", "RCS"]
        self.dim_labels = {
            "TES": "Task Efficiency",
            "CQS": "Code Quality",
            "CLS": "Cognitive Load",
            "AIS": "AI Assistance",
            "RCS": "Repository Context",
        }

        # ── RobustScaler parameters (median, IQR) from training data ──
        # These are for the log1p-transformed features from cleaned_features_v2.csv
        self.scaler_params = {
            "pr_completion_time_hours":   {"median": 0.014593, "q25": 0.004689, "q75": 0.070272},
            "commit_count":              {"median": 0.693147, "q25": 0.693147, "q75": 0.693147},
            "issue_resolution_count":    {"median": 0.0,      "q25": 0.0,      "q75": 0.0},
            "approval_status":           {"median": 0.0,      "q25": 0.0,      "q75": 0.0},
            "bug_fix_indicator":         {"median": 0.0,      "q25": 0.0,      "q75": 0.0},
            "high_context_changes":      {"median": 0.0,      "q25": 0.0,      "q75": 0.0},
            "timeline_interruptions":    {"median": 0.693147, "q25": 0.693147, "q75": 0.693147},
            "pr_lifecycle_duration_hours":{"median": 0.030626, "q25": 0.005286, "q75": 0.660676},
            "ai_contribution_intensity": {"median": 3.5,      "q25": 1.5,      "q75": 5.5},
            "ai_review_friction":        {"median": 0.8,      "q25": 0.0,      "q75": 1.5},
            "ai_rework_ratio":           {"median": 0.5,      "q25": 0.0,      "q75": 1.2},
            "repo_stars":                {"median": 5.726848, "q25": 5.529429, "q75": 5.834811},
            "contributor_count":         {"median": 1.945910, "q25": 0.693147, "q75": 2.484907},
            "repo_size_kb":              {"median": 11.553837,"q25": 10.255235,"q75": 11.553837},
        }

    def _engineer_ais_features(self, raw):
        """
        Compute the 3 AIS engineered features from raw input values.
        These are the same formulas used in dps_pipeline_v2.py Stage 3.
        """
        EPS = 1e-6
        ai_ratio = raw.get("ai_contribution_ratio", 0)
        total_changes = raw.get("total_code_changes", 0)
        commit_count = raw.get("commit_count", 1)
        pr_review_count = raw.get("pr_review_count", 0)
        rework_count = raw.get("rework_count", 0)

        # 1. AI Contribution Intensity
        ai_intensity = (ai_ratio * total_changes) / (commit_count + 1)

        # 2. AI Review Friction
        ai_friction = pr_review_count / (ai_ratio + EPS)

        # 3. AI Rework Ratio
        ai_rework = rework_count / (ai_ratio + EPS)

        # Apply log1p (same as pipeline)
        return {
            "ai_contribution_intensity": float(np.log1p(ai_intensity)),
            "ai_review_friction":        float(np.log1p(ai_friction)),
            "ai_rework_ratio":           float(np.log1p(ai_rework)),
        }

    def _preprocess(self, raw):
        """
        Apply the same preprocessing as the pipeline:
        1. log1p transform for continuous features
        2. RobustScaler normalization
        3. Polarity inversion (1 - x for negative polarity)
        4. Clip to [0, 1]
        """
        # Engineer AIS features
        ais_feats = self._engineer_ais_features(raw)

        # Build the 14-feature vector with log1p
        log_feats = {}
        for name in self.features:
            if name in ais_feats:
                log_feats[name] = ais_feats[name]  # Already log1p'd
            elif name in ["approval_status", "bug_fix_indicator", "high_context_changes"]:
                log_feats[name] = float(raw.get(name, 0))  # Binary, no log
            else:
                log_feats[name] = float(np.log1p(max(0, raw.get(name, 0))))

        # RobustScaler: (x - median) / IQR
        scaled = {}
        for name, val in log_feats.items():
            params = self.scaler_params[name]
            iqr = params["q75"] - params["q25"]
            if iqr > 0:
                scaled[name] = (val - params["median"]) / iqr
            else:
                # IQR = 0 means constant feature; use raw value clipped
                scaled[name] = val

        # Polarity inversion: for -1 features, flip so higher = better
        inverted = {}
        for name, val in scaled.items():
            if self.features[name]["pol"] == -1:
                inverted[name] = 1.0 - val
            else:
                inverted[name] = val

        # Clip to [0, 1]
        clipped = {name: max(0.0, min(1.0, val)) for name, val in inverted.items()}

        return clipped, log_feats, scaled

    def predict(self, raw_input):
        """
        Full prediction pipeline.
        Returns a dict with DPS, dimension scores, feature contributions, and explanations.
        """
        clipped, log_feats, scaled = self._preprocess(raw_input)

        # ── Compute dimension scores ───────────────────────────
        dim_scores = {}
        dim_contributions = {}

        for dim in self.dims:
            feats_in_dim = {n: v for n, v in self.features.items() if v["dim"] == dim}
            score = 0.0
            contributions = []

            for feat_name, feat_info in feats_in_dim.items():
                w = feat_info["w"]
                val = clipped.get(feat_name, 0)
                contrib = w * val
                score += contrib
                contributions.append({
                    "feature": feat_name,
                    "normalized_value": round(val, 4),
                    "weight": round(w, 6),
                    "contribution": round(contrib, 6),
                    "polarity": feat_info["pol"],
                })

            dim_scores[dim] = round(max(0.0, min(1.0, score)), 4)
            dim_contributions[dim] = contributions

        # ── DPS = average of 5 dimensions ──────────────────────
        dps = round(sum(dim_scores.values()) / 5.0, 4)
        dps = max(0.0, min(1.0, dps))

        # ── Productivity class ─────────────────────────────────
        if dps >= 0.60:
            prod_class = "High"
        elif dps >= 0.40:
            prod_class = "Moderate"
        else:
            prod_class = "Low"

        # ── Explainability summary ─────────────────────────────
        # Find top positive and negative contributors
        all_contribs = []
        for dim, contribs in dim_contributions.items():
            for c in contribs:
                c["dimension"] = dim
                all_contribs.append(c)

        sorted_contribs = sorted(all_contribs, key=lambda x: x["contribution"], reverse=True)
        top_positive = [c for c in sorted_contribs if c["contribution"] > 0.01][:3]
        top_negative = [c for c in sorted_contribs if c["contribution"] <= 0.01][:3]

        explanation = {
            "model_type": "White-Box Statistical Model",
            "method": "Hierarchical Entropy-Weighted Composite Index",
            "formula": "DPS = (TES + CQS + CLS + AIS + RCS) / 5",
            "weighting": "Shannon entropy within dimensions, equal (20%) across dimensions",
            "normalization": "RobustScaler (median/IQR)",
            "top_contributors": top_positive,
            "low_contributors": top_negative,
            "interpretation": self._generate_interpretation(dps, dim_scores, prod_class),
        }

        return {
            "dps": dps,
            "productivity_class": prod_class,
            "dimension_scores": dim_scores,
            "dimension_labels": self.dim_labels,
            "contributions": dim_contributions,
            "explanation": explanation,
        }

    def _generate_interpretation(self, dps, dim_scores, prod_class):
        """Generate a human-readable interpretation of the DPS result."""
        lines = []
        lines.append(f"This PR scores {dps:.4f} on the DPS index, classified as '{prod_class}' productivity.")

        # Find strongest and weakest dimensions
        sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1], reverse=True)
        best = sorted_dims[0]
        worst = sorted_dims[-1]

        lines.append(f"Strongest dimension: {self.dim_labels[best[0]]} ({best[0]} = {best[1]:.4f}).")
        lines.append(f"Weakest dimension: {self.dim_labels[worst[0]]} ({worst[0]} = {worst[1]:.4f}).")

        if worst[1] < 0.2:
            lines.append(f"Recommendation: Focus on improving {self.dim_labels[worst[0]]} to raise the overall DPS.")

        return " ".join(lines)
