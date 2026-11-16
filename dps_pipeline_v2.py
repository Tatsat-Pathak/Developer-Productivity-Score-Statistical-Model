"""
DPS Pipeline v2 — Hierarchical Entropy-Weighted Composite Index
===============================================================
Updates from Update.txt:
  1. Remove zero-variance features (review_change_requests,
     review_comments_count, ai_generated_commit)
  2. AI Behavioral Feature Engineering (ai_review_friction, ai_rework_ratio)
  3. Hierarchical entropy weights (within-dim) + equal dim weights
     -> DPS = (TES + CQS + CLS + AIS + RCS) / 5
  4. Normalization comparison: MinMax vs Robust vs Quantile
  5. Full internal + external validation recomputed

Reference: OECD/JRC Composite Indicators Handbook (2008)
           Shannon (1948). A Mathematical Theory of Communication.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

# ── Paths ─────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
INPUT    = os.path.join(BASE, "cleaned_features_v2.csv")
OUT_DPS  = os.path.join(BASE, "dps_final.csv")
OUT_WTS  = os.path.join(BASE, "hierarchical_feature_weights.csv")
OUT_NORM = os.path.join(BASE, "normalization_comparison.csv")
OUT_RPT  = os.path.join(BASE, "internal_validation_report.txt")
OUT_PNG  = os.path.join(BASE, "dps_final_plots.png")

report = []
def sec(t, w=72):
    s = f"\n{'='*w}\n  {t}\n{'='*w}"
    print(s); report.append(s)
def log(m):
    print(f"  {m}"); report.append(f"  {m}")

EPS = 1e-6

# ══════════════════════════════════════════════════════════════
# STAGE 1 — FEATURE REGISTRY
# ══════════════════════════════════════════════════════════════
# polarity: +1 = higher -> better, -1 = higher -> worse (inverted)
FEATURES = {
    # TES — Task Efficiency
    "pr_completion_time_hours"   : {"dim":"TES","polarity":-1},
    "commit_count"               : {"dim":"TES","polarity":-1},
    "issue_resolution_count"     : {"dim":"TES","polarity":+1},
    # CQS — Code Quality
    "approval_status"            : {"dim":"CQS","polarity":+1},
    "bug_fix_indicator"          : {"dim":"CQS","polarity":-1},
    # CLS — Cognitive Load
    "high_context_changes"       : {"dim":"CLS","polarity":-1},
    "timeline_interruptions"     : {"dim":"CLS","polarity":-1},
    "pr_lifecycle_duration_hours": {"dim":"CLS","polarity":-1},
    # AIS — AI Assistance (3 engineered features only)
    "ai_contribution_intensity"  : {"dim":"AIS","polarity":+1},
    "ai_review_friction"         : {"dim":"AIS","polarity":-1},
    "ai_rework_ratio"            : {"dim":"AIS","polarity":-1},
    # RCS — Repository Context
    "repo_stars"                 : {"dim":"RCS","polarity":+1},
    "contributor_count"          : {"dim":"RCS","polarity":+1},
    "repo_size_kb"               : {"dim":"RCS","polarity":+1},
}
DIMS       = ["TES","CQS","CLS","AIS","RCS"]
DIM_LABELS = {"TES":"Task Efficiency","CQS":"Code Quality",
              "CLS":"Cognitive Load","AIS":"AI Assistance",
              "RCS":"Repository Context"}
FEAT_NAMES = list(FEATURES.keys())

# ══════════════════════════════════════════════════════════════
# STAGE 2 — LOAD DATA
# ══════════════════════════════════════════════════════════════
sec("STAGE 2 — DATA LOADING")
df = pd.read_csv(INPUT)
log(f"Rows: {len(df):,}  Columns: {df.shape[1]}")
log(f"Tools: {df['ai_tool_type'].value_counts().to_dict()}")
log(f"Nulls: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════
# STAGE 3 — AI BEHAVIORAL FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
sec("STAGE 3 — AI BEHAVIORAL FEATURE ENGINEERING")
log("""
  REMOVED from pipeline (multicollinearity):
    pr_review_count   -> embedded inside ai_review_friction
    rework_count      -> embedded inside ai_rework_ratio
    ai_contribution_ratio -> replaced by ai_contribution_intensity

  AIS ENGINEERED FEATURES (3 total):

  1. AI_Contribution_Intensity
     = (AI_Contribution_Ratio * Total_Code_Changes) / (Commit_Count + 1)
     -> Captures actual volume of AI-generated code per PR, accounting
        for PR size and commit density. A PR with high AI ratio but tiny
        changes scores lower than one with high ratio AND many changes.

  2. AI_Review_Friction
     = (PR_Review_Count + Review_Comments_Count) / (AI_Contribution_Ratio + eps)
     -> Review overhead relative to AI contribution.
        High friction with high AI = AI-induced complexity.

  3. AI_Rework_Ratio
     = Rework_Count / (AI_Contribution_Ratio + eps)
     -> Rework burden per unit of AI contribution.
        High ratio = AI did NOT reduce rework effectively.

  Preprocessing for all AIS features:
    Step 1: log(1 + x)
    Step 2: global 1st-99th percentile clip
    (Robust normalization applied in Stage 4 with other features)
""")

# ── Reconstruct raw-scale values (undo log1p from cleaned_features_v2) ──
pr_rc  = np.expm1(df["pr_review_count"].fillna(0))
rw     = np.expm1(df["rework_count"].fillna(0))
ai_r   = df["ai_contribution_ratio"].fillna(0)
tcc    = np.expm1(df["total_code_changes"].fillna(0))
cc     = np.expm1(df["commit_count"].fillna(0))

# 1. AI Contribution Intensity
df["ai_contribution_intensity"] = (ai_r * tcc) / (cc + 1)
log(f"  ai_contribution_intensity: raw min={df['ai_contribution_intensity'].min():.4f}  "
    f"max={df['ai_contribution_intensity'].max():.4f}  "
    f"mean={df['ai_contribution_intensity'].mean():.4f}")

# 2. AI Review Friction (review_comments_count is all-zero -> just pr_review_count)
df["ai_review_friction"] = (pr_rc + 0.0) / (ai_r + EPS)
log(f"  ai_review_friction       : raw min={df['ai_review_friction'].min():.4f}  "
    f"max={df['ai_review_friction'].max():.4f}")

# 3. AI Rework Ratio
df["ai_rework_ratio"] = rw / (ai_r + EPS)
log(f"  ai_rework_ratio          : raw min={df['ai_rework_ratio'].min():.4f}  "
    f"max={df['ai_rework_ratio'].max():.4f}")

# Apply log1p + global 1-99 percentile clip to all three AIS features
for col in ["ai_contribution_intensity", "ai_review_friction", "ai_rework_ratio"]:
    df[col] = np.log1p(df[col])
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)
    log(f"  {col}: after log1p+clip -> [{df[col].min():.4f}, {df[col].max():.4f}]  "
        f"std={df[col].std():.4f}")

# ══════════════════════════════════════════════════════════════
# STAGE 4 — NORMALIZATION COMPARISON
# ══════════════════════════════════════════════════════════════
sec("STAGE 4 — NORMALIZATION METHOD COMPARISON")
log("""
  Three normalization methods compared:
    MinMax     : X' = (X - min) / (max - min)   -> [0,1] strict bounds
    Robust     : X' = (X - median) / IQR        -> handles outliers better
    Quantile   : X' = F(X) using rank-based CDF -> near-uniform output

  Selection criterion: lowest |skewness| + lowest kurtosis + best DPS balance.
  Polarity inversion applied AFTER normalization for -1 features.
""")

base_feats = [f for f in FEAT_NAMES if f in df.columns]
X_raw = df[base_feats].copy().fillna(0)

scalers = {
    "MinMax"  : MinMaxScaler(),
    "Robust"  : RobustScaler(),
    "Quantile": QuantileTransformer(output_distribution="uniform",
                                     n_quantiles=min(1000, len(df)),
                                     random_state=42),
}

norm_results = []
norm_data    = {}

for sname, scaler in scalers.items():
    X_sc = pd.DataFrame(scaler.fit_transform(X_raw),
                        columns=base_feats, index=df.index)
    # Apply polarity inversion
    for feat in base_feats:
        if FEATURES[feat]["polarity"] == -1:
            X_sc[feat] = 1.0 - X_sc[feat]
    X_sc = X_sc.clip(0, 1)

    # Compute DPS with equal dim weights for comparison
    dim_scores = {}
    for dim in DIMS:
        feats_d = [f for f in base_feats if FEATURES[f]["dim"] == dim]
        if feats_d:
            dim_scores[dim] = X_sc[feats_d].mean(axis=1)
        else:
            dim_scores[dim] = pd.Series(0.5, index=df.index)
    dps_tmp = sum(dim_scores[d] for d in DIMS) / len(DIMS)

    skew = float(dps_tmp.skew())
    kurt = float(dps_tmp.kurtosis())
    mean = float(dps_tmp.mean())
    std  = float(dps_tmp.std())

    norm_results.append({"Method":sname,"Skewness":round(skew,4),
                         "Kurtosis":round(kurt,4),"Mean":round(mean,4),
                         "Std":round(std,4),"AbsSkew":round(abs(skew),4)})
    norm_data[sname] = {"X_sc":X_sc,"DPS":dps_tmp,"dim_scores":dim_scores}
    log(f"  {sname:<12}: skew={skew:>7.4f}  kurt={kurt:>8.4f}  "
        f"mean={mean:.4f}  std={std:.4f}")

norm_df = pd.DataFrame(norm_results).sort_values("AbsSkew")
norm_df.to_csv(OUT_NORM, index=False)
log(f"\n  Normalization comparison saved -> {OUT_NORM}")

# Select best method
best_method = norm_df.iloc[0]["Method"]
log(f"\n  SELECTED: {best_method} (lowest |skewness|={norm_df.iloc[0]['AbsSkew']})")
X_norm = norm_data[best_method]["X_sc"]

# ══════════════════════════════════════════════════════════════
# STAGE 5 — HIERARCHICAL ENTROPY WEIGHTS (within dimensions)
# ══════════════════════════════════════════════════════════════
sec("STAGE 5 — HIERARCHICAL ENTROPY WEIGHTING")
log(f"""
  HIERARCHICAL STRATEGY (per Update.txt):
    Level 1: Within each dimension, entropy weights determine
             each feature's contribution to its dimension score.
    Level 2: All 5 dimensions weighted EQUALLY (1/5 = 0.20 each).

  Final DPS = (TES + CQS + CLS + AIS + RCS) / 5

  WHY EQUAL DIMENSION WEIGHTS:
    Equal weighting is the most defensible choice when dimensions are
    theoretically equivalent productivity facets. It avoids privileging
    any single dimension and ensures interpretability.
    (Greco et al., 2019; OECD Handbook 2008, §6.3)

  WITHIN-DIM ENTROPY FORMULA:
    p_ij  = x_ij / sum_i(x_ij)          [column proportion]
    e_j   = -k * sum_i(p_ij * ln(p_ij)) [Shannon entropy, k=1/ln(n)]
    d_j   = 1 - e_j                      [divergence]
    w_j   = d_j / sum(d_k) in dim        [normalised within-dim weight]
""")

n    = len(X_norm)
k_e  = 1.0 / np.log(n)

dim_weights    = {}   # within-dim weights per feature
feature_weight = {}   # final feature weight = (1/5) * within-dim weight

weight_rows = []

for dim in DIMS:
    feats_d = [f for f in FEAT_NAMES if FEATURES[f]["dim"] == dim and f in X_norm.columns]
    e_vals  = {}
    for feat in feats_d:
        col = X_norm[feat].values
        s   = col.sum()
        if s == 0:
            e_vals[feat] = 1.0
        else:
            p = col / s
            p = np.where(p > 0, p, 1e-10)
            e_vals[feat] = min(-k_e * np.sum(p * np.log(p)), 1.0)

    d_vals  = {f: 1.0 - e for f, e in e_vals.items()}
    d_sum   = sum(d_vals.values())
    if d_sum == 0:
        w_within = {f: 1.0/len(feats_d) for f in feats_d}
    else:
        w_within = {f: d_vals[f]/d_sum for f in feats_d}

    dim_weights[dim] = w_within
    dim_w = 1.0 / len(DIMS)          # equal dimension weight = 0.20

    log(f"\n  {dim} ({DIM_LABELS[dim]}) — dim weight = {dim_w:.4f}")
    for feat in feats_d:
        fw = dim_w * w_within[feat]
        feature_weight[feat] = fw
        log(f"    {feat:<38} entropy={e_vals[feat]:.6f}  "
            f"w_within={w_within[feat]:.6f}  w_final={fw:.6f}")
        weight_rows.append({
            "Feature"      : feat,
            "Dimension"    : dim,
            "Dim_Label"    : DIM_LABELS[dim],
            "Polarity"     : FEATURES[feat]["polarity"],
            "Entropy"      : round(e_vals[feat],6),
            "W_within_dim" : round(w_within[feat],6),
            "Dim_weight"   : round(dim_w,6),
            "W_final"      : round(fw,6),
        })

wt_df = pd.DataFrame(weight_rows).sort_values(["Dimension","W_within_dim"],ascending=[True,False])
wt_df.to_csv(OUT_WTS, index=False)
log(f"\n  Hierarchical weights saved -> {OUT_WTS}")
log(f"  Sum of W_final: {sum(feature_weight.values()):.6f}  (must be 1.0)")

# ══════════════════════════════════════════════════════════════
# STAGE 6 — DIMENSION SCORES & DPS
# ══════════════════════════════════════════════════════════════
sec("STAGE 6 — DIMENSION SCORES & DPS CONSTRUCTION")

for dim in DIMS:
    feats_d = [f for f in FEAT_NAMES if FEATURES[f]["dim"] == dim and f in X_norm.columns]
    w_arr   = np.array([dim_weights[dim][f] for f in feats_d])
    scores  = X_norm[feats_d].values @ w_arr
    df[dim] = np.clip(scores, 0, 1)
    log(f"  {dim}: mean={df[dim].mean():.4f}  std={df[dim].std():.4f}  "
        f"min={df[dim].min():.4f}  max={df[dim].max():.4f}")

# Equal dimension weights
df["DPS"] = df[DIMS].mean(axis=1).clip(0, 1)

def classify(s):
    if s >= 0.60: return "High"
    elif s >= 0.40: return "Moderate"
    return "Low"
df["Productivity_Class"] = df["DPS"].apply(classify)

log(f"\n  DPS — mean={df['DPS'].mean():.4f}  median={df['DPS'].median():.4f}  "
    f"std={df['DPS'].std():.4f}  skew={df['DPS'].skew():.4f}  kurt={df['DPS'].kurtosis():.4f}")

for cls in ["High","Moderate","Low"]:
    n_cls = (df["Productivity_Class"]==cls).sum()
    log(f"    {cls:10}: {n_cls:6,}  ({n_cls/len(df)*100:.1f}%)")

IS_NONNORMAL = abs(df["DPS"].skew()) > 1

# ══════════════════════════════════════════════════════════════
# STAGE 7 — INTERNAL VALIDATION
# ══════════════════════════════════════════════════════════════
sec("STAGE 7 — INTERNAL VALIDATION")

# 7a. Feature-DPS Spearman correlation
log("\n  7a. Spearman rho (feature vs DPS):")
corr_rows = []
for feat in FEAT_NAMES:
    if feat not in X_norm.columns: continue
    rho, p = spearmanr(X_norm[feat], df["DPS"])
    sig = p < 0.05
    corr_rows.append({"Feature":feat,"Dim":FEATURES[feat]["dim"],
                      "rho":round(rho,4),"p":round(p,6),"significant":sig})
    log(f"    {feat:<38} rho={rho:>7.4f}  p={p:.4f}  {'*' if sig else ' '}")
corr_df = pd.DataFrame(corr_rows).sort_values("rho",ascending=False)

# 7b. Cronbach's Alpha per dimension
log("\n  7b. Cronbach's Alpha per dimension:")
alpha_res = {}
for dim in DIMS:
    feats_d = [f for f in FEAT_NAMES if FEATURES[f]["dim"]==dim and f in X_norm.columns]
    if len(feats_d) < 2:
        alpha_res[dim] = float("nan"); continue
    X_d  = X_norm[feats_d].values
    k_d  = X_d.shape[1]
    vi   = X_d.var(axis=0, ddof=1).sum()
    vt   = X_d.sum(axis=1).var(ddof=1)
    alpha = (k_d/(k_d-1))*(1-vi/vt) if vt > 0 else float("nan")
    alpha_res[dim] = round(alpha, 4)
    q = "acceptable" if alpha >= 0.7 else "low (few items or weak intercorrelation)"
    log(f"    {dim} ({DIM_LABELS[dim]:25}): alpha={alpha:.4f}  [{q}]")

# 7c. VIF
log("\n  7c. VIF (non-zero-variance features only):")
vif_feats = [f for f in FEAT_NAMES if f in X_norm.columns and X_norm[f].std() > 0]
vif_rows  = []
if len(vif_feats) >= 2:
    X_vif = sm.add_constant(X_norm[vif_feats])
    for i, feat in enumerate(vif_feats):
        try:
            v = variance_inflation_factor(X_vif.values, i+1)
        except Exception:
            v = float("nan")
        vif_rows.append({"Feature":feat,"VIF":round(v,3),"Dim":FEATURES[feat]["dim"]})
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF",ascending=False)
    for _, r in vif_df.iterrows():
        flag = " !! HIGH" if r["VIF"] > 10 else ""
        log(f"    {r['Feature']:<38} VIF={r['VIF']:>7.3f}{flag}")
    n_high = (vif_df["VIF"].dropna() > 10).sum()
    log(f"\n    Features with VIF>10: {n_high}")

# 7d. Sensitivity Analysis
log("\n  7d. Sensitivity Analysis (+10% perturbation):")
sens_rows = []
for feat in FEAT_NAMES:
    if feat not in X_norm.columns: continue
    w   = feature_weight.get(feat, 0)
    delta = 0.1 * w
    sens_rows.append({"Feature":feat,"Dim":FEATURES[feat]["dim"],
                      "W_final":round(w,6),"DPS_delta":round(delta,6)})
sens_df = pd.DataFrame(sens_rows).sort_values("DPS_delta",ascending=False)
log("    Top 10 most sensitive features:")
for _, r in sens_df.head(10).iterrows():
    log(f"    {r['Feature']:<38} w={r['W_final']:.6f}  delta={r['DPS_delta']:.6f}")

# 7e. Variance explained per dimension
log("\n  7e. Variance explained per dimension:")
tv = df[DIMS].var().sum()
for dim in DIMS:
    vp = df[dim].var()/tv*100 if tv > 0 else 0
    log(f"    {dim}: {vp:.2f}%")

# ══════════════════════════════════════════════════════════════
# STAGE 8 — EXTERNAL VALIDATION (H1-H6)
# ══════════════════════════════════════════════════════════════
sec("STAGE 8 — EXTERNAL VALIDATION (H1-H6)")

claude = df[df["is_claude_code"]==1]["DPS"]
codex  = df[df["is_claude_code"]==0]["DPS"]

log(f"  Claude Code  N={len(claude):,}  mean={claude.mean():.4f}  std={claude.std():.4f}")
log(f"  OpenAI Codex N={len(codex):,}  mean={codex.mean():.4f}  std={codex.std():.4f}")

# H1: High-AI PRs vs Low-AI PRs (median split on ai_contribution_ratio)
ai_med  = df["ai_contribution_ratio"].median()
hi_ai   = df[df["ai_contribution_ratio"] >= ai_med]["DPS"]
lo_ai   = df[df["ai_contribution_ratio"] <  ai_med]["DPS"]
stat_h1, p_h1 = stats.mannwhitneyu(hi_ai, lo_ai, alternative="two-sided")
ps_h1 = np.sqrt((hi_ai.std()**2 + lo_ai.std()**2) / 2)
cd_h1 = (hi_ai.mean() - lo_ai.mean()) / ps_h1 if ps_h1 > 0 else 0
log(f"  H1 split: High-AI N={len(hi_ai):,} mean={hi_ai.mean():.4f} | "
    f"Low-AI N={len(lo_ai):,} mean={lo_ai.mean():.4f}")

# H2: Claude Code vs OpenAI Codex (tool-level comparison)
if IS_NONNORMAL:
    stat, p_val = stats.mannwhitneyu(claude, codex, alternative="two-sided")
    tname = "Mann-Whitney U"
else:
    stat, p_val = stats.ttest_ind(claude, codex, equal_var=False)
    tname = "Welch t-test"
ps = np.sqrt((claude.std()**2 + codex.std()**2)/2)
cd = (claude.mean()-codex.mean())/ps if ps > 0 else 0
mag = ("negligible" if abs(cd)<0.2 else "small" if abs(cd)<0.5
       else "medium" if abs(cd)<0.8 else "large")

# H3-H6: Spearman correlations
# H3: ai_review_friction NEGATIVELY impacts DPS (higher friction -> lower DPS -> rho < 0)
# H4: ai_rework_ratio NEGATIVELY impacts DPS (higher rework -> lower DPS -> rho < 0)
# H5: TES POSITIVELY contributes to DPS (TES is a sub-score -> rho > 0)
# H6: CLS POSITIVELY links to DPS (CLS score is already inverted -> higher = less load = better)
rho_h3,p_h3 = spearmanr(df["ai_review_friction"],       df["DPS"])
rho_h4,p_h4 = spearmanr(df["ai_rework_ratio"],          df["DPS"])
rho_h5,p_h5 = spearmanr(df["TES"],                      df["DPS"])
rho_h6,p_h6 = spearmanr(df["CLS"],                      df["DPS"])
rho_ai,p_ai = spearmanr(df["ai_contribution_intensity"], df["DPS"])

h1 = (hi_ai.mean() > lo_ai.mean()) and (p_h1 < 0.05)  # High-AI DPS > Low-AI DPS
h2 = (p_val < 0.05) and (claude.mean() > codex.mean())
h3 = (rho_h3 < 0) and (p_h3 < 0.05)   # friction NEGATIVELY correlates with DPS
h4 = (rho_h4 < 0) and (p_h4 < 0.05)   # rework NEGATIVELY correlates with DPS
h5 = (rho_h5 > 0) and (p_h5 < 0.05)   # TES POSITIVELY correlates with DPS
h6 = (rho_h6 > 0) and (p_h6 < 0.05)   # CLS score POSITIVELY correlates with DPS

for lbl, res, detail in [
    ("H1 High-AI PRs outperform Low-AI PRs in DPS",      h1,
     f"Mann-Whitney: stat={stat_h1:.2f} p={p_h1:.6f} "
     f"Cohen_d={cd_h1:.4f} | High-AI={hi_ai.mean():.4f} Low-AI={lo_ai.mean():.4f}"),
    ("H2 Claude Code DPS > OpenAI Codex",                 h2,
     f"{tname}: stat={stat:.2f} p={p_val:.6f} Cohen_d={cd:.4f}({mag})"),
    ("H3 AI Review Friction (negative) harms DPS",        h3,
     f"rho(ai_review_friction, DPS)={rho_h3:.4f}  p={p_h3:.6f}  "
     f"[expected sign: negative]"),
    ("H4 AI Rework Ratio (negative) harms DPS",           h4,
     f"rho(ai_rework_ratio, DPS)={rho_h4:.4f}  p={p_h4:.6f}  "
     f"[expected sign: negative]"),
    ("H5 Task Efficiency (TES) positively contributes",   h5,
     f"rho(TES, DPS)={rho_h5:.4f}  p={p_h5:.6f}  "
     f"[expected sign: positive]"),
    ("H6 Cognitive Load Score (CLS) positively links",    h6,
     f"rho(CLS, DPS)={rho_h6:.4f}  p={p_h6:.6f}  "
     f"[expected sign: positive — CLS is inverted load]"),
]:
    log(f"\n  {lbl}:")
    log(f"    {detail}")
    log(f"    -> {'SUPPORTED' if res else 'NOT SUPPORTED'}")

# ══════════════════════════════════════════════════════════════
# STAGE 8b — H3 PARTIAL SPEARMAN CORRELATION
# ══════════════════════════════════════════════════════════════
sec("STAGE 8b — H3 PARTIAL SPEARMAN CORRELATION (Controlling for AI Contribution)")
log("""
  MOTIVATION:
    H3 was NOT SUPPORTED: rho(ai_review_friction, DPS) = +0.031.
    Likely cause: confounding by ai_contribution_intensity / ai_contribution_ratio.
    High AI contribution simultaneously drives:
      (a) HIGH ai_review_friction (denominator = ai_contribution_ratio + eps is large,
          but numerator pr_review_count grows too in active repos)
      (b) HIGH DPS (AIS dimension is dominated by ai_contribution_intensity w=0.87)
    => A spurious positive association masks the expected negative effect.

  METHOD — Rank-Residual Partial Spearman:
    1. Convert all variables to ranks (scipy.stats.rankdata).
    2. OLS-regress rank(ai_review_friction) ~ rank(confounder) → residuals_X
    3. OLS-regress rank(DPS)               ~ rank(confounder) → residuals_Y
    4. Pearson r(residuals_X, residuals_Y) = partial Spearman rho.
    Repeated for three control sets:
      C1 — control for ai_contribution_intensity  (the AIS score driver)
      C2 — control for ai_contribution_ratio      (the raw ratio; denominator of friction)
      C3 — control for BOTH simultaneously        (most conservative)

  INTERPRETATION:
    If partial rho turns negative and significant after controlling, the raw
    positive rho was purely confounding. H3 would then be supported
    after proper statistical adjustment.
""")

from scipy.stats import rankdata

def partial_spearman(x, y, *controls):
    """
    Compute partial Spearman rho between x and y controlling for *controls.
    Uses the rank-residual OLS method.
    Returns (rho, p_value, n).
    """
    n = len(x)
    rx = rankdata(x).reshape(-1, 1)
    ry = rankdata(y).reshape(-1, 1)
    # Build control rank matrix
    if controls:
        Z = np.column_stack([rankdata(c) for c in controls])
        Z_c = sm.add_constant(Z)
    else:
        Z_c = np.ones((n, 1))

    res_x = rx.ravel() - sm.OLS(rx, Z_c).fit().fittedvalues
    res_y = ry.ravel() - sm.OLS(ry, Z_c).fit().fittedvalues

    rho, pval = stats.pearsonr(res_x, res_y)
    return float(rho), float(pval)

# Variables
arfric  = df["ai_review_friction"].values
dps_v   = df["DPS"].values
aci     = df["ai_contribution_intensity"].values
ac_rat  = df["ai_contribution_ratio"].values

# Raw (unadjusted) Spearman — already in H3 above; recompute for reference
rho_raw,  p_raw  = spearmanr(arfric, dps_v)

# C1: control for ai_contribution_intensity
rho_c1, p_c1 = partial_spearman(arfric, dps_v, aci)

# C2: control for ai_contribution_ratio
rho_c2, p_c2 = partial_spearman(arfric, dps_v, ac_rat)

# C3: control for both simultaneously
rho_c3, p_c3 = partial_spearman(arfric, dps_v, aci, ac_rat)

log(f"\n  {'Control':<45} {'rho':>8}  {'p-value':>12}  {'Sign':>6}  Result")
log(f"  {'-'*85}")
for label, rho_v, p_v in [
    ("None (raw Spearman)",                    rho_raw, p_raw),
    ("C1: ai_contribution_intensity",           rho_c1,  p_c1),
    ("C2: ai_contribution_ratio (raw)",         rho_c2,  p_c2),
    ("C3: both intensity + ratio",              rho_c3,  p_c3),
]:
    sig_str  = "p<.05" if p_v < 0.05 else "n.s."
    sign_str = "-" if rho_v < 0 else "+"
    supported = (rho_v < 0) and (p_v < 0.05)
    log(f"  {label:<45} {rho_v:>8.4f}  {p_v:>12.6f}  {sign_str:>6}  "
        f"[{sig_str}]  {'H3 SUPPORTED' if supported else 'NOT SUPPORTED'}")

# Canonical result for H3 revised verdict
h3_revised      = (rho_c3 < 0) and (p_c3 < 0.05)
h3_revised_rho  = rho_c3
h3_revised_p    = p_c3

log(f"\n  ── REVISED H3 VERDICT (conservative, C3 control) ──")
log(f"     partial rho = {rho_c3:.4f}   p = {p_c3:.6f}")
if h3_revised:
    log("     H3 IS SUPPORTED after controlling for AI contribution confounds.")
    log("     Interpretation: ai_review_friction NEGATIVELY impacts DPS when")
    log("     the collinear effect of AI usage level is held constant.")
else:
    log("     H3 remains NOT SUPPORTED even after controlling for confounds.")
    log("     Interpretation: The positive raw rho is not fully explained by")
    log("     confounding alone; ai_review_friction has no independent negative")
    log("     effect on DPS in this dataset.")

log("\n  NOTE FOR REPORT:")
log("    The shift from raw rho to partial rho quantifies how much of the")
log("    observed association is attributable to confounding by AI usage level.")
log(f"    Confounding contribution ≈ {rho_raw - rho_c3:+.4f} rho units.")

# Store partial rho table for plotting
partial_rows = [
    {"Control": "Raw Spearman",     "rho": rho_raw, "p": p_raw},
    {"Control": "C1: intensity",    "rho": rho_c1,  "p": p_c1},
    {"Control": "C2: ratio",        "rho": rho_c2,  "p": p_c2},
    {"Control": "C3: both",         "rho": rho_c3,  "p": p_c3},
]
partial_df = pd.DataFrame(partial_rows)

# ══════════════════════════════════════════════════════════════
# STAGE 9 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════
sec("STAGE 9 — GENERATING VISUALISATIONS")

PALETTE    = {"Claude_Code":"#5B8FF9","OpenAI_Codex":"#5AD8A6"}
dim_colors = {"TES":"#5B8FF9","CQS":"#5AD8A6","CLS":"#F6BD16",
              "AIS":"#E86452","RCS":"#6DC8EC"}
tools      = df["ai_tool_type"].unique()

fig = plt.figure(figsize=(22,35), constrained_layout=True)
fig.suptitle("DPS Pipeline v2 — Hierarchical Entropy | Equal Dimension Weights",
             fontsize=15, fontweight="bold")
gs  = gridspec.GridSpec(5,3, figure=fig)

# 1. DPS KDE
ax = fig.add_subplot(gs[0,0])
for t in tools:
    g = df[df["ai_tool_type"]==t]["DPS"]
    if g.std()>0: g.plot.kde(ax=ax, label=t, color=PALETTE.get(t,"gray"), lw=2)
ax.set_title("DPS Distribution (KDE)"); ax.set_xlabel("DPS")
ax.set_xlim(0,1); ax.legend(fontsize=8)

# 2. DPS Boxplot
ax = fig.add_subplot(gs[0,1])
bp = ax.boxplot([df[df["ai_tool_type"]==t]["DPS"].values for t in tools],
                patch_artist=True, labels=tools)
for patch,t in zip(bp["boxes"],tools):
    patch.set_facecolor(PALETTE.get(t,"gray")); patch.set_alpha(0.7)
ax.set_title("DPS Boxplot by Tool"); ax.set_ylabel("DPS"); ax.set_ylim(0,1)

# 3. Normalization comparison
ax = fig.add_subplot(gs[0,2])
nm = pd.read_csv(OUT_NORM).sort_values("AbsSkew")
colors_n = ["#2ecc71" if i==0 else "#95a5a6" for i in range(len(nm))]
ax.bar(nm["Method"], nm["AbsSkew"], color=colors_n)
ax.set_title("Normalization |Skewness| Comparison")
ax.set_ylabel("|Skewness|")
for i,(m,v) in enumerate(zip(nm["Method"],nm["AbsSkew"])):
    ax.text(i, v+0.002, f"{v:.4f}", ha="center", fontsize=9)

# 4. Within-dim feature weights
ax = fig.add_subplot(gs[1,0])
wt_s = wt_df.sort_values(["Dimension","W_within_dim"])
ax.barh(wt_s["Feature"], wt_s["W_within_dim"],
        color=[dim_colors.get(d,"gray") for d in wt_s["Dimension"]])
ax.set_title("Within-Dim Entropy Weights"); ax.set_xlabel("W_within_dim")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=dim_colors[d],label=d) for d in DIMS],
          fontsize=7, loc="lower right")

# 5. Spearman rho
ax = fig.add_subplot(gs[1,1])
cs = corr_df.set_index("Feature")["rho"].sort_values()
ax.barh(cs.index, cs.values,
        color=["#e74c3c" if v<0 else "#2ecc71" for v in cs.values])
ax.axvline(0,color="black",lw=0.8)
ax.set_title("Spearman rho (Feature vs DPS)"); ax.set_xlabel("rho")

# 6. Productivity class %
ax = fig.add_subplot(gs[1,2])
cls_list = ["High","Moderate","Low"]
x = np.arange(len(cls_list)); bw = 0.35
for i,t in enumerate(tools):
    g   = df[df["ai_tool_type"]==t]
    pct = [(g["Productivity_Class"]==c).mean()*100 for c in cls_list]
    ax.bar(x+i*bw, pct, bw, label=t, color=PALETTE.get(t,"gray"), alpha=0.8)
ax.set_xticks(x+bw/2); ax.set_xticklabels(cls_list)
ax.set_title("Productivity Class (%)"); ax.set_ylabel("%"); ax.legend(fontsize=8)

# 7. Dim scores grouped bar
ax = fig.add_subplot(gs[2,0])
x = np.arange(len(DIMS)); bw = 0.35
for i,t in enumerate(tools):
    means=[df[df["ai_tool_type"]==t][d].mean() for d in DIMS]
    ax.bar(x+i*bw, means, bw, label=t, color=PALETTE.get(t,"gray"), alpha=0.8)
ax.set_title("Dimension Scores by Tool"); ax.set_xticks(x+bw/2)
ax.set_xticklabels(DIMS); ax.set_ylim(0,1); ax.legend(fontsize=8)

# 8. AIS engineered features scatter
ax = fig.add_subplot(gs[2,1])
for t in tools:
    g = df[df["ai_tool_type"]==t]
    ax.scatter(g["ai_review_friction"], g["DPS"],
               alpha=0.15, s=5, color=PALETTE.get(t,"gray"), label=t)
ax.set_title("AI Review Friction vs DPS")
ax.set_xlabel("ai_review_friction"); ax.set_ylabel("DPS"); ax.legend(fontsize=8)

# 9. ai_rework_ratio vs DPS
ax = fig.add_subplot(gs[2,2])
for t in tools:
    g = df[df["ai_tool_type"]==t]
    ax.scatter(g["ai_rework_ratio"], g["DPS"],
               alpha=0.15, s=5, color=PALETTE.get(t,"gray"), label=t)
ax.set_title("AI Rework Ratio vs DPS")
ax.set_xlabel("ai_rework_ratio"); ax.set_ylabel("DPS"); ax.legend(fontsize=8)

# 10. Equal dim weight pie
ax = fig.add_subplot(gs[3,0])
ax.pie([0.20]*5, labels=DIMS, autopct="%1.0f%%",
       colors=[dim_colors[d] for d in DIMS], startangle=90)
ax.set_title("Equal Dimension Weights (1/5 each)")

# 11. Sensitivity top features
ax = fig.add_subplot(gs[3,1])
top10 = sens_df.head(12)
ax.barh(top10["Feature"], top10["DPS_delta"],
        color=[dim_colors.get(d,"gray") for d in top10["Dim"]])
ax.set_title("Top Feature Sensitivity (+10%)"); ax.set_xlabel("ΔDPS")

# 12. Hypothesis chart
ax = fig.add_subplot(gs[3,2])
hl  = ["H1","H2","H3","H4","H5","H6"]
hv  = [h1,h2,h3,h4,h5,h6]
ax.bar(hl, [int(v) for v in hv],
       color=["#2ecc71" if v else "#e74c3c" for v in hv])
ax.set_title("Hypothesis Validation"); ax.set_ylim(0,1.4)
for i,(l,v) in enumerate(zip(hl,hv)):
    ax.text(i, int(v)+0.07, "YES" if v else "NO",
            ha="center", fontsize=9, fontweight="bold")

# 13. H3 Partial rho bar chart
ax = fig.add_subplot(gs[4, 0])
colors_p = ["#e74c3c" if r < 0 else "#e67e22" for r in partial_df["rho"]]
bars = ax.bar(partial_df["Control"], partial_df["rho"], color=colors_p, edgecolor="white", lw=0.8)
ax.axhline(0, color="black", lw=1)
ax.set_title("H3: Partial Spearman rho\n(ai_review_friction vs DPS)", fontsize=10)
ax.set_ylabel("Spearman rho")
ax.set_xticklabels(partial_df["Control"], rotation=15, ha="right", fontsize=8)
for bar, (_, row) in zip(bars, partial_df.iterrows()):
    sig_lbl = "*" if row["p"] < 0.05 else "n.s."
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.001 if bar.get_height() >= 0 else -0.003),
            f"{row['rho']:.4f}\n{sig_lbl}",
            ha="center", va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=8, fontweight="bold")

# 14. Partial residual scatter (C3 control — most conservative)
ax = fig.add_subplot(gs[4, 1])
rx_c3 = rankdata(arfric)
ry_c3 = rankdata(dps_v)
Z_c3  = sm.add_constant(np.column_stack([rankdata(aci), rankdata(ac_rat)]))
res_fric_c3 = rx_c3 - sm.OLS(rx_c3, Z_c3).fit().fittedvalues
res_dps_c3  = ry_c3 - sm.OLS(ry_c3, Z_c3).fit().fittedvalues
ax.scatter(res_fric_c3, res_dps_c3, alpha=0.07, s=4, color="#9b59b6")
m_c3, b_c3, *_ = stats.linregress(res_fric_c3, res_dps_c3)
x_line = np.linspace(res_fric_c3.min(), res_fric_c3.max(), 200)
ax.plot(x_line, m_c3 * x_line + b_c3, color="#c0392b", lw=2, label=f"rho={rho_c3:.4f}")
ax.axhline(0, color="grey", lw=0.6, ls="--")
ax.axvline(0, color="grey", lw=0.6, ls="--")
ax.set_title("H3: Partial Residual Scatter\n(C3: controlling intensity + ratio)", fontsize=10)
ax.set_xlabel("rank(friction) residuals")
ax.set_ylabel("rank(DPS) residuals")
ax.legend(fontsize=9)

# 15. Summary text panel
ax = fig.add_subplot(gs[4, 2])
ax.axis("off")
summary_txt = (
    f"H3 PARTIAL CORRELATION SUMMARY\n"
    f"{'─'*32}\n"
    f"Raw rho     : {rho_raw:+.4f}  {'*' if p_raw<0.05 else 'n.s.'}\n"
    f"C1 (intens.): {rho_c1:+.4f}  {'*' if p_c1<0.05 else 'n.s.'}\n"
    f"C2 (ratio)  : {rho_c2:+.4f}  {'*' if p_c2<0.05 else 'n.s.'}\n"
    f"C3 (both)   : {rho_c3:+.4f}  {'*' if p_c3<0.05 else 'n.s.'}\n"
    f"{'─'*32}\n"
    f"Confound Δrho: {rho_raw - rho_c3:+.4f}\n\n"
    f"REVISED H3:\n"
    f"{'SUPPORTED' if h3_revised else 'NOT SUPPORTED'}\n"
    f"(C3 conservative control)"
)
ax.text(0.05, 0.95, summary_txt, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.8))
ax.set_title("H3 Revised Verdict", fontsize=10)

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
log(f"  Plot saved -> {OUT_PNG}")

# ══════════════════════════════════════════════════════════════
# STAGE 10 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════
sec("STAGE 10 — SAVING OUTPUTS")

out_cols = (["pr_id","ai_tool_type","is_claude_code"] +
            DIMS + ["DPS","Productivity_Class"] +
            ["ai_contribution_intensity","ai_review_friction","ai_rework_ratio"])
df[[c for c in out_cols if c in df.columns]].to_csv(OUT_DPS, index=False)
log(f"  dps_final.csv              -> {OUT_DPS}")
log(f"  hierarchical_weights.csv   -> {OUT_WTS}")
log(f"  normalization_comparison   -> {OUT_NORM}")

rpt_lines = [
    "="*72,
    "DPS PIPELINE v2 — INTERNAL VALIDATION REPORT",
    f"Normalization: {best_method} (selected by min |skewness|)",
    "Dimension weights: Equal (1/5 = 0.20 each)",
    "Feature weights: Entropy-based within each dimension",
    "="*72, "",
    "DIMENSION SCORES:",
] + [f"  {d} ({DIM_LABELS[d]}): mean={df[d].mean():.4f}  std={df[d].std():.4f}"
     for d in DIMS] + [
    "",
    f"DPS: mean={df['DPS'].mean():.4f}  median={df['DPS'].median():.4f}  "
    f"std={df['DPS'].std():.4f}  skew={df['DPS'].skew():.4f}",
    "",
    "CRONBACH'S ALPHA:",
] + [f"  {d}: {alpha_res.get(d,'nan')}" for d in DIMS] + [
    "",
    "VIF (top 5):",
] + ([f"  {r['Feature']}: {r['VIF']}" for _,r in vif_df.head(5).iterrows()]
     if vif_rows else ["  N/A"]) + [
    "",
    "HYPOTHESIS RESULTS:",
    f"  H1: {'SUPPORTED' if h1 else 'NOT SUPPORTED'}",
    f"  H2: {'SUPPORTED' if h2 else 'NOT SUPPORTED'}  ({tname} p={p_val:.6f})",
    f"  H3: {'SUPPORTED' if h3 else 'NOT SUPPORTED'}  (rho={rho_h3:.4f})",
    f"  H4: {'SUPPORTED' if h4 else 'NOT SUPPORTED'}  (rho={rho_h4:.4f})",
    f"  H5: {'SUPPORTED' if h5 else 'NOT SUPPORTED'}  (rho={rho_h5:.4f})",
    f"  H6: {'SUPPORTED' if h6 else 'NOT SUPPORTED'}  (rho={rho_h6:.4f})",
    "",
    "H3 PARTIAL CORRELATION ANALYSIS:",
    f"  Raw Spearman rho          : {rho_raw:+.4f}  (p={p_raw:.6f})",
    f"  C1 control (intensity)    : {rho_c1:+.4f}  (p={p_c1:.6f})",
    f"  C2 control (ratio)        : {rho_c2:+.4f}  (p={p_c2:.6f})",
    f"  C3 control (both)         : {rho_c3:+.4f}  (p={p_c3:.6f})",
    f"  Confounding contribution  : {rho_raw - rho_c3:+.4f} rho units",
    f"  REVISED H3 VERDICT        : {'SUPPORTED' if h3_revised else 'NOT SUPPORTED'} (C3 conservative)",
    "", "="*72, "END OF REPORT", "="*72,
]
report.extend(rpt_lines)
with open(OUT_RPT,"w",encoding="utf-8") as f:
    f.write("\n".join(report))
log(f"  internal_validation_report -> {OUT_RPT}")

sec("PIPELINE COMPLETE")
for p in [OUT_DPS,OUT_WTS,OUT_NORM,OUT_RPT,OUT_PNG]:
    log(p)
