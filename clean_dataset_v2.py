"""
Data Cleaning Pipeline v2 — Developer Productivity Score Research
=================================================================
Input  : extracted_features.csv
Output : cleaned_features_v2.csv

KEY CHANGE FROM v1:
  REMOVED: Group-wise IQR Winsorisation (caused variance collapse)
  ADDED  : log1p transformation + global 1st-99th percentile clipping

WHY v1 WAS PROBLEMATIC:
  IQR Winsorisation applied per AI-tool group collapsed all variance
  when Q1=Q3=0 (IQR=0). A single non-zero value became an "outlier"
  and was capped to 0. This destroyed review friction, cognitive load,
  and rework signals — the very signals needed for DPS construction.

WHY LOG TRANSFORMATION IS BETTER FOR SE DATASETS:
  Software engineering metrics (PR counts, review rounds, commit counts)
  follow heavy-tailed / power-law distributions. These are NOT noise —
  a PR with 14 review rounds genuinely represents high review friction.
  log(1+X) compresses the tail while preserving relative ordering and
  all non-zero behavioral signals (Shannon, 1948; Herraiz et al., 2013).

WHY GLOBAL PERCENTILE CLIPPING (NOT GROUP-WISE):
  Clipping at 1st-99th percentile globally removes only extreme outliers
  (top/bottom 1%) without collapsing within-group variance. Thresholds
  are computed on the full dataset so tool distributions remain comparable.

Pipeline Steps:
  1. Load & audit
  2. Duplicate removal
  3. Structural validation
  4. Missing value imputation
  5. Skewness analysis (BEFORE transformation)
  6. log1p transformation on heavy-tailed features
  7. Global 1st-99th percentile clipping
  8. Skewness analysis (AFTER transformation)
  9. Categorical encoding
  10. Sanity corrections
  11. Drop redundant columns
  12. Data type optimisation
  13. Diagnostics + visualisations
  14. Save outputs
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(BASE, "extracted_features.csv")
OUTPUT_CSV  = os.path.join(BASE, "cleaned_features_v2.csv")
REPORT_TXT  = os.path.join(BASE, "preprocessing_report.txt")
VIZ_DIR     = os.path.join(BASE, "preprocessing_visualizations")
SKEW_CSV    = os.path.join(BASE, "skewness_analysis.csv")
CLIP_CSV    = os.path.join(BASE, "clipping_summary.csv")
os.makedirs(VIZ_DIR, exist_ok=True)

report_lines = []

def sec(title, w=68):
    line = f"\n{'='*w}\n  {title}\n{'='*w}"
    print(line); report_lines.append(line)

def log(msg):
    m = f"  {msg}"
    print(m); report_lines.append(m)

# ── Feature lists ─────────────────────────────────────────────
# Features to apply log1p transformation (heavy-tailed counts / times)
LOG_TRANSFORM_COLS = [
    "pr_review_count",
    "review_comments_count",
    "commit_count",
    "total_code_changes",
    "rework_count",
    "timeline_interruptions",
    "review_change_requests",
    "pr_completion_time_hours",
    "pr_lifecycle_duration_hours",
    "lines_added",
    "lines_deleted",
    "files_changed_count",
    "bug_fix_commit_count",
    "repo_stars",
    "repo_size_kb",
    "contributor_count",
]

# Features NOT transformed (already bounded, binary, or categorical)
NO_TRANSFORM = [
    "merge_success", "approval_status", "bug_fix_indicator",
    "high_context_changes", "large_pr_flag", "ai_generated_commit",
    "ai_generated_commit_count", "ai_contribution_ratio",
    "issue_resolution_count", "review_turnaround_time_hours",
    "comment_density",
]

REDUNDANT_COLS = [
    "review_friction",
    "large_diff_complexity",
    "repeated_review_cycles",
    "ai_assisted_pr",
]

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD & AUDIT
# ══════════════════════════════════════════════════════════════
sec("STEP 1 — LOAD & AUDIT")
df = pd.read_csv(INPUT_CSV)
log(f"Loaded        : {df.shape[0]:,} rows x {df.shape[1]} columns")
log(f"Total nulls   : {df.isnull().sum().sum():,}")
log(f"Tool counts   : {df['ai_tool_type'].value_counts().to_dict()}")
log(f"Merge rate    : {df['merge_success'].mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# STEP 2 — DUPLICATE REMOVAL
# ══════════════════════════════════════════════════════════════
sec("STEP 2 — DUPLICATE REMOVAL")
before = len(df)
df = df.drop_duplicates(subset=["pr_id"], keep="first")
log(f"Duplicates removed : {before - len(df)}")
log(f"Rows remaining     : {len(df):,}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — STRUCTURAL VALIDATION
# ══════════════════════════════════════════════════════════════
sec("STEP 3 — STRUCTURAL VALIDATION")
before = len(df)
df = df.dropna(subset=["pr_id", "repo_url", "ai_tool_type"])
log(f"Invalid rows dropped : {before - len(df)}")
log(f"Rows remaining       : {len(df):,}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — MISSING VALUE IMPUTATION
# ══════════════════════════════════════════════════════════════
sec("STEP 4 — MISSING VALUE IMPUTATION")

log("pr_completion_time_hours -> group median (ai_tool_type x merge_success)")
grp_med = df.groupby(["ai_tool_type","merge_success"])["pr_completion_time_hours"].transform("median")
df["pr_completion_time_hours"] = df["pr_completion_time_hours"].fillna(grp_med)
df["pr_completion_time_hours"] = df["pr_completion_time_hours"].fillna(df["pr_completion_time_hours"].median())
log(f"  Remaining nulls: {df['pr_completion_time_hours'].isnull().sum()}")

log("review_turnaround_time_hours -> has_review flag + fill 0")
df["has_review"] = df["review_turnaround_time_hours"].notna().astype(int)
df["review_turnaround_time_hours"] = df["review_turnaround_time_hours"].fillna(0)
log(f"  PRs with review: {df['has_review'].sum():,}")

log("comment_density -> fill 0")
df["comment_density"] = df["comment_density"].fillna(0)

log("ai_contribution_ratio -> fill 0")
df["ai_contribution_ratio"] = df["ai_contribution_ratio"].fillna(0)

log("programming_language -> mode per tool group")
gm = df["programming_language"].mode()[0]
df["programming_language"] = (
    df.groupby("ai_tool_type")["programming_language"]
    .transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else gm))
)
df["programming_language"] = df["programming_language"].fillna(gm)

log(f"\nTotal nulls after imputation: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════
# STEP 5 — SKEWNESS ANALYSIS BEFORE TRANSFORMATION
# ══════════════════════════════════════════════════════════════
sec("STEP 5 — SKEWNESS ANALYSIS (BEFORE LOG TRANSFORMATION)")

cols_exist = [c for c in LOG_TRANSFORM_COLS if c in df.columns]
skew_before = {}
kurt_before  = {}
for col in cols_exist:
    skew_before[col] = df[col].skew()
    kurt_before[col]  = df[col].kurtosis()
    log(f"  {col:<38} skew={skew_before[col]:>8.3f}  kurt={kurt_before[col]:>10.3f}")

# ══════════════════════════════════════════════════════════════
# STEP 6 — LOG1P TRANSFORMATION
# ══════════════════════════════════════════════════════════════
sec("STEP 6 — LOG1P TRANSFORMATION  [X' = log(1 + X)]")
log("""
  WHY: Software engineering behavioral metrics are naturally heavy-tailed.
  High values (e.g. 27 review rounds) are genuine productivity signals,
  NOT noise. log(1+X) compresses extreme tails while fully preserving
  the relative ordering and non-zero behavioral information.
  
  Applied to: count-based and time-based features only.
  NOT applied to: binary flags, ratios already in [0,1].
""")

log_transformed = {}
for col in cols_exist:
    original = df[col].copy()
    df[col] = np.log1p(df[col])
    log_transformed[col] = {
        "orig_min" : original.min(),
        "orig_max" : original.max(),
        "trans_min": df[col].min(),
        "trans_max": df[col].max(),
    }
    log(f"  {col:<38} [{original.min():.2f}, {original.max():.2f}] -> [{df[col].min():.4f}, {df[col].max():.4f}]")

# ══════════════════════════════════════════════════════════════
# STEP 7 — GLOBAL PERCENTILE CLIPPING (1st - 99th)
# ══════════════════════════════════════════════════════════════
sec("STEP 7 — GLOBAL PERCENTILE CLIPPING (1st-99th, NOT group-wise)")
log("""
  WHY GLOBAL (not per-tool):
  Per-tool clipping collapses variance when one group has IQR=0.
  Global thresholds keep tool distributions comparable and only
  remove truly extreme outliers (top/bottom 1% of entire dataset).
""")

clip_summary = []
for col in cols_exist:
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    before_clip = df[col].copy()
    df[col] = df[col].clip(lower=lo, upper=hi)
    n_lo = (before_clip < lo).sum()
    n_hi = (before_clip > hi).sum()
    clip_summary.append({
        "Feature": col,
        "p01_threshold": round(lo, 6),
        "p99_threshold": round(hi, 6),
        "clipped_low" : n_lo,
        "clipped_high": n_hi,
        "total_clipped": n_lo + n_hi,
    })
    log(f"  {col:<38} p01={lo:.4f}  p99={hi:.4f}  clipped={n_lo+n_hi}")

clip_df = pd.DataFrame(clip_summary)
clip_df.to_csv(CLIP_CSV, index=False)
log(f"\n  Clipping summary saved -> {CLIP_CSV}")

# ══════════════════════════════════════════════════════════════
# STEP 8 — SKEWNESS ANALYSIS AFTER TRANSFORMATION
# ══════════════════════════════════════════════════════════════
sec("STEP 8 — SKEWNESS ANALYSIS (AFTER TRANSFORMATION)")

skew_rows = []
for col in cols_exist:
    s_after = df[col].skew()
    k_after = df[col].kurtosis()
    s_before = skew_before[col]
    reduction = abs(s_before) - abs(s_after)
    log(f"  {col:<38} before={s_before:>8.3f} -> after={s_after:>7.3f}  reduction={reduction:>7.3f}")
    skew_rows.append({
        "Feature"          : col,
        "Skew_Before"      : round(s_before, 4),
        "Skew_After"       : round(s_after, 4),
        "Skew_Reduction"   : round(reduction, 4),
        "Kurt_Before"      : round(kurt_before[col], 4),
        "Kurt_After"       : round(k_after, 4),
        "Trans_Min"        : round(log_transformed[col]["trans_min"], 6),
        "Trans_Max"        : round(log_transformed[col]["trans_max"], 6),
        "Orig_Min"         : round(log_transformed[col]["orig_min"], 4),
        "Orig_Max"         : round(log_transformed[col]["orig_max"], 4),
    })

skew_df = pd.DataFrame(skew_rows)
skew_df.to_csv(SKEW_CSV, index=False)
log(f"\n  Skewness analysis saved -> {SKEW_CSV}")

# ══════════════════════════════════════════════════════════════
# STEP 9 — CATEGORICAL ENCODING
# ══════════════════════════════════════════════════════════════
sec("STEP 9 — CATEGORICAL ENCODING")

df["is_claude_code"] = (df["ai_tool_type"] == "Claude_Code").astype(int)
log(f"  is_claude_code: Claude={df['is_claude_code'].sum()}  Codex={(df['is_claude_code']==0).sum()}")

TOP_N = 10
top_langs = df["programming_language"].value_counts().nlargest(TOP_N).index.tolist()
df["lang_bucketed"] = df["programming_language"].apply(lambda x: x if x in top_langs else "Other")
lang_dummies = pd.get_dummies(df["lang_bucketed"], prefix="lang", dtype=int)
df = pd.concat([df, lang_dummies], axis=1)
df.drop(columns=["lang_bucketed"], inplace=True)
log(f"  One-hot langs: {list(lang_dummies.columns)}")

# ══════════════════════════════════════════════════════════════
# STEP 10 — SANITY CORRECTIONS
# ══════════════════════════════════════════════════════════════
sec("STEP 10 — SANITY CORRECTIONS")

neg = (df["pr_completion_time_hours"] < 0).sum()
if neg: df.loc[df["pr_completion_time_hours"] < 0, "pr_completion_time_hours"] = 0
log(f"  pr_completion_time_hours negative values corrected: {neg}")

changed = (df["ai_contribution_ratio"].clip(0,1) != df["ai_contribution_ratio"]).sum()
df["ai_contribution_ratio"] = df["ai_contribution_ratio"].clip(0, 1)
log(f"  ai_contribution_ratio clipped to [0,1]: {changed}")

# total_code_changes: recompute from original lines_added + lines_deleted
# Both were log-transformed, so reconstruct additively in log-space is wrong.
# We keep the separately-transformed columns. total_code_changes was
# independently log-transformed, which is the correct approach.
log("  total_code_changes: independently log-transformed (consistent)")

# ══════════════════════════════════════════════════════════════
# STEP 11 — DROP REDUNDANT COLUMNS
# ══════════════════════════════════════════════════════════════
sec("STEP 11 — DROP REDUNDANT COLUMNS")
to_drop = [c for c in REDUNDANT_COLS if c in df.columns]
df.drop(columns=to_drop, inplace=True)
log(f"  Dropped: {to_drop}")
log(f"  Remaining columns: {df.shape[1]}")

# ══════════════════════════════════════════════════════════════
# STEP 12 — DATA TYPE OPTIMISATION
# ══════════════════════════════════════════════════════════════
sec("STEP 12 — DATA TYPE OPTIMISATION")

# After log transformation, count features are now floats
float_cols = cols_exist + [
    "review_turnaround_time_hours", "comment_density",
    "ai_contribution_ratio",
]
int_cols = [
    "merge_success","approval_status","bug_fix_indicator","high_context_changes",
    "large_pr_flag","ai_generated_commit","ai_generated_commit_count",
    "has_review","is_claude_code",
]
for col in float_cols:
    if col in df.columns:
        df[col] = df[col].astype(float).round(6)
for col in int_cols:
    if col in df.columns:
        df[col] = df[col].round().astype(int)

log(f"  Float columns: {len([c for c in float_cols if c in df.columns])}")
log(f"  Int columns  : {len([c for c in int_cols if c in df.columns])}")

# ══════════════════════════════════════════════════════════════
# STEP 13 — VARIANCE PRESERVATION VALIDATION
# ══════════════════════════════════════════════════════════════
sec("STEP 13 — VARIANCE PRESERVATION VALIDATION")
log("""
  Check: Are behavioral signals preserved across tools?
  A collapsed distribution (std ~ 0) means signal is lost.
""")

key_signals = [c for c in ["pr_review_count","review_comments_count",
               "rework_count","timeline_interruptions","ai_contribution_ratio",
               "review_change_requests"] if c in df.columns]

for tool in df["ai_tool_type"].unique():
    grp = df[df["ai_tool_type"] == tool]
    log(f"\n  Tool: {tool}  (N={len(grp):,})")
    for col in key_signals:
        std = grp[col].std()
        mx  = grp[col].max()
        log(f"    {col:<35} std={std:.6f}  max={mx:.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 14 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════

sec("STEP 14 — GENERATING VISUALISATIONS")

raw = pd.read_csv(INPUT_CSV)
raw = raw.drop_duplicates(subset=["pr_id"], keep="first")
raw["pr_completion_time_hours"] = raw["pr_completion_time_hours"].fillna(raw["pr_completion_time_hours"].median())
raw["review_turnaround_time_hours"] = raw["review_turnaround_time_hours"].fillna(0)
raw["comment_density"] = raw["comment_density"].fillna(0)
raw["ai_contribution_ratio"] = raw["ai_contribution_ratio"].fillna(0)

PLOT_COLS = [c for c in [
    "pr_review_count","review_comments_count","commit_count",
    "total_code_changes","rework_count","timeline_interruptions",
    "review_change_requests","pr_completion_time_hours","pr_lifecycle_duration_hours",
] if c in raw.columns and c in df.columns]

PALETTE = {"Claude_Code":"#5B8FF9","OpenAI_Codex":"#5AD8A6"}
tools   = df["ai_tool_type"].unique()

# ── Plot A: Before vs After histograms ────────────────────────
n_cols = min(4, len(PLOT_COLS))
n_rows = (len(PLOT_COLS) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(20, n_rows * 8))
axes = axes.flatten()

for i, col in enumerate(PLOT_COLS):
    ax_before = axes[i * 2] if n_rows * 2 > 1 else axes[i]
    ax_after  = axes[i * 2 + 1] if n_rows * 2 > 1 else axes[i + n_cols]
    raw[col].dropna().hist(ax=ax_before, bins=50, color="#e74c3c", alpha=0.7, edgecolor="none")
    ax_before.set_title(f"{col}\nBEFORE (skew={raw[col].skew():.2f})", fontsize=9)
    ax_before.set_ylabel("Count")
    df[col].dropna().hist(ax=ax_after, bins=50, color="#2ecc71", alpha=0.7, edgecolor="none")
    ax_after.set_title(f"AFTER log1p+clip (skew={df[col].skew():.2f})", fontsize=9)
    ax_after.set_ylabel("Count")

for j in range(len(PLOT_COLS) * 2, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Before vs After Preprocessing — Distribution Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
path_A = os.path.join(VIZ_DIR, "01_before_after_histograms.png")
plt.savefig(path_A, dpi=130, bbox_inches="tight")
plt.close()
log(f"  Saved: {path_A}")

# ── Plot B: Boxplots by tool — before vs after ────────────────
fig, axes = plt.subplots(2, len(PLOT_COLS), figsize=(22, 8))
for i, col in enumerate(PLOT_COLS):
    data_before = [raw[raw["ai_tool_type"] == t][col].dropna().values for t in tools]
    bp = axes[0, i].boxplot(data_before, patch_artist=True, labels=tools)
    for patch, t in zip(bp["boxes"], tools):
        patch.set_facecolor(PALETTE.get(t, "gray")); patch.set_alpha(0.7)
    axes[0, i].set_title(f"{col}\nBEFORE", fontsize=8)
    axes[0, i].tick_params(axis="x", labelsize=7, rotation=20)

    data_after = [df[df["ai_tool_type"] == t][col].dropna().values for t in tools]
    bp2 = axes[1, i].boxplot(data_after, patch_artist=True, labels=tools)
    for patch, t in zip(bp2["boxes"], tools):
        patch.set_facecolor(PALETTE.get(t, "gray")); patch.set_alpha(0.7)
    axes[1, i].set_title(f"AFTER", fontsize=8)
    axes[1, i].tick_params(axis="x", labelsize=7, rotation=20)

plt.suptitle("Boxplots — Before vs After (by AI Tool)", fontsize=14, fontweight="bold")
plt.tight_layout()
path_B = os.path.join(VIZ_DIR, "02_boxplots_before_after.png")
plt.savefig(path_B, dpi=130, bbox_inches="tight")
plt.close()
log(f"  Saved: {path_B}")

# ── Plot C: Skewness reduction bar chart ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
skew_df_sorted = skew_df.sort_values("Skew_Before", ascending=False)
x = range(len(skew_df_sorted))
axes[0].bar(x, skew_df_sorted["Skew_Before"], color="#e74c3c", alpha=0.8, label="Before")
axes[0].bar(x, skew_df_sorted["Skew_After"],  color="#2ecc71", alpha=0.8, label="After")
axes[0].set_xticks(list(x)); axes[0].set_xticklabels(skew_df_sorted["Feature"], rotation=45, ha="right", fontsize=8)
axes[0].set_title("Skewness Before vs After"); axes[0].set_ylabel("Skewness"); axes[0].legend()

axes[1].bar(x, skew_df_sorted["Skew_Reduction"], color="#5B8FF9", alpha=0.8)
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(skew_df_sorted["Feature"], rotation=45, ha="right", fontsize=8)
axes[1].set_title("Skewness Reduction (|before| - |after|)"); axes[1].set_ylabel("Reduction")

plt.suptitle("Skewness Analysis — log1p Impact", fontsize=13, fontweight="bold")
plt.tight_layout()
path_C = os.path.join(VIZ_DIR, "03_skewness_analysis.png")
plt.savefig(path_C, dpi=130, bbox_inches="tight")
plt.close()
log(f"  Saved: {path_C}")

# ── Plot D: QQ plots for key features ────────────────────────
qq_cols = [c for c in ["pr_review_count","commit_count","total_code_changes",
           "pr_completion_time_hours"] if c in df.columns]
fig, axes = plt.subplots(len(qq_cols), 2, figsize=(12, len(qq_cols)*4))
for i, col in enumerate(qq_cols):
    stats.probplot(raw[col].dropna(),  dist="norm", plot=axes[i, 0])
    axes[i, 0].set_title(f"{col} — BEFORE (QQ)", fontsize=9)
    stats.probplot(df[col].dropna(),   dist="norm", plot=axes[i, 1])
    axes[i, 1].set_title(f"{col} — AFTER (QQ)", fontsize=9)
plt.suptitle("QQ Plots — Before vs After Transformation", fontsize=13, fontweight="bold")
plt.tight_layout()
path_D = os.path.join(VIZ_DIR, "04_qq_plots.png")
plt.savefig(path_D, dpi=130, bbox_inches="tight")
plt.close()
log(f"  Saved: {path_D}")

# ── Plot E: Tool distribution comparison (KDE with fallback) ──────
fig, axes = plt.subplots(1, len(PLOT_COLS), figsize=(22, 5))
for i, col in enumerate(PLOT_COLS):
    for t in tools:
        g = df[df["ai_tool_type"] == t][col].dropna()
        if g.std() > 0:
            g.plot.kde(ax=axes[i], label=t, color=PALETTE.get(t,"gray"), lw=2)
        else:
            axes[i].hist(g, bins=10, alpha=0.5, label=t, color=PALETTE.get(t,"gray"))
    axes[i].set_title(col, fontsize=8); axes[i].legend(fontsize=7)
plt.suptitle("Tool Distribution Comparison — After Preprocessing", fontsize=13, fontweight="bold")
plt.tight_layout()
path_E = os.path.join(VIZ_DIR, "05_tool_distributions.png")
plt.savefig(path_E, dpi=130, bbox_inches="tight")
plt.close()
log(f"  Saved: {path_E}")

# ══════════════════════════════════════════════════════════════
# STEP 15 — SAVE cleaned_features_v2.csv
# ══════════════════════════════════════════════════════════════

sec("STEP 15 — SAVE cleaned_features_v2.csv")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
log(f"  Saved: {OUTPUT_CSV}")
log(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════
# STEP 16 — FINAL VALIDATION REPORT
# ══════════════════════════════════════════════════════════════

sec("STEP 16 — FINAL VALIDATION REPORT")

df_v = pd.read_csv(OUTPUT_CSV)
nulls = df_v.isnull().sum().sum()
cells = df_v.shape[0] * df_v.shape[1]
completeness = (1 - nulls/cells) * 100

log(f"  Rows         : {df_v.shape[0]:,}")
log(f"  Columns      : {df_v.shape[1]}")
log(f"  Total nulls  : {nulls}")
log(f"  Completeness : {completeness:.4f}%")
log(f"\n  Column list:")
for i, col in enumerate(df_v.columns, 1):
    dtype = str(df_v[col].dtype)
    nuls  = df_v[col].isnull().sum()
    log(f"    {i:>2}. {col:<42} [{dtype:<8}] nulls={nuls}")

log("\n  Variance check for key behavioral signals:")
for col in ["pr_review_count","review_comments_count","rework_count",
            "timeline_interruptions","ai_contribution_ratio"]:
    if col in df_v.columns:
        log(f"    {col:<38} std={df_v[col].std():.6f}  max={df_v[col].max():.4f}  unique={df_v[col].nunique()}")

log("\n  Tool distribution:")
log(df_v["ai_tool_type"].value_counts().to_string())

# Save report
report_lines.append("\n" + "="*68)
report_lines.append("  PREPROCESSING COMPLETE — cleaned_features_v2.csv is ready")
report_lines.append("="*68)
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"\n  Report saved -> {REPORT_TXT}")

print("\n" + "="*68)
if nulls == 0:
    print("  ALL CHECKS PASSED — cleaned_features_v2.csv is ready")
else:
    print(f"  WARNING: {nulls} nulls remain")
print("="*68)
print(f"\n  Output files:")
print(f"    {OUTPUT_CSV}")
print(f"    {REPORT_TXT}")
print(f"    {SKEW_CSV}")
print(f"    {CLIP_CSV}")
print(f"    {VIZ_DIR}/  (5 visualisation files)")
