# Developer Productivity Score (DPS)

A hierarchical, entropy-weighted composite index that measures developer
productivity across five dimensions — and a Flask web app that computes it
two ways and compares the results.

**Live demo:** https://huggingface.co/spaces/TatsatPathak/Statistical-Model

**Research paper:** *Measuring the Impact of AI-Assisted Coding on Developer
Productivity: A Multi-Dimensional Composite Index Approach* — the full paper
behind this project (methodology, hypothesis tests, and results below are
taken from it).

## What is DPS?

Most productivity measures use a single number — lines of code, task speed,
test pass rate. DPS argues productivity is multi-dimensional and measures it
as one: **14 behavioral metrics across five dimensions**, combined into a
single normalized score in [0, 1].

```
DPS = (TES + CQS + CLS + AIS + RCS) / 5
```

| Dimension | What it captures | Features |
| --- | --- | --- |
| **TES** — Task Efficiency | How fast and completely work gets done | pr_completion_time_hours, commit_count, issue_resolution_count |
| **CQS** — Code Quality | Whether the code passes peer review cleanly | approval_status, bug_fix_indicator |
| **CLS** — Cognitive Load | Complexity and interruptions the PR carries | high_context_changes, timeline_interruptions, pr_lifecycle_duration_hours |
| **AIS** — AI Assistance | How effectively AI code is used — and what it costs | ai_contribution_intensity, ai_review_friction, ai_rework_ratio |
| **RCS** — Repository Context | The maturity of the repo the PR lands in | repo_stars, contributor_count, repo_size_kb |

Each dimension is a weighted sum of its normalized features. Feature weights
come from **Shannon Information Entropy** (within each dimension) with equal
weight across the five dimensions, following the OECD Handbook on Constructing
Composite Indicators (2008). No expert-assigned weights, no black box.

## The research in brief

### Dataset

| | |
| --- | --- |
| Total pull requests | **22,258** |
| OpenAI Codex | 21,799 PRs (97.9%) |
| Claude Code | 459 PRs (2.1%) |
| Overall merge rate | 82.1% |
| Raw feature columns | 45 |
| Null values after cleaning | 0 (median imputation) |

14 of the 45 raw features were retained for the index. Dropped, with reasons:
`merge_success` (82.1% constant — near-zero variance), `total_code_changes`
(redundant with `commit_count`), raw `ai_contribution_ratio` (superseded by
the richer `ai_contribution_intensity`), and `review_comments_count`,
`review_change_requests`, `ai_generated_commit` (near-zero variance).

### Three new AI behavioral features

Raw AI usage percentages don't say whether the AI help was efficient, costly,
or wasteful. Three engineered features separate those:

- **ai_contribution_intensity** — AI code volume per unit of work:
  `(ai_contribution_ratio × total_code_changes) / (commit_count + 1)`.
  Turns the ratio into an absolute volume and penalizes fragmented work.
- **ai_review_friction** — review overhead per unit of AI code:
  `(pr_review_count + review_comments_count) / (ai_contribution_ratio + ε)`.
  A PR where AI wrote 10% of the code but caused 10 review rounds scores far
  higher friction than one where AI wrote 90% with the same 10 rounds.
- **ai_rework_ratio** — rewrite tax per unit of AI code:
  `rework_count / (ai_contribution_ratio + ε)`.

### Normalization

Raw GitHub metrics are extremely right-skewed (worst: `review_change_requests`
skewness **38.50**, `pr_lifecycle_duration_hours` 31.33,
`total_code_changes` 29.32). Three methods were compared on the final DPS
distribution:

| Method | Skewness | Kurtosis | Selected |
| --- | --- | --- | --- |
| MinMaxScaler | 1.7499 | 4.6340 | No |
| QuantileTransformer | 1.3363 | 2.2337 | No |
| **RobustScaler** | **1.1351** | 2.3439 | **Yes** |

RobustScaler won on lowest absolute skewness. The pipeline is three steps:
(1) Log1p transform, (2) global 1st–99th percentile clipping,
(3) RobustScaler (median–IQR centering, resistant to outliers). Features with
negative polarity are inverted after scaling. Net effect: worst-feature
skewness **38.5 → 1.1**.

### Entropy weights

Within each dimension, a feature's weight comes from its Shannon entropy
across the 22,258 PRs: features that vary a lot (high discriminating power)
get high weight; features that are nearly constant get near-zero weight.
Final weights sum to 1.0000:

| Dimension | Feature | Entropy | Weight |
| --- | --- | --- | --- |
| CQS | approval_status | 0.6853 | 0.1856 |
| TES | issue_resolution_count | 0.5708 | 0.1840 |
| AIS | ai_contribution_intensity | 0.9069 | 0.1744 |
| CLS | pr_lifecycle_duration_hours | 0.9691 | 0.1137 |
| RCS | repo_size_kb | 0.8003 | 0.0859 |
| CLS | high_context_changes | 0.9796 | 0.0752 |
| RCS | repo_stars | 0.8626 | 0.0591 |
| RCS | contributor_count | 0.8722 | 0.0550 |
| AIS | ai_review_friction | 0.9910 | 0.0169 |
| CQS | bug_fix_indicator | 0.9756 | 0.0144 |
| TES | pr_completion_time_hours | 0.9696 | 0.0131 |
| CLS | timeline_interruptions | 0.9970 | 0.0111 |
| AIS | ai_rework_ratio | 0.9954 | 0.0087 |
| TES | commit_count | 0.9932 | 0.0029 |

`issue_resolution_count` dominates TES at 92% because most PRs resolve zero
issues — the few that do carry enormous discriminating information. Same
story for `ai_contribution_intensity` in AIS (87.2%).

### Validation

- **Spearman correlation:** all 14 features correlate significantly with DPS
  (p < 0.0001). Strongest single predictor: `ai_contribution_intensity`
  (ρ = +0.5202).
- **Multicollinearity (VIF):** max VIF 2.92 (`ai_review_friction`), well below
  the danger threshold of 10. Zero features above it.
- **Sensitivity:** a +10% perturbation of each feature moves DPS by exactly
  `weight × 0.1` — the entropy weights and the model's sensitivity agree
  perfectly, confirming self-consistency.
- **Variance explained:** AIS 31.97%, CLS 29.56%, RCS 23.09%, CQS 11.51%,
  TES 3.88%. AI assistance is the largest single driver of DPS variance.
- **Cronbach's Alpha:** RCS 0.74 (acceptable); TES 0.24, CQS −0.25,
  CLS 0.32, AIS 0.46 (low, as expected — these dimensions sample *different
  facets* of productivity, not parallel measurements of the same thing;
  Greco et al. 2019).

## Key findings

DPS distribution across the 22,258 PRs: mean **0.2865**, median 0.2694,
SD 0.0812. High productivity (DPS ≥ 0.60): **55 PRs (0.2%)**. Moderate
(0.40–0.59): 1,864 (8.4%). Low (< 0.40): 20,339 (**91.4%**). The Low class
dominates because scoring high on all five dimensions at once is rare — most
PRs do well on one or two (typically CLS, mean 0.7554) and poorly on others
(typically TES, mean 0.0713, and CQS, mean 0.0961).

| H# | Hypothesis | Test | Statistic | p-value | Verdict |
| --- | --- | --- | --- | --- | --- |
| H1 | High-AI PRs achieve higher DPS | Mann-Whitney U | U = 37,012,489 | < 0.001 | **Supported** (d = 0.31) |
| H2 | Claude Code DPS > OpenAI Codex | Welch's t | t = 4.20 | 0.000032 | **Supported** (d = 0.24) |
| H3 | AI review friction harms DPS | Spearman + partial | ρ = +0.031 | 0.000004 | **Not supported** |
| H4 | AI rework ratio harms DPS | Spearman ρ | ρ = −0.042 | < 0.001 | **Supported** |
| H5 | Task Efficiency contributes to DPS | Spearman ρ | ρ = +0.244 | < 0.001 | **Supported** |
| H6 | Cognitive Load links to DPS | Spearman ρ | ρ = +0.225 | < 0.001 | **Supported** |

**Five of six hypotheses supported.** The one that failed is the most
interesting: H3 predicted review friction would *hurt* productivity, but the
correlation is positive (ρ = +0.031) — and it gets *more* positive
(ρ = +0.161) after controlling for AI usage level. The paper calls this the
**Active Review Culture Hypothesis**: repositories that review AI-generated
code rigorously produce higher-productivity PRs overall. Review friction
isn't a cost — it's a signal of disciplined engineering. AI tools don't
substitute for rigorous review; they demand it.

Caveat on H2: Claude Code's higher mean DPS (+0.028, +9.8%) comes with a
confound — Claude Code PRs sit in larger, more mature repos (RCS ≈ 0.41 vs
0.16 for Codex), and the Claude sample is small (n = 459). Tool superiority
needs a balanced, controlled comparison before it can be claimed causally.

## XGBoost surrogate — independent validation

An XGBoost regressor (500 trees, max depth 6, learning rate 0.05) was trained
on the same 22,258 PRs with DPS as the target, as a black-box check on the
white-box index. It near-perfectly reverse-engineered the formula:

| Metric | Value |
| --- | --- |
| R² | **0.9981** |
| RMSE | **0.00356** |
| MAE | 0.00190 |
| Residual mean | 0.00002 (no systematic bias) |

Features: **38** (14 DPS pipeline features + 13 extra behavioral features +
11 language dummies). Train/test split: 17,806 / 4,452. XGBoost feature
importance by gain aligns with the entropy-derived weights (top: approval
status, issue resolution count, high context changes, AI contribution
intensity) — two independent methods agreeing on what matters.

## Repository structure

```
.
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── app.py                          # Flask entry point
├── extract_features.py             # PR metadata -> extracted_features.csv (45 raw columns)
├── clean_dataset_v2.py             # Log1p + percentile clipping -> cleaned_features_v2.csv
├── dps_pipeline_v2.py              # DPS construction, entropy weights, H1-H6 tests
├── dps_xgboost_model.ipynb         # XGBoost surrogate training and evaluation
├── models/
│   ├── __init__.py
│   ├── statistical_model.py        # White-box DPS computation (from dps_pipeline_v2.py)
│   └── xgboost_model.py            # XGBoost model wrapper (38 features)
├── trained_models/
│   └── dps_xgboost_model.json      # Pre-trained XGBoost model
└── templates/
    ├── index.html                  # Main prediction page
    └── about.html                  # Model explanation page
```

## Reproducing the pipeline

```bash
python extract_features.py     # reads Claude_Code/ and OpenAI_Codex/ folders
python clean_dataset_v2.py     # -> cleaned_features_v2.csv (22,258 rows, 0 nulls)
python dps_pipeline_v2.py      # -> dps_final.csv + validation reports + weight tables
```

Then run `dps_xgboost_model.ipynb` to train the surrogate and regenerate
`trained_models/dps_xgboost_model.json`.

### Data

The raw datasets (1.43 GB and 170 MB) are **not in this repo** — GitHub
blocks files over 100 MB. They are hosted externally; download links to be
added here.

## The web app

A Flask app that computes DPS two ways for the same PR and compares them
side by side.

### Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (developed on 3.12).

### Running

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser. (A live demo is also hosted on
Hugging Face Spaces — see the link at the top.)

### How it works

A single form collects 15 PR characteristics grouped by dimension
(TES, CQS, CLS, AIS, RCS). The same inputs go to **both** models at once:

- **Statistical Model** (white-box) applies the exact entropy-weighted
  formula from `dps_pipeline_v2.py`.
- **XGBoost Model** (black-box) loads the pre-trained model and runs
  inference on the engineered 38-feature vector.

### Output

Side-by-side comparison showing DPS from both models, productivity
classification (High / Moderate / Low), dimension score breakdown
(statistical model), feature contribution chart, feature importance chart
(XGBoost), radar chart of dimension scores, a full explainability summary,
and a model comparison table.

### Model details

| | Statistical model | XGBoost model |
| --- | --- | --- |
| Type | Unsupervised composite index | Supervised regression (gradient boosted trees) |
| Weighting | Shannon entropy (within-dimension) + equal (across dimensions) | Learned from data |
| Normalization | Log1p + clipping + RobustScaler | Same pipeline |
| Training data | 22,258 GitHub PRs | 22,258 GitHub PRs (DPS as target) |
| Performance | — | R² = 0.9981, RMSE = 0.00356 |
| Features | 14 | 38 (14 pipeline + 13 behavioral + 11 language dummies) |
| Reference | OECD/JRC Handbook on Composite Indicators (2008) | Chen & Guestrin (2016) |

## Limitations

From the paper, stated plainly:

1. **Sample imbalance** — OpenAI Codex is 97.9% of the data, Claude Code
   2.1%. H2 is significant but the Claude group (n = 459) is too small for
   fully generalizable inter-tool conclusions.
2. **Observational design** — associations, not causation. High AI
   contribution is *associated* with higher DPS; the data can't prove AI
   *causes* it.
3. **Data completeness** — `ai_contribution_ratio` depends on self-reporting
   through commit messages or CI annotations; some misclassification is
   possible.
4. **Static weights** — Shannon entropy weights are computed from this
   dataset and may differ in other organizational contexts.
5. **Cronbach's Alpha** — four of five dimensions fall below the conventional
   0.70 threshold (see Validation above for why this is expected for
   heterogeneous composite indices).

Future work: longitudinal (before/after AI adoption) designs, causal
inference (propensity score matching / instrumental variables), SHAP
explanations from the XGBoost model, more AI tools with balanced samples,
confirmatory factor analysis, and context-adaptive entropy weights.

## References

- OECD/JRC (2008). *Handbook on Constructing Composite Indicators:
  Methodology and User Guide.* OECD Publishing.
- Shannon, C. E. (1948). A Mathematical Theory of Communication.
  *Bell System Technical Journal*, 27(3–4).
- Greco, S., Ishizaka, A., Tasiou, M., & Torrisi, G. (2019). On the
  methodological framework of composite indices. *Social Indicators
  Research*, 141(1).
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
  *Proc. KDD 2016*.
- Forsgren, N., Storey, M.-A., et al. (2021). The SPACE of Developer
  Productivity. *ACM Queue*, 19(1).
- Murphy-Hill, E., et al. (2019). Software Developers' Perceptions of
  Productivity. *Proc. ESEC/FSE 2019*.
- Sadowski, C., et al. (2018). What Improves Developer Productivity at
  Google? Code Quality. *Proc. ICSE-SEIP 2018*.
- Peng, S., et al. (2023). The Impact of AI on Developer Productivity:
  Evidence from GitHub Copilot. arXiv:2302.06590.
- Siddiq, M. L., et al. (2024). AI-Assisted Programming Decreases the
  Productivity of Experienced Developers. arXiv preprint.

Full reference list (32 sources) is in the research paper.
