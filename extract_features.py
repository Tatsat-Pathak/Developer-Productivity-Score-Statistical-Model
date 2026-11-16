"""
Feature Extraction Script for Developer Productivity Score (DPS) Research
=========================================================================
Extracts features from Claude_Code and OpenAI_Codex datasets as specified in
the research methodology. Each PR becomes one row with all 30 features.

Output: extracted_features.csv (combined) in the Statistical Model directory.
"""

import json
import csv
import os
import re
from datetime import datetime

# ──────────────────────── Configuration ────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS = {
    "Claude_Code": os.path.join(BASE_DIR, "Claude_Code"),
    "OpenAI_Codex": os.path.join(BASE_DIR, "OpenAI_Codex"),
}
OUTPUT_FILE = os.path.join(BASE_DIR, "extracted_features.csv")

# Bug-fix keyword patterns for commit messages
BUG_FIX_PATTERN = re.compile(
    r"\b(fix|bug|error|patch|hotfix|bugfix|resolve|resolves|resolved|fixing|fixed)\b",
    re.IGNORECASE,
)


# ──────────────────────── Helper Functions ────────────────────────
def parse_dt(s):
    """Parse ISO 8601 datetime string, return datetime or None."""
    if not s or s == "" or s == "None":
        return None
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def hours_diff(dt1, dt2):
    """Return hours between two datetimes, or None if either is None."""
    if dt1 is None or dt2 is None:
        return None
    diff = (dt2 - dt1).total_seconds() / 3600.0
    return round(abs(diff), 4)


def safe_get(d, *keys, default=None):
    """Safely navigate nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def load_json(filepath):
    """Load JSON file with error handling."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {filepath}: {e}")
        return {}


def load_csv_rows(filepath):
    """Load CSV into list of dicts."""
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"  [WARN] Could not load {filepath}: {e}")
    return rows


# ──────────────────────── Feature Extraction ────────────────────────
def extract_features_for_tool(tool_name, tool_dir):
    """Extract all features for a single AI tool's dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {tool_name}")
    print(f"Directory:  {tool_dir}")
    print(f"{'='*60}")

    # ── Load all data sources ──
    print("  Loading prs.csv...")
    prs_rows = load_csv_rows(os.path.join(tool_dir, "prs.csv"))
    print(f"    -> {len(prs_rows)} pull requests")

    print("  Loading pr_commits.json...")
    pr_commits = load_json(os.path.join(tool_dir, "pr_commits.json"))
    print(f"    -> {len(pr_commits)} entries")

    print("  Loading pr_commit_details.json...")
    pr_commit_details = load_json(os.path.join(tool_dir, "pr_commit_details.json"))
    print(f"    -> {len(pr_commit_details)} entries")

    print("  Loading pr_reviews.json...")
    pr_reviews = load_json(os.path.join(tool_dir, "pr_reviews.json"))
    print(f"    -> {len(pr_reviews)} entries")

    print("  Loading pr_review_comments.json...")
    pr_review_comments = load_json(os.path.join(tool_dir, "pr_review_comments.json"))
    print(f"    -> {len(pr_review_comments)} entries")

    print("  Loading pr_timelines.json...")
    pr_timelines = load_json(os.path.join(tool_dir, "pr_timelines.json"))
    print(f"    -> {len(pr_timelines)} entries")

    print("  Loading related_issues.csv...")
    related_issues = load_csv_rows(os.path.join(tool_dir, "related_issues.csv"))
    print(f"    -> {len(related_issues)} issue links")

    print("  Loading gpt_conventional_commits.csv...")
    gpt_commits = load_csv_rows(os.path.join(tool_dir, "gpt_conventional_commits.csv"))
    print(f"    -> {len(gpt_commits)} GPT commits")

    print("  Loading repo_metadata.json...")
    repo_metadata = load_json(os.path.join(tool_dir, "repo_metadata.json"))
    print(f"    -> {len(repo_metadata)} repositories")

    # ── Build lookup indices ──
    # related_issues: group by pr_id
    issues_by_pr = {}
    for row in related_issues:
        pr_id = row.get("pr_id", "")
        issues_by_pr.setdefault(pr_id, []).append(row)

    # gpt_commits: index by PR id
    gpt_by_pr = {}
    for row in gpt_commits:
        pr_id = str(row.get("id", ""))
        gpt_by_pr.setdefault(pr_id, []).append(row)

    # ── Process each PR ──
    features_list = []
    total = len(prs_rows)
    for idx, pr in enumerate(prs_rows):
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f"  Processing PR {idx+1}/{total}...")

        pr_id = str(pr.get("id", ""))
        pr_number = pr.get("number", "")
        repo_url = pr.get("repo_url", "")
        created_at = parse_dt(pr.get("created_at"))
        closed_at = parse_dt(pr.get("closed_at"))
        merged_at = parse_dt(pr.get("merged_at"))

        # JSON key for this PR (format: "PR_ID.json")
        json_key = f"{pr_id}.json"

        # ─────────────────────────────────────────────────────────────
        # 1. TASK EFFICIENCY FEATURES
        # ─────────────────────────────────────────────────────────────

        # PR Completion Time (hours): created_at -> merged_at
        pr_completion_time = hours_diff(created_at, merged_at)

        # Commit Count: from pr_commits.json
        commits_list = pr_commits.get(json_key, [])
        commit_count = len(commits_list)

        # Files Changed, Lines Added, Lines Deleted, Total Code Changes:
        # from pr_commit_details.json
        details_list = pr_commit_details.get(json_key, [])
        files_changed = 0
        lines_added = 0
        lines_deleted = 0
        all_files_seen = set()
        for commit_detail in details_list:
            stats = commit_detail.get("stats", {}) or {}
            lines_added += stats.get("additions", 0) or 0
            lines_deleted += stats.get("deletions", 0) or 0
            for f in commit_detail.get("files", []) or []:
                fname = f.get("filename", "")
                if fname:
                    all_files_seen.add(fname)
        files_changed = len(all_files_seen)
        total_code_changes = lines_added + lines_deleted

        # Merge Success: from merged_at field
        merge_success = 1 if (merged_at is not None) else 0

        # Review Turnaround Time (hours): created_at -> first review submitted_at
        reviews_list = pr_reviews.get(json_key, [])
        review_turnaround = None
        if reviews_list and created_at:
            review_times = []
            for rev in reviews_list:
                rt = parse_dt(rev.get("submitted_at"))
                if rt:
                    review_times.append(rt)
            if review_times:
                first_review = min(review_times)
                review_turnaround = hours_diff(created_at, first_review)

        # Issue Resolution Count: from related_issues.csv
        issue_resolution_count = len(issues_by_pr.get(pr_id, []))

        # PR Lifecycle Duration (hours): first to last event in pr_timelines.json
        timeline_events = pr_timelines.get(json_key, [])
        pr_lifecycle_duration = None
        if timeline_events:
            event_times = []
            for evt in timeline_events:
                et = parse_dt(evt.get("created_at"))
                if et:
                    event_times.append(et)
                # Some timeline events store time in 'committer.date' or 'author.date'
                et2 = parse_dt(safe_get(evt, "committer", "date"))
                if et2:
                    event_times.append(et2)
                et3 = parse_dt(safe_get(evt, "author", "date"))
                if et3:
                    event_times.append(et3)
            if len(event_times) >= 2:
                pr_lifecycle_duration = hours_diff(min(event_times), max(event_times))
            elif len(event_times) == 1 and created_at:
                pr_lifecycle_duration = hours_diff(created_at, event_times[0])

        # ─────────────────────────────────────────────────────────────
        # 2. CODE QUALITY FEATURES
        # ─────────────────────────────────────────────────────────────

        # Review Comments Count: from pr_review_comments.json
        review_comments_list = pr_review_comments.get(json_key, [])
        review_comments_count = len(review_comments_list)

        # PR Review Count: from pr_reviews.json
        pr_review_count = len(reviews_list)

        # Approval Status: 1 if any review has state "APPROVED"
        approval_status = 0
        for rev in reviews_list:
            if rev.get("state", "").upper() == "APPROVED":
                approval_status = 1
                break

        # Rework Count: number of commits after the first review
        rework_count = 0
        if reviews_list and commits_list:
            review_times_all = []
            for rev in reviews_list:
                rt = parse_dt(rev.get("submitted_at"))
                if rt:
                    review_times_all.append(rt)
            if review_times_all:
                first_review_time = min(review_times_all)
                for commit in commits_list:
                    commit_date = parse_dt(safe_get(commit, "commit", "committer", "date"))
                    if commit_date and commit_date > first_review_time:
                        rework_count += 1

        # Bug Fix Indicators: from commit messages in pr_commits.json
        bug_fix_count = 0
        for commit in commits_list:
            msg = safe_get(commit, "commit", "message", default="") or ""
            if BUG_FIX_PATTERN.search(msg):
                bug_fix_count += 1
        bug_fix_indicator = 1 if bug_fix_count > 0 else 0

        # Large PR Flag: >500 total code changes OR >20 files changed
        large_pr_flag = 1 if (total_code_changes > 500 or files_changed > 20) else 0

        # Review Change Requests: count of CHANGES_REQUESTED reviews
        review_change_requests = 0
        for rev in reviews_list:
            if rev.get("state", "").upper() == "CHANGES_REQUESTED":
                review_change_requests += 1

        # Comment Density: review_comments_count / total_code_changes
        comment_density = None
        if total_code_changes > 0:
            comment_density = round(review_comments_count / total_code_changes, 6)

        # ─────────────────────────────────────────────────────────────
        # 3. AI USAGE FEATURES
        # ─────────────────────────────────────────────────────────────

        # AI Tool Type: from folder name
        ai_tool_type = tool_name

        # AI-Assisted PR: binary (always 1 since all PRs are from AI tool folders)
        ai_assisted_pr = 1

        # AI-Generated Commit: count from gpt_conventional_commits.csv
        gpt_entries = gpt_by_pr.get(pr_id, [])
        ai_generated_commit_count = len(gpt_entries)
        ai_generated_commit = 1 if ai_generated_commit_count > 0 else 0

        # AI Contribution Ratio: AI-generated commits / total commits
        ai_contribution_ratio = None
        if commit_count > 0:
            ai_contribution_ratio = round(ai_generated_commit_count / commit_count, 6)

        # ─────────────────────────────────────────────────────────────
        # 4. COGNITIVE LOAD PROXY FEATURES
        # ─────────────────────────────────────────────────────────────

        # Review Friction: review_comments_count (higher = more friction)
        review_friction = review_comments_count

        # Repeated Review Cycles: count of review rounds (distinct review events)
        repeated_review_cycles = pr_review_count

        # High Context Changes: 1 if files_changed > 10 (multiple modules)
        high_context_changes = 1 if files_changed > 10 else 0

        # Timeline Interruptions: count of reopened, labeled, unlabeled,
        # head_ref_force_pushed events etc.
        interruption_events = {
            "reopened", "head_ref_force_pushed", "labeled", "unlabeled",
            "review_dismissed", "convert_to_draft", "ready_for_review",
        }
        timeline_interruptions = 0
        for evt in timeline_events:
            if evt.get("event", "") in interruption_events:
                timeline_interruptions += 1

        # Large Diff Complexity: total_code_changes (as a continuous measure)
        large_diff_complexity = total_code_changes

        # ─────────────────────────────────────────────────────────────
        # 5. REPOSITORY CONTEXT FEATURES
        # ─────────────────────────────────────────────────────────────

        # Look up repo metadata using repo_url
        repo_meta = repo_metadata.get(repo_url, {})

        # Repository Size (KB)
        repo_size = repo_meta.get("size", None)

        # Repository Stars
        repo_stars = repo_meta.get("stargazers_count", None)

        # Programming Language
        programming_language = repo_meta.get("language", None)

        # Contributor Count (using subscribers_count as proxy from API data)
        # Note: GitHub API repo endpoint doesn't directly include contributor count,
        # so we use subscribers_count + forks_count as an activity proxy,
        # or open_issues_count. We'll use the available field.
        contributor_count = repo_meta.get("subscribers_count", None)

        # ─────────────────────────────────────────────────────────────
        # Build the row
        # ─────────────────────────────────────────────────────────────
        row = {
            # Identifiers
            "pr_id": pr_id,
            "pr_number": pr_number,
            "repo_url": repo_url,

            # 1. Task Efficiency Features
            "pr_completion_time_hours": pr_completion_time,
            "commit_count": commit_count,
            "files_changed_count": files_changed,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "total_code_changes": total_code_changes,
            "merge_success": merge_success,
            "review_turnaround_time_hours": review_turnaround,
            "issue_resolution_count": issue_resolution_count,
            "pr_lifecycle_duration_hours": pr_lifecycle_duration,

            # 2. Code Quality Features
            "review_comments_count": review_comments_count,
            "pr_review_count": pr_review_count,
            "approval_status": approval_status,
            "rework_count": rework_count,
            "bug_fix_indicator": bug_fix_indicator,
            "bug_fix_commit_count": bug_fix_count,
            "large_pr_flag": large_pr_flag,
            "review_change_requests": review_change_requests,
            "comment_density": comment_density,

            # 3. AI Usage Features
            "ai_tool_type": ai_tool_type,
            "ai_assisted_pr": ai_assisted_pr,
            "ai_generated_commit": ai_generated_commit,
            "ai_generated_commit_count": ai_generated_commit_count,
            "ai_contribution_ratio": ai_contribution_ratio,

            # 4. Cognitive Load Proxy Features
            "review_friction": review_friction,
            "repeated_review_cycles": repeated_review_cycles,
            "high_context_changes": high_context_changes,
            "timeline_interruptions": timeline_interruptions,
            "large_diff_complexity": large_diff_complexity,

            # 5. Repository Context Features
            "repo_size_kb": repo_size,
            "repo_stars": repo_stars,
            "programming_language": programming_language,
            "contributor_count": contributor_count,
        }
        features_list.append(row)

    print(f"[OK] Extracted features for {len(features_list)} PRs from {tool_name}")
    return features_list


# ──────────────────────── Main Execution ────────────────────────
def main():
    print("=" * 60)
    print("  FEATURE EXTRACTION FOR DPS RESEARCH")
    print("  Extracting from Claude_Code & OpenAI_Codex")
    print("=" * 60)

    all_features = []

    for tool_name, tool_dir in TOOLS.items():
        if not os.path.exists(tool_dir):
            print(f"\n[ERROR] Directory not found: {tool_dir}")
            continue
        features = extract_features_for_tool(tool_name, tool_dir)
        all_features.extend(features)

    if not all_features:
        print("\n[ERROR] No features extracted!")
        return

    # ── Write combined CSV ──
    fieldnames = list(all_features[0].keys())

    print(f"\n{'='*60}")
    print(f"  Writing {len(all_features)} rows to: {OUTPUT_FILE}")
    print(f"  Columns: {len(fieldnames)}")
    print(f"{'='*60}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)

    # ── Summary Statistics ──
    claude_count = sum(1 for r in all_features if r["ai_tool_type"] == "Claude_Code")
    codex_count = sum(1 for r in all_features if r["ai_tool_type"] == "OpenAI_Codex")
    merged_count = sum(1 for r in all_features if r["merge_success"] == 1)
    approved_count = sum(1 for r in all_features if r["approval_status"] == 1)
    bugfix_count = sum(1 for r in all_features if r["bug_fix_indicator"] == 1)

    print(f"\n{'='*60}")
    print(f"  EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total PRs extracted:          {len(all_features)}")
    print(f"  Claude Code PRs:              {claude_count}")
    print(f"  OpenAI Codex PRs:             {codex_count}")
    print(f"  Merged PRs:                   {merged_count}")
    print(f"  Approved PRs:                 {approved_count}")
    print(f"  PRs with bug-fix commits:     {bugfix_count}")
    print(f"\n  Output file: {OUTPUT_FILE}")
    print(f"  [OK] Feature extraction complete!")


if __name__ == "__main__":
    main()
