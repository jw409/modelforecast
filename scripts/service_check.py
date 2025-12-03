#!/usr/bin/env python3
"""
8-hour service check for ModelForecast project engagement.

Run this every 8 hours to track:
- GitHub stars/forks/watchers
- Issues/PRs activity
- Repo traffic (if API key has push access)

Usage:
    uv run python scripts/service_check.py
    uv run python scripts/service_check.py --json  # Machine-readable output
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path

REPO = "jw409/modelforecast"


def run_gh(args: list[str]) -> dict | list | str:
    """Run gh CLI command and return JSON result."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except subprocess.CalledProcessError as e:
        return {"error": e.stderr}
    except json.JSONDecodeError:
        return result.stdout.strip()


def get_repo_stats() -> dict:
    """Get basic repo stats."""
    result = run_gh([
        "repo", "view", REPO, "--json",
        "stargazerCount,forkCount,watchers,description,homepageUrl"
    ])
    # Normalize watchers to watcherCount for consistency
    if isinstance(result, dict) and "watchers" in result:
        result["watcherCount"] = result["watchers"].get("totalCount", 0)
    return result


def get_issues_prs() -> dict:
    """Get open issues and PRs."""
    issues = run_gh(["issue", "list", "--repo", REPO, "--json", "number,title,createdAt", "--limit", "5"])
    prs = run_gh(["pr", "list", "--repo", REPO, "--json", "number,title,createdAt", "--limit", "5"])
    return {"issues": issues, "prs": prs}


def get_recent_activity() -> dict:
    """Get recent commits and releases."""
    # Recent commits
    commits = run_gh([
        "api", f"repos/{REPO}/commits",
        "--jq", ".[0:5] | map({sha: .sha[0:7], message: .commit.message | split(\"\\n\")[0], date: .commit.committer.date})"
    ])

    # Latest release (if any)
    releases = run_gh([
        "api", f"repos/{REPO}/releases",
        "--jq", ".[0:1] | map({tag: .tag_name, name: .name, date: .published_at})"
    ])

    return {"commits": commits, "releases": releases}


def get_traffic() -> dict:
    """Get traffic stats (requires push access)."""
    try:
        views = run_gh(["api", f"repos/{REPO}/traffic/views", "--jq", "{count: .count, uniques: .uniques}"])
        clones = run_gh(["api", f"repos/{REPO}/traffic/clones", "--jq", "{count: .count, uniques: .uniques}"])
        return {"views": views, "clones": clones}
    except Exception:
        return {"note": "Traffic requires push access to repo"}


def load_history() -> list:
    """Load engagement history."""
    history_file = Path(__file__).parent.parent / "var" / "engagement_history.jsonl"
    if not history_file.exists():
        return []
    return [json.loads(line) for line in history_file.read_text().strip().split("\n") if line]


def save_snapshot(snapshot: dict):
    """Save current snapshot to history."""
    history_file = Path(__file__).parent.parent / "var" / "engagement_history.jsonl"
    history_file.parent.mkdir(exist_ok=True)
    with history_file.open("a") as f:
        f.write(json.dumps(snapshot) + "\n")


def calculate_delta(current: dict, history: list) -> dict:
    """Calculate change since last check."""
    if not history:
        return {"note": "First check - no delta available"}

    last = history[-1]
    last_stats = last.get("repo", {})
    curr_stats = current.get("repo", {})

    return {
        "stars": curr_stats.get("stargazerCount", 0) - last_stats.get("stargazerCount", 0),
        "forks": curr_stats.get("forkCount", 0) - last_stats.get("forkCount", 0),
        "since": last.get("timestamp", "unknown"),
    }


def format_report(data: dict) -> str:
    """Format human-readable report."""
    repo = data.get("repo", {})
    traffic = data.get("traffic", {})
    delta = data.get("delta", {})

    lines = [
        "=" * 50,
        f"ModelForecast Service Check - {data['timestamp']}",
        "=" * 50,
        "",
        "## Engagement",
        f"  Stars:    {repo.get('stargazerCount', '?')} ({delta.get('stars', '?'):+d} since last check)" if isinstance(delta.get('stars'), int) else f"  Stars:    {repo.get('stargazerCount', '?')}",
        f"  Forks:    {repo.get('forkCount', '?')} ({delta.get('forks', '?'):+d})" if isinstance(delta.get('forks'), int) else f"  Forks:    {repo.get('forkCount', '?')}",
        f"  Watchers: {repo.get('watcherCount', '?')}",
        "",
    ]

    if isinstance(traffic.get("views"), dict):
        lines.extend([
            "## Traffic (14 days)",
            f"  Views:  {traffic['views'].get('count', '?')} total, {traffic['views'].get('uniques', '?')} unique",
            f"  Clones: {traffic['clones'].get('count', '?')} total, {traffic['clones'].get('uniques', '?')} unique",
            "",
        ])

    issues = data.get("activity", {}).get("issues", [])
    prs = data.get("activity", {}).get("prs", [])

    lines.append(f"## Issues: {len(issues)} open")
    for issue in issues[:3]:
        if isinstance(issue, dict):
            lines.append(f"  #{issue.get('number', '?')}: {issue.get('title', '?')[:40]}")

    lines.append(f"\n## PRs: {len(prs)} open")
    for pr in prs[:3]:
        if isinstance(pr, dict):
            lines.append(f"  #{pr.get('number', '?')}: {pr.get('title', '?')[:40]}")

    commits = data.get("commits", {}).get("commits", [])
    if commits:
        lines.append("\n## Recent Commits")
        for c in commits[:3]:
            if isinstance(c, dict):
                lines.append(f"  {c.get('sha', '?')}: {c.get('message', '?')[:40]}")

    lines.extend(["", "=" * 50])
    return "\n".join(lines)


def main():
    import sys

    json_output = "--json" in sys.argv

    # Gather data
    timestamp = datetime.now().isoformat()
    repo = get_repo_stats()
    activity = get_issues_prs()
    commits = get_recent_activity()
    traffic = get_traffic()

    history = load_history()

    snapshot = {
        "timestamp": timestamp,
        "repo": repo,
        "activity": activity,
        "commits": commits,
        "traffic": traffic,
    }

    delta = calculate_delta(snapshot, history)
    snapshot["delta"] = delta

    # Save to history
    save_snapshot(snapshot)

    if json_output:
        print(json.dumps(snapshot, indent=2))
    else:
        print(format_report(snapshot))
        print(f"\nHistory saved to var/engagement_history.jsonl ({len(history) + 1} entries)")


if __name__ == "__main__":
    main()
