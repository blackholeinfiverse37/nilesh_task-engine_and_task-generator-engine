"""
TaskFlow AI - GitHub Repository Reviewer

This module:
- Clones a GitHub repository
- Analyzes its structure and files
- Checks for README and languages
- Computes a rubric-based score out of 10
- Returns a structured JSON-serializable review result
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Set, Optional

from git import Repo, GitCommandError
import json
import requests
from pathlib import Path
import ast
import re


# -------------------------------------------------------------------
# Load reviewer schema if available
# -------------------------------------------------------------------
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schemas", "reviewer_schema.json")
if os.path.exists(SCHEMA_PATH):
    with open(SCHEMA_PATH, "r") as f:
        reviewer_schema = json.load(f)
else:
    reviewer_schema = None


# ------------ Data Models ------------ #

@dataclass
class RepoAnalysis:
    repo_url: str
    local_path: str
    total_files: int
    code_files: int
    readme_exists: bool
    languages: List[str]
    file_examples: List[str]
    # Deep analysis fields
    total_lines: int
    comment_lines: int
    function_count: int
    class_count: int
    test_files: int
    complexity_score: float
    documentation_score: float
    github_stars: int
    github_forks: int
    last_commit_days: int


@dataclass
class RepoReview:
    repo_url: str
    score: int  # 0–10
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    analysis: RepoAnalysis


# ------------ Core Functions ------------ #

def clone_repo(repo_url: str, target_dir: Optional[str] = None) -> str:
    """
    Clone the GitHub repository to a local directory.
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="taskflow_repo_")

    # If folder exists, clear it
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    try:
        print(f"[TaskFlow Reviewer] Cloning repository: {repo_url}")
        Repo.clone_from(repo_url, target_dir)
        print(f"[TaskFlow Reviewer] Cloned to: {target_dir}")
        return target_dir
    except GitCommandError as e:
        raise RuntimeError(f"Failed to clone repository: {e}") from e


def analyze_repo(repo_path: str, repo_url: str) -> RepoAnalysis:
    """
    Perform DEEP analysis of the cloned repository including:
    - File structure analysis
    - Code quality metrics
    - Documentation assessment
    - GitHub API data integration
    """
    total_files = 0
    code_files = 0
    languages: Set[str] = set()
    readme_exists = False
    file_examples: List[str] = []

    # Deep analysis metrics
    total_lines = 0
    comment_lines = 0
    function_count = 0
    class_count = 0
    test_files = 0
    complexity_indicators = []

    CODE_EXTENSIONS = {
        "py": "Python",
        "js": "JavaScript",
        "ts": "TypeScript",
        "tsx": "TypeScript",
        "jsx": "JavaScript",
        "java": "Java",
        "cpp": "C++",
        "c": "C",
        "cs": "C#",
        "go": "Go",
        "rs": "Rust",
        "php": "PHP",
        "rb": "Ruby",
        "kt": "Kotlin",
        "swift": "Swift",
    }

    # First pass: collect file information
    for root, dirs, files in os.walk(repo_path):
        # Skip .git folder
        dirs[:] = [d for d in dirs if d != ".git"]

        for filename in files:
            total_files += 1
            rel_path = os.path.relpath(os.path.join(root, filename), repo_path)

            if len(file_examples) < 20:
                file_examples.append(rel_path)

            if filename.lower() in {"readme.md", "readme", "readme.txt", "readme.rst"}:
                readme_exists = True

            if "." in filename:
                ext = filename.rsplit(".", 1)[-1].lower()
                if ext in CODE_EXTENSIONS:
                    code_files += 1
                    languages.add(CODE_EXTENSIONS[ext])

                    # Check for test files
                    if any(test_indicator in filename.lower() for test_indicator in ['test', 'spec', '_test', 'tests']):
                        test_files += 1

                    # Deep analysis for supported languages
                    if ext == "py":
                        analysis = analyze_python_file(os.path.join(root, filename))
                        total_lines += analysis['lines']
                        comment_lines += analysis['comments']
                        function_count += analysis['functions']
                        class_count += analysis['classes']
                        complexity_indicators.extend(analysis['complexity'])

    # Calculate quality scores
    complexity_score = calculate_complexity_score(complexity_indicators, total_lines)
    documentation_score = calculate_documentation_score(comment_lines, total_lines, readme_exists)

    # Get GitHub API data
    github_data = get_github_repo_data(repo_url)

    return RepoAnalysis(
        repo_url=repo_url,
        local_path=repo_path,
        total_files=total_files,
        code_files=code_files,
        readme_exists=readme_exists,
        languages=sorted(languages),
        file_examples=file_examples,
        total_lines=total_lines,
        comment_lines=comment_lines,
        function_count=function_count,
        class_count=class_count,
        test_files=test_files,
        complexity_score=complexity_score,
        documentation_score=documentation_score,
        github_stars=github_data['stars'],
        github_forks=github_data['forks'],
        last_commit_days=github_data['days_since_commit']
    )


def analyze_python_file(file_path: str) -> Dict[str, Any]:
    """
    Perform deep analysis of a Python file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        lines = content.split('\n')
        total_lines = len(lines)

        # Count comments and blank lines
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                comment_lines += 1
            elif '"""' in line or "'''" in line:
                # Multi-line docstring detection (simplified)
                comment_lines += 1

        # Parse AST for code structure
        functions = 0
        classes = 0
        complexity_indicators = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions += 1
                    # Simple complexity: count of statements in function
                    complexity_indicators.append(len(node.body))
                elif isinstance(node, ast.ClassDef):
                    classes += 1

        except SyntaxError:
            # If file has syntax errors, still count basic metrics
            pass

        return {
            'lines': total_lines,
            'comments': comment_lines,
            'functions': functions,
            'classes': classes,
            'complexity': complexity_indicators
        }

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return {'lines': 0, 'comments': 0, 'functions': 0, 'classes': 0, 'complexity': []}


def calculate_complexity_score(complexity_indicators: List[int], total_lines: int) -> float:
    """Calculate code complexity score (0-10 scale)."""
    if not complexity_indicators or total_lines == 0:
        return 5.0  # Neutral score

    avg_complexity = sum(complexity_indicators) / len(complexity_indicators)
    # Normalize to 0-10 scale (lower is better)
    score = max(0, min(10, 10 - (avg_complexity / 20)))
    return round(score, 1)


def calculate_documentation_score(comment_lines: int, total_lines: int, has_readme: bool) -> float:
    """Calculate documentation quality score (0-10 scale)."""
    if total_lines == 0:
        return 0.0

    comment_ratio = comment_lines / total_lines
    base_score = min(10, comment_ratio * 50)  # 20% comment ratio = 10 points

    if has_readme:
        base_score += 2  # Bonus for README

    return min(10, round(base_score, 1))


def get_github_repo_data(repo_url: str) -> Dict[str, Any]:
    """
    Fetch repository data from GitHub API.
    Note: Requires GitHub token for full functionality.
    """
    if not repo_url:
        return {'stars': 0, 'forks': 0, 'days_since_commit': 0}

    try:
        # Extract owner/repo from URL
        parts = repo_url.rstrip('/').split('/')
        if len(parts) >= 2:
            owner, repo = parts[-2], parts[-1].replace('.git', '')
            api_url = f"https://api.github.com/repos/{owner}/{repo}"

            # Check for GitHub token in environment
            github_token = os.getenv('GITHUB_TOKEN')
            headers = {'Authorization': f'token {github_token}'} if github_token else {}

            response = requests.get(api_url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return {
                    'stars': data.get('stargazers_count', 0),
                    'forks': data.get('forks_count', 0),
                    'days_since_commit': calculate_days_since_commit(data.get('updated_at', ''))
                }
            elif response.status_code == 404:
                print(f"[GitHub API] Repository not found: {owner}/{repo}")
            elif response.status_code == 401:
                print("[GitHub API] Authentication failed - check GITHUB_TOKEN")
            else:
                print(f"[GitHub API] Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[GitHub API] Failed to fetch data: {e}")

    return {'stars': 0, 'forks': 0, 'days_since_commit': 0}


def calculate_days_since_commit(updated_at: str) -> int:
    """
    Calculate days since last commit from ISO date string.
    """
    from datetime import datetime
    try:
        # Parse ISO format date
        commit_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        now = datetime.now(commit_date.tzinfo)
        return (now - commit_date).days
    except Exception:
        return 0


def score_repo(analysis: RepoAnalysis):
    """
    Apply comprehensive rubric scoring using deep analysis metrics.
    """
    rubric_path = os.path.join(os.path.dirname(__file__), 'config', 'rubric.json')
    with open(rubric_path) as f:
        rubric = json.load(f)

    score = 0
    strengths: List[str] = []
    weaknesses: List[str] = []

    categories = rubric['categories']

    # README assessment
    if analysis.readme_exists:
        score += categories['readme_exists']['weight']
        strengths.append(categories['readme_exists']['description'])
    else:
        weaknesses.append("Repository is missing a README file")

    # Programming languages
    if analysis.languages:
        score += categories['programming_languages']['weight']
        strengths.append(f"Repository contains code in: {', '.join(analysis.languages)}")
    else:
        weaknesses.append("No programming language files detected")

    # File count
    if analysis.total_files > 5:
        score += categories['file_count']['weight']
        strengths.append("Repository has substantial content (more than 5 files)")
    else:
        weaknesses.append("Repository has minimal content")

    # NEW: Code quality assessment
    if analysis.documentation_score >= 7:
        score += 2
        strengths.append("Excellent documentation quality")
    elif analysis.documentation_score >= 4:
        score += 1
        strengths.append("Good documentation quality")
    else:
        weaknesses.append("Poor documentation quality")

    # NEW: Complexity assessment
    if analysis.complexity_score >= 7:
        score += 1
        strengths.append("Well-structured, maintainable code")
    elif analysis.complexity_score <= 3:
        weaknesses.append("Code may be overly complex or poorly structured")

    # NEW: Testing assessment
    if analysis.test_files > 0:
        score += 2
        strengths.append(f"Repository includes {analysis.test_files} test files")
    else:
        weaknesses.append("No test files detected")

    # NEW: Code metrics assessment
    if analysis.function_count > 0 and analysis.total_lines > 0:
        functions_per_line = analysis.function_count / analysis.total_lines
        if functions_per_line > 0.01:  # Good function density
            score += 1
            strengths.append("Good code organization with proper function structure")

    # NEW: GitHub metrics (when available)
    if analysis.github_stars > 10:
        score += 1
        strengths.append("Popular repository with community interest")
    if analysis.last_commit_days < 30:
        score += 1
        strengths.append("Recently active development")

    score = max(0, min(score, rubric['max_score']))

    return score, strengths, weaknesses


def review_repository(repo_url: str, metadata: dict = None) -> Dict[str, Any]:
    """
    High-level function:
    - Clone repo
    - Analyze structure
    - Score with rubric
    - Return JSON dict
    """
    repo_path = clone_repo(repo_url)

    try:
        analysis = analyze_repo(repo_path, repo_url)
        analysis.repo_url = repo_url

        score, strengths, weaknesses = score_repo(analysis)

        summary = (
            f"Repository has {analysis.total_files} total files, "
            f"{analysis.code_files} code files, "
            f"{'a' if analysis.readme_exists else 'no'} README, "
            f"and uses {len(analysis.languages)} language(s). "
            f"Overall score: {score}/10."
        )

        review = RepoReview(
            repo_url=repo_url,
            score=score,
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            analysis=analysis,
        )

        result = {
            "repo_url": review.repo_url,
            "score": review.score,
            "summary": review.summary,
            "strengths": review.strengths,
            "weaknesses": review.weaknesses,
            "analysis": asdict(review.analysis),
        }

        if metadata:
            result["metadata"] = metadata

        return result

    finally:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)


# -------------------------------------------------------------------
# ✅ CLASS WRAPPER FOR DEMO COMPATIBILITY
# -------------------------------------------------------------------

class TaskReviewer:
    """
    Wrapper class required by demo.py.
    Makes the reviewer behave like an object with .schema and .review().
    """

    def __init__(self, mini_lm=None):
        self.mini_lm = mini_lm
        self.schema = reviewer_schema  # used by RewardSystem

    def review(self, repo_url: str, metadata: dict = None):
        return review_repository(repo_url, metadata)
