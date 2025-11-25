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
import json as json_lib
import json


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


def analyze_repo(repo_path: str) -> RepoAnalysis:
    """
    Walk through the cloned repo and collect basic statistics.
    """
    total_files = 0
    code_files = 0
    languages: Set[str] = set()
    readme_exists = False
    file_examples: List[str] = []

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

    for root, dirs, files in os.walk(repo_path):
        # Skip .git folder
        dirs[:] = [d for d in dirs if d != ".git"]

        for filename in files:
            total_files += 1
            rel_path = os.path.relpath(os.path.join(root, filename), repo_path)

            if len(file_examples) < 20:
                file_examples.append(rel_path)

            if filename.lower() in {"readme.md", "readme", "readme.txt"}:
                readme_exists = True

            if "." in filename:
                ext = filename.rsplit(".", 1)[-1].lower()
                if ext in CODE_EXTENSIONS:
                    code_files += 1
                    languages.add(CODE_EXTENSIONS[ext])

    return RepoAnalysis(
        repo_url="",
        local_path=repo_path,
        total_files=total_files,
        code_files=code_files,
        readme_exists=readme_exists,
        languages=sorted(languages),
        file_examples=file_examples,
    )


def score_repo(analysis: RepoAnalysis):
    """
    Apply the fixed rubric from rubric.json to score the repository out of 10.
    """
    rubric_path = os.path.join(os.path.dirname(__file__), 'config', 'rubric.json')
    with open(rubric_path) as f:
        rubric = json_lib.load(f)

    score = 0
    strengths: List[str] = []
    weaknesses: List[str] = []

    categories = rubric['categories']

    if analysis.readme_exists:
        score += categories['readme_exists']['weight']
        strengths.append(categories['readme_exists']['description'])
    else:
        weaknesses.append("Repository is missing a README file")

    if analysis.languages:
        score += categories['programming_languages']['weight']
        strengths.append(f"Repository contains code in: {', '.join(analysis.languages)}")
    else:
        weaknesses.append("No programming language files detected")

    if analysis.total_files > 5:
        score += categories['file_count']['weight']
        strengths.append("Repository has substantial content (more than 5 files)")
    else:
        weaknesses.append("Repository has minimal content")

    score = max(0, min(score, rubric['max_score']))

    return score, strengths, weaknesses


def review_repository(repo_url: str) -> Dict[str, Any]:
    """
    High-level function:
    - Clone repo
    - Analyze structure
    - Score with rubric
    - Return JSON dict
    """
    repo_path = clone_repo(repo_url)

    try:
        analysis = analyze_repo(repo_path)
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

        return {
            "repo_url": review.repo_url,
            "score": review.score,
            "summary": review.summary,
            "strengths": review.strengths,
            "weaknesses": review.weaknesses,
            "analysis": asdict(review.analysis),
        }

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
        return review_repository(repo_url)
