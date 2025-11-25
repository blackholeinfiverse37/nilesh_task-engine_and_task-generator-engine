#!/usr/bin/env python3
"""
Demo script showing real GitHub repo analysis functionality.
"""

import os
import sys
from git import Repo

def clone_repo(repo_url, target_dir="repo_temp"):
    if os.path.exists(target_dir):
        os.system(f"rm -rf {target_dir}")
    Repo.clone_from(repo_url, target_dir)
    return target_dir

def analyze_repo(path):
    result = {
        "files": [],
        "languages": set(),
        "readme_exists": False
    }

    for root, dirs, files in os.walk(path):
        for f in files:
            ext = f.split(".")[-1]
            result["files"].append(f)

            if ext in ["py", "js", "ts", "cpp", "java", "go"]:
                result["languages"].add(ext)

            if f.lower() == "readme.md":
                result["readme_exists"] = True

    result["languages"] = list(result["languages"])
    return result

def score_repo(analysis):
    score = 0

    if analysis["readme_exists"]:
        score += 3
    if len(analysis["languages"]) > 0:
        score += 3
    if len(analysis["files"]) > 5:
        score += 4

    return min(score, 10)

def review(repo_url):
    print(f"🔄 Cloning repository: {repo_url}")
    path = clone_repo(repo_url)
    print(f"✅ Cloned to: {path}")

    print("🔍 Analyzing repository structure...")
    analysis = analyze_repo(path)
    print(f"📊 Found {len(analysis['files'])} files")
    print(f"💻 Languages detected: {analysis['languages']}")
    print(f"📖 README exists: {analysis['readme_exists']}")

    print("📈 Scoring repository...")
    score = score_repo(analysis)
    print(f"🎯 Score: {score}/10")

    result = {
        "analysis": analysis,
        "score": score,
        "summary": f"Repo contains {len(analysis['files'])} files and uses {analysis['languages']}."
    }

    print("🧹 Cleaning up...")
    os.system(f"rm -rf {path}")

    return result

def generate_next_task(developer_id, current_skill, last_task, review):
    return {
        "developer_id": developer_id,
        "difficulty": "beginner" if review["score"] < 5 else "intermediate",
        "next_task": "Improve the repo by adding a README and refactoring code for clarity.",
        "based_on_score": review["score"]
    }

def reward_output(is_valid):
    return 1 if is_valid else -1

def main():
    print("🚀 TaskFlow AI - Real GitHub Repo Analysis Demo")
    print("=" * 50)

    # Test with octocat/Hello-World (small, public repo)
    repo_url = "https://github.com/octocat/Hello-World"

    try:
        # Step 1: Review the repo
        print("\n📋 STEP 1: Reviewing GitHub Repository")
        review_result = review(repo_url)

        # Step 2: Generate next task
        print("\n🎯 STEP 2: Generating Next Task")
        task_result = generate_next_task("demo_dev", "beginner", "Created basic repo", review_result)
        print(f"👤 Developer: {task_result['developer_id']}")
        print(f"📚 Difficulty: {task_result['difficulty']}")
        print(f"✅ Next Task: {task_result['next_task']}")

        # Step 3: Test rewards
        print("\n🏆 STEP 3: Testing Reward System")
        valid_reward = reward_output(True)
        invalid_reward = reward_output(False)
        print(f"✅ Valid output reward: {valid_reward}")
        print(f"❌ Invalid output reward: {invalid_reward}")

        print("\n🎉 SUCCESS: All GitHub repo analysis functions are working!")
        print("The system can clone repos, analyze code structure, detect languages,")
        print("check for README files, generate rubric scores, and create next tasks.")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()