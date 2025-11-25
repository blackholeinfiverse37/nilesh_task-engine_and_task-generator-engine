#!/usr/bin/env python3
"""
Test script to verify real GitHub repo analysis functionality.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

# Direct execution to avoid relative import issues
exec(open('reviewer.py').read())
exec(open('generator.py').read())
exec(open('rewards.py').read())

def test_clone_and_analyze():
    """Test the actual GitHub repo cloning and analysis."""
    print("Testing GitHub repo cloning and analysis...")

    try:
        # Test with a small public repo
        repo_url = "https://github.com/octocat/Hello-World"

        print(f"Cloning {repo_url}...")
        path = clone_repo(repo_url)
        print(f"Cloned to: {path}")

        print("Analyzing repository...")
        analysis = analyze_repo(path)
        print(f"Analysis: {analysis}")

        print("Scoring repository...")
        score = score_repo(analysis)
        print(f"Score: {score}/10")

        print("Full review...")
        result = review(repo_url)
        print(f"Review result: {result}")

        # Test task generation
        print("Testing task generation...")
        task = generate_next_task("dev123", "intermediate", "Built calculator", result)
        print(f"Generated task: {task}")

        # Test rewards
        print("Testing rewards...")
        reward = reward_output(True)
        print(f"Reward for valid: {reward}")
        reward = reward_output(False)
        print(f"Reward for invalid: {reward}")

        print("✅ All tests passed! GitHub repo analysis is working.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clone_and_analyze()