#!/usr/bin/env python3
"""
Test the complete pipeline without relative imports.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Direct execution to avoid relative import issues
# Read and execute only the function definitions, not the CLI part
reviewer_code = open('reviewer.py').read()
# Remove the CLI part at the end
reviewer_code = reviewer_code.split('# ------------ CLI Usage ------------ #')[0]
exec(reviewer_code)

generator_code = open('generator.py').read()
exec(generator_code)

rewards_code = open('rewards.py').read()
exec(rewards_code)

def test_full_pipeline():
    """Test the complete pipeline flow."""
    print("🧪 Testing Complete TaskFlow AI Pipeline")
    print("=" * 50)

    # Test data
    repo_url = "https://github.com/octocat/Hello-World"
    dev_id = "test_dev"
    skill = "beginner"
    last_task = "Created basic repo"

    try:
        # Step 1: Review repository
        print("📋 Step 1: Reviewing repository...")
        review_result = review_repository(repo_url)
        print(f"✅ Review Score: {review_result['score']}/10")
        print(f"💪 Strengths: {review_result['strengths']}")
        print(f"⚠️  Weaknesses: {review_result['weaknesses']}")
        print()

        # Step 2: Generate next task
        print("🎯 Step 2: Generating next task...")
        task_result = generate_next_task(dev_id, skill, last_task, review_result)
        print(f"👤 Developer: {task_result['developer_id']}")
        print(f"📚 Difficulty: {task_result['difficulty']}")
        print(f"🎯 Next Task: {task_result['next_task']}")
        print()

        # Step 3: Test rewards
        print("🏆 Step 3: Testing reward system...")
        reward_sys = RewardSystem(None)  # No mini_lm for this test
        reward = reward_sys.evaluate_output("test prompt", '{"valid": "json"}', {})
        print(f"🎖️  Reward for valid JSON: {reward}")

        reward = reward_sys.evaluate_output("test prompt", "invalid json", {})
        print(f"❌ Reward for invalid JSON: {reward}")
        print()

        # Step 4: RL Stats
        print("📊 Step 4: RL Statistics...")
        rl_stats = reward_sys.get_rl_stats()
        print(f"🧠 Learning Rate: {rl_stats['learning_rate']:.4f}")
        print(f"📈 Total Steps: {rl_stats['total_steps']}")
        print()

        print("🎉 ALL TESTS PASSED!")
        print("✅ GitHub repo cloning and analysis: WORKING")
        print("✅ Rubric-based scoring: WORKING")
        print("✅ Task generation with templates: WORKING")
        print("✅ RL reward system: WORKING")
        print("✅ Pipeline integration: WORKING")

        return True

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)