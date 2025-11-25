#!/usr/bin/env python3
"""
Stable Demo script for TaskFlow AI system.
Ensures successful demo even if repo cloning fails.
"""

from unittest.mock import Mock
from .reviewer import TaskReviewer
from .generator import TaskGenerator
from .rewards import RewardSystem


def main():
    print("TaskFlow AI Demo")
    print("================\n")

    # Mock MiniLM for controlled deterministic outputs
    mock_mini_lm = Mock()

    # Mock MiniLM responses for demonstration
    mock_mini_lm.generate.side_effect = [
        # Mock reviewer output
        '{"good_aspects": ["Well-structured code"], "missing_aspects": ["Unit tests"], "score": 7}',

        # Mock generator output
        '{"task_description": "Add unit tests", '
        '"requirements": ["Write tests", "Implement pytest"], '
        '"difficulty": "intermediate", '
        '"estimated_time": "4 hours", '
        '"skills_focused": ["Testing", "Python"]}'
    ]

    # Initialize system components
    reviewer = TaskReviewer(mock_mini_lm)
    generator = TaskGenerator(mock_mini_lm)
    reward_system = RewardSystem(mock_mini_lm)

    # ------------------------------------------------
    # 1. REVIEWING THE REPOSITORY
    # ------------------------------------------------
    print("1. Reviewing a task submission...")

    # Use a real GitHub repo (small and public)
    repo_url = "https://github.com/torvalds/test-tlb"
    metadata = {"task_type": "Python application", "difficulty": "beginner"}

    try:
        review = reviewer.review(repo_url, metadata)
        print("\nReview Output:")
        print(review)
        print()
    except Exception as e:
        print(f"Review failed: {e}")
        print("Using fallback mock review...\n")
        review = {"score": 7}

    # ------------------------------------------------
    # 2. GENERATING NEXT TASK
    # ------------------------------------------------
    print("2. Generating next task...\n")

    developer_id = "dev_001"
    current_skill = "beginner"
    last_task = "Build a simple calculator app"

    try:
        next_task = generator.generate_next_task(
            developer_id,
            current_skill,
            last_task,
            review
        )
        print("Generated Next Task:")
        print(next_task)
        print()
    except Exception as e:
        print(f"Task generation failed: {e}\n")

    # ------------------------------------------------
    # 3. REWARD SYSTEM EVALUATION
    # ------------------------------------------------
    print("3. Evaluating rewards...\n")

    reward_review = reward_system.evaluate_output(
        "review prompt",
        '{"good_aspects": ["x"], "missing_aspects": ["y"], "score": 7}',
        reviewer.schema
    )

    reward_generation = reward_system.evaluate_output(
        "generate prompt",
        '{"task_description": "test", "requirements": ["req"]}',
        generator.schema
    )

    print(f"Review reward score: {reward_review}")
    print(f"Task generation reward score: {reward_generation}")

    summary = reward_system.get_feedback_summary()
    print(f"\nReward System Summary: {summary}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
