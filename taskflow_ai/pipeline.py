#!/usr/bin/env python3
"""
End-to-end pipeline for TaskFlow AI system.
"""

import logging
import sys
from mini_lm import MiniLM
from reviewer import TaskReviewer
from generator import TaskGenerator
from rewards import RewardSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('taskflow.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline(repo_url, metadata, developer_info, last_task, last_review=None):
    """
    Run the complete TaskFlow AI pipeline.

    Args:
        repo_url (str): GitHub repository URL to review
        metadata (dict): Task metadata
        developer_info (dict): Developer information
        last_task (str): Description of last completed task
        last_review (dict, optional): Previous review

    Returns:
        dict: Pipeline results
    """
    logger.info("Starting TaskFlow AI pipeline")

    try:
        # Initialize components
        mini_lm = MiniLM()
        reviewer = TaskReviewer(mini_lm)
        generator = TaskGenerator(mini_lm)
        reward_system = RewardSystem(mini_lm)

        # Step 1: Review the task submission
        logger.info("Step 1: Reviewing task submission")
        review = reviewer.review(repo_url, metadata)
        logger.info(f"Review completed: Score {review['score']}/10")

        # Step 2: Evaluate review quality
        logger.info("Step 2: Evaluating review quality")
        review_prompt = f"Review this code submission: {repo_url}"
        reward = reward_system.evaluate_output(review_prompt, str(review), reviewer.schema)
        logger.info(f"Review reward: {reward}")

        # Step 3: Generate next task
        logger.info("Step 3: Generating next task")
        next_task = generator.generate_next_task(
            developer_info.get('id', 'dev_001'),
            developer_info.get('skill', 'intermediate'),
            last_task,
            review
        )
        logger.info(f"Next task generated: {next_task['task_description']}")

        # Step 4: Evaluate task generation quality
        logger.info("Step 4: Evaluating task generation quality")
        gen_prompt = f"Generate next task for {developer_info}"
        reward = reward_system.evaluate_output(gen_prompt, str(next_task), generator.schema)
        logger.info(f"Generation reward: {reward}")

        # Step 5: Check if retraining needed
        logger.info("Step 5: Checking retraining requirements")
        reward_system.retrain_if_needed()

        # Step 6: Get feedback summary
        summary = reward_system.get_feedback_summary()
        logger.info(f"Feedback summary: {summary}")

        result = {
            "review": review,
            "next_task": next_task,
            "feedback_summary": summary,
            "status": "success"
        }

        logger.info("Pipeline completed successfully")
        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    """Main function for command line usage."""
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <repo_url> <developer_skill> [last_task]")
        sys.exit(1)

    repo_url = sys.argv[1]
    developer_skill = sys.argv[2]
    last_task = sys.argv[3] if len(sys.argv) > 3 else "Basic coding task"

    metadata = {"task_type": "coding", "language": "python"}
    developer_info = {"id": "cli_user", "skill": developer_skill}

    result = run_pipeline(repo_url, metadata, developer_info, last_task)

    if result["status"] == "success":
        print("\n=== PIPELINE RESULTS ===")
        print(f"Review Score: {result['review']['score']}/10")
        print(f"Good Aspects: {result['review']['good_aspects']}")
        print(f"Missing Aspects: {result['review']['missing_aspects']}")
        print(f"\nNext Task: {result['next_task']['task_description']}")
        print(f"Difficulty: {result['next_task']['difficulty']}")
        print(f"Estimated Time: {result['next_task']['estimated_time']}")
        print(f"\nFeedback Summary: {result['feedback_summary']}")
    else:
        print(f"Pipeline failed: {result['error']}")

if __name__ == "__main__":
    main()