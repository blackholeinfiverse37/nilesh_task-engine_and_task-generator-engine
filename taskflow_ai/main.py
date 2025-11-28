import os
import json

from .reviewer import review_repository
from .generator import generate_next_task
from .mini_lm import MiniLM
from .rewards import RewardSystem


def run_pipeline():
    print("\n=== TaskFlow AI Pipeline ===\n")

    repo_url = input("Enter GitHub repository URL: ").strip()
    developer_id = input("Enter Developer ID: ").strip()
    skill = input("Enter developer skill level (beginner/intermediate/advanced): ").strip()

    print("\nReviewing repository...\n")
    review_output = review_repository(repo_url)

    print("Repository Review Result:")
    print(json.dumps(review_output, indent=4))

    # Last task placeholder
    last_task = "No previous task"  # You can change this if you store task history

    # --- Correct Schema-Compatible Prompt ---
    task_prompt = f"""
    You are TaskFlow AI. Generate the next development task for developer '{developer_id}' (skill: '{skill}').

    Last completed task: {last_task}
    Repository review score: {review_output['score']}/10
    Weaknesses identified: {', '.join(review_output.get('weaknesses', []))}

    Generate ONLY valid JSON following this EXACT schema:

    {{
      "task_description": "Explain what the developer must do next.",
      "requirements": [
        "Clear requirement 1",
        "Clear requirement 2",
        "Clear requirement 3",
        "Clear requirement 4"
      ],
      "difficulty": "beginner | intermediate | advanced",
      "estimated_time": "X-Y hours",
      "skills_focused": [
        "Skill 1",
        "Skill 2",
        "Skill 3"
      ]
    }}

    Output ONLY JSON. No text outside the JSON.
    """

    # === MiniLM Generation ===
    print("\nGenerating task with MiniLM...\n")
    mini_lm = MiniLM()
    generated_task_json = mini_lm.generate(
        prompt=task_prompt,
        schema_path=os.path.join(os.path.dirname(__file__), "schemas", "generator_schema.json")
    )

    print("Generated Task (MiniLM):")
    print(json.dumps(generated_task_json, indent=4))

    # === Rule-Based Task Generator ===
    print("\nGenerating rule-based next task...\n")
    rule_task = generate_next_task(
        developer_id,
        skill,
        last_task,
        review_output
    )

    print("Rule-Based Task Output:")
    print(json.dumps(rule_task, indent=4))

    # === RL Feedback Loop ===
    print("\nRunning RL feedback training...\n")
    reward_system = RewardSystem()

    reward = reward_system.evaluate_output(
        prompt=task_prompt,
        output=generated_task_json,
        schema_path=os.path.join(os.path.dirname(__file__), "schemas", "generator_schema.json")
    )

    print(f"Reward from validation: {reward}")

    # Training Loop
    reward_system.train_model(
        model=mini_lm,
        prompt=task_prompt,
        expected_schema_path=os.path.join(os.path.dirname(__file__), "schemas", "generator_schema.json"),
        num_iterations=3
    )

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    run_pipeline()
