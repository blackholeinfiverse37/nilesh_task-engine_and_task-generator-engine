import json
import os
import uuid
from jsonschema import validate, ValidationError

# Load generator schema if it exists
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schemas", "generator_schema.json")
if os.path.exists(SCHEMA_PATH):
    with open(SCHEMA_PATH, "r") as f:
        generator_schema = json.load(f)
else:
    generator_schema = None

# STRICT RULE-BOUND TEMPLATE - NO LLM INVOLVEMENT
TASK_TEMPLATE = {
    "task_id": "task_{task_id}",
    "title": "{title}",
    "description": "{description}",
    "steps": [
        "Step 1: {step1}",
        "Step 2: {step2}",
        "Step 3: {step3}",
        "Step 4: {step4}"
    ],
    "acceptance_criteria": [
        "Criteria 1: {criteria1}",
        "Criteria 2: {criteria2}",
        "Criteria 3: {criteria3}",
        "Criteria 4: {criteria4}"
    ],
    "difficulty": "{difficulty}",
    "estimated_time": "{estimated_time}"
}


def generate_next_task(developer_id: str, current_skill: str, last_task: str, review: dict) -> dict:
    """
    Generates the next task using STRICT RULE-BOUND TEMPLATE - NO LLM.
    Pure deterministic template filling based on review score.
    """
    score = review.get("score", 5)

    # RULE-BASED TEMPLATE SELECTION - STRICT AND DETERMINISTIC
    if score <= 3:
        template_data = {
            "task_id": uuid.uuid4().hex[:8],
            "title": "Fix Repository Basics",
            "description": "Address fundamental repository issues and establish proper structure",
            "step1": "Create comprehensive README.md with project overview",
            "step2": "Add proper project structure and organization",
            "step3": "Include basic documentation for all files",
            "step4": "Add license and contribution guidelines",
            "criteria1": "README.md exists with clear project description",
            "criteria2": "Project has logical folder structure",
            "criteria3": "All code files have basic comments",
            "criteria4": "License file is present",
            "difficulty": "beginner",
            "estimated_time": "2-3 hours"
        }

    elif score <= 6:
        template_data = {
            "task_id": uuid.uuid4().hex[:8],
            "title": "Improve Code Quality",
            "description": "Enhance code quality through refactoring and testing",
            "step1": "Analyze existing code for improvement areas",
            "step2": "Refactor code for better readability and maintainability",
            "step3": "Add unit tests for core functionality",
            "step4": "Improve error handling and input validation",
            "criteria1": "Code follows consistent style and naming conventions",
            "criteria2": "Unit tests exist for main functions",
            "criteria3": "Error handling is comprehensive",
            "criteria4": "Code is more modular and reusable",
            "difficulty": "intermediate",
            "estimated_time": "4-6 hours"
        }

    else:
        template_data = {
            "task_id": uuid.uuid4().hex[:8],
            "title": "Add Advanced Features",
            "description": "Implement advanced functionality and performance optimizations",
            "step1": "Identify performance bottlenecks and optimization opportunities",
            "step2": "Implement advanced features based on project requirements",
            "step3": "Add comprehensive integration tests",
            "step4": "Implement logging and monitoring capabilities",
            "criteria1": "Performance is measurably improved",
            "criteria2": "Advanced features are fully functional",
            "criteria3": "Test coverage exceeds 80%",
            "criteria4": "Application is production-ready with monitoring",
            "difficulty": "advanced",
            "estimated_time": "6-8 hours"
        }

    # STRICT TEMPLATE FILLING
    task = TASK_TEMPLATE.copy()
    task["task_id"] = task["task_id"].format(task_id=template_data["task_id"])
    task["title"] = task["title"].format(title=template_data["title"])
    task["description"] = task["description"].format(description=template_data["description"])
    task["difficulty"] = task["difficulty"].format(difficulty=template_data["difficulty"])
    task["estimated_time"] = task["estimated_time"].format(estimated_time=template_data["estimated_time"])

    task["steps"] = [
        f"Step 1: {template_data['step1']}",
        f"Step 2: {template_data['step2']}",
        f"Step 3: {template_data['step3']}",
        f"Step 4: {template_data['step4']}",
    ]

    task["acceptance_criteria"] = [
        f"Criteria 1: {template_data['criteria1']}",
        f"Criteria 2: {template_data['criteria2']}",
        f"Criteria 3: {template_data['criteria3']}",
        f"Criteria 4: {template_data['criteria4']}",
    ]

    return task


# -------------------------------------------------------------------
# ✅ CLASS WRAPPER FOR DEMO COMPATIBILITY
# -------------------------------------------------------------------
class TaskGenerator:
    """
    Wrapper class required by demo.py.
    This class simply calls the deterministic template function.
    """

    def __init__(self, mini_lm=None):
        self.mini_lm = mini_lm
        self.schema = generator_schema  # Used by RewardSystem in demo

    def generate_next_task(self, developer_id, current_skill, last_task, review):
        return generate_next_task(developer_id, current_skill, last_task, review)
