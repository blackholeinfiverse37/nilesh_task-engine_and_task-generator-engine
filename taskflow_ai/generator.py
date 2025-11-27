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

# STRICT RULE-BOUND TEMPLATE - MATCHES generator_schema.json
TASK_TEMPLATE = {
    "task_description": "{task_description}",
    "requirements": [
        "{requirement1}",
        "{requirement2}",
        "{requirement3}",
        "{requirement4}"
    ],
    "difficulty": "{difficulty}",
    "estimated_time": "{estimated_time}",
    "skills_focused": [
        "{skill1}",
        "{skill2}",
        "{skill3}"
    ]
}


def generate_next_task(developer_id: str, current_skill: str, last_task: str, review: dict) -> dict:
    """
    Generates the next task using STRICT RULE-BOUND TEMPLATE - NO LLM.
    Pure deterministic template filling based on review score.
    """
    score = review.get("score", 5)

    # RULE-BASED TEMPLATE SELECTION - STRICT DETERMINISTIC
    if score <= 3:
        # POOR SCORE - FOCUS ON BASICS
        template_data = {
            "task_description": "Address fundamental repository issues and establish proper structure by creating comprehensive documentation and organizing the project layout.",
            "requirement1": "Create comprehensive README.md with project overview, setup instructions, and usage examples",
            "requirement2": "Add proper project structure and organization with clear folder hierarchy",
            "requirement3": "Include basic documentation for all code files with comments and docstrings",
            "requirement4": "Add license file and contribution guidelines to make the project professional",
            "difficulty": "beginner",
            "estimated_time": "2-3 hours",
            "skill1": "Documentation",
            "skill2": "Project Organization",
            "skill3": "Professional Development Practices"
        }
    elif score <= 6:
        # FAIR SCORE - FOCUS ON QUALITY
        template_data = {
            "task_description": "Enhance code quality through systematic refactoring, testing implementation, and improved error handling to make the codebase more maintainable and reliable.",
            "requirement1": "Analyze existing code for improvement areas and potential bugs",
            "requirement2": "Refactor code for better readability, maintainability, and following best practices",
            "requirement3": "Add comprehensive unit tests for core functionality with good coverage",
            "requirement4": "Improve error handling and input validation throughout the application",
            "difficulty": "intermediate",
            "estimated_time": "4-6 hours",
            "skill1": "Code Quality",
            "skill2": "Testing",
            "skill3": "Error Handling"
        }
    else:
        # GOOD SCORE - FOCUS ON ADVANCED FEATURES
        template_data = {
            "task_description": "Implement advanced functionality and performance optimizations to take the project to the next level with production-ready features and monitoring capabilities.",
            "requirement1": "Identify performance bottlenecks and implement optimization strategies",
            "requirement2": "Implement advanced features based on project requirements and user needs",
            "requirement3": "Add comprehensive integration tests and end-to-end testing",
            "requirement4": "Implement logging, monitoring, and observability capabilities for production use",
            "difficulty": "advanced",
            "estimated_time": "6-8 hours",
            "skill1": "Performance Optimization",
            "skill2": "Advanced Development",
            "skill3": "Production Engineering"
        }

    # STRICT TEMPLATE FILLING - MATCHES generator_schema.json
    task = TASK_TEMPLATE.copy()
    task["task_description"] = task["task_description"].format(task_description=template_data["task_description"])
    task["difficulty"] = task["difficulty"].format(difficulty=template_data["difficulty"])
    task["estimated_time"] = task["estimated_time"].format(estimated_time=template_data["estimated_time"])

    # Fill requirements array
    task["requirements"] = [
        task["requirements"][0].format(requirement1=template_data["requirement1"]),
        task["requirements"][1].format(requirement2=template_data["requirement2"]),
        task["requirements"][2].format(requirement3=template_data["requirement3"]),
        task["requirements"][3].format(requirement4=template_data["requirement4"])
    ]

    # Fill skills_focused array
    task["skills_focused"] = [
        task["skills_focused"][0].format(skill1=template_data["skill1"]),
        task["skills_focused"][1].format(skill2=template_data["skill2"]),
        task["skills_focused"][2].format(skill3=template_data["skill3"])
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
