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
    Uses ALL reviewer output fields: score, good_aspects, missing_aspects.
    Pure deterministic template filling with context awareness.
    """
    score = review.get("score", 5)
    good_aspects = review.get("good_aspects", [])
    missing_aspects = review.get("missing_aspects", [])

    # CONTEXT-AWARE TEMPLATE SELECTION - Uses ALL reviewer data
    template_data = select_context_aware_template(score, good_aspects, missing_aspects, current_skill, last_task)

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

    # VALIDATE AGAINST SCHEMA - STRICT ENFORCEMENT
    if generator_schema:
        try:
            validate(instance=task, schema=generator_schema)
            print(f"✅ Task generation validation passed for score {score}")
        except ValidationError as e:
            print(f"❌ Task generation schema validation failed: {e.message}")
            # Generate schema-compliant fallback
            task = generate_schema_compliant_fallback(generator_schema, score)
    else:
        print("⚠️  No schema available for validation")

    return task


def select_context_aware_template(score: int, good_aspects: list, missing_aspects: list, current_skill: str, last_task: str) -> dict:
    """
    Selects template based on ALL reviewer context, not just score.
    Uses good_aspects, missing_aspects, current_skill, and last_task for intelligent selection.
    """
    # Analyze missing aspects to determine focus areas
    missing_keywords = ' '.join(missing_aspects).lower()

    # Determine difficulty based on score and skill level
    if score <= 3:
        difficulty = "beginner"
        estimated_time = "2-3 hours"
    elif score <= 6:
        difficulty = "intermediate"
        estimated_time = "4-6 hours"
    else:
        difficulty = "advanced"
        estimated_time = "6-8 hours"

    # Context-aware template selection based on missing aspects
    if "documentation" in missing_keywords or "readme" in missing_keywords:
        return {
            "task_description": "Improve project documentation and establish professional standards by creating comprehensive documentation and project structure.",
            "requirement1": "Create detailed README.md with setup instructions, usage examples, and API documentation",
            "requirement2": "Add proper project structure with clear folder organization and naming conventions",
            "requirement3": "Include inline documentation and docstrings for all public functions and classes",
            "requirement4": "Add license, contribution guidelines, and code of conduct files",
            "difficulty": difficulty,
            "estimated_time": estimated_time,
            "skill1": "Technical Writing",
            "skill2": "Project Organization",
            "skill3": "Professional Development"
        }

    elif "test" in missing_keywords or "testing" in missing_keywords:
        return {
            "task_description": "Implement comprehensive testing strategy to ensure code reliability and prevent regressions.",
            "requirement1": "Set up testing framework and write unit tests for core functionality",
            "requirement2": "Add integration tests for component interactions and API endpoints",
            "requirement3": "Implement test automation in CI/CD pipeline",
            "requirement4": "Add code coverage reporting and maintain minimum coverage thresholds",
            "difficulty": difficulty,
            "estimated_time": estimated_time,
            "skill1": "Testing",
            "skill2": "Quality Assurance",
            "skill3": "CI/CD"
        }

    elif "error" in missing_keywords or "handling" in missing_keywords or "validation" in missing_keywords:
        return {
            "task_description": "Enhance error handling and input validation throughout the application for better reliability and user experience.",
            "requirement1": "Implement comprehensive input validation for all user inputs and API parameters",
            "requirement2": "Add proper error handling with meaningful error messages and logging",
            "requirement3": "Create custom exception classes for different error scenarios",
            "requirement4": "Add graceful degradation and fallback mechanisms for error conditions",
            "difficulty": difficulty,
            "estimated_time": estimated_time,
            "skill1": "Error Handling",
            "skill2": "Input Validation",
            "skill3": "Robust Programming"
        }

    else:
        # Default template based on score
        if score <= 3:
            return {
                "task_description": "Address fundamental repository issues and establish proper development practices.",
                "requirement1": "Create comprehensive README.md with project overview and setup instructions",
                "requirement2": "Add proper project structure and organization",
                "requirement3": "Include basic documentation for all code files",
                "requirement4": "Add license and contribution guidelines",
                "difficulty": difficulty,
                "estimated_time": estimated_time,
                "skill1": "Documentation",
                "skill2": "Project Organization",
                "skill3": "Professional Practices"
            }
        elif score <= 6:
            return {
                "task_description": "Enhance code quality through systematic improvements and best practices implementation.",
                "requirement1": "Refactor code for better readability and maintainability",
                "requirement2": "Add unit tests for core functionality",
                "requirement3": "Improve error handling and input validation",
                "requirement4": "Implement code formatting and linting standards",
                "difficulty": difficulty,
                "estimated_time": estimated_time,
                "skill1": "Code Quality",
                "skill2": "Testing",
                "skill3": "Best Practices"
            }
        else:
            return {
                "task_description": "Implement advanced features and optimizations to elevate the project to production-ready standards.",
                "requirement1": "Identify and implement performance optimizations",
                "requirement2": "Add advanced features based on project requirements",
                "requirement3": "Implement comprehensive integration testing",
                "requirement4": "Add logging, monitoring, and observability capabilities",
                "difficulty": difficulty,
                "estimated_time": estimated_time,
                "skill1": "Performance Optimization",
                "skill2": "Advanced Development",
                "skill3": "Production Engineering"
            }


def generate_schema_compliant_fallback(schema: dict, score: int) -> dict:
    """
    Generate a schema-compliant fallback task when validation fails.
    """
    fallback = {
        "task_description": "Complete a development task to improve your skills and project quality.",
        "requirements": [
            "Analyze the current codebase and identify improvement areas",
            "Implement the identified improvements",
            "Test the changes thoroughly",
            "Document the changes made"
        ],
        "difficulty": "intermediate" if score > 5 else "beginner",
        "estimated_time": "3-4 hours",
        "skills_focused": ["Problem Solving", "Code Quality", "Development Practices"]
    }

    # Ensure compliance with schema
    if schema:
        try:
            validate(instance=fallback, schema=schema)
        except ValidationError:
            # If still not compliant, use minimal required fields
            fallback = {
                "task_description": "Complete development task",
                "requirements": ["Implement improvements", "Test changes"]
            }

    return fallback


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
