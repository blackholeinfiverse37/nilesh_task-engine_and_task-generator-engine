import unittest
from unittest.mock import Mock
import json
from taskflow_ai.generator import TaskGenerator

class TestTaskGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_mini_lm = Mock()
        self.generator = TaskGenerator(self.mock_mini_lm)

    def test_generate_next_task_valid(self):
        valid_output = json.dumps({
            "task_description": "Implement a REST API endpoint",
            "requirements": ["Use Flask", "Handle GET/POST", "Add validation"],
            "difficulty": "intermediate",
            "estimated_time": "3 hours",
            "skills_focused": ["API development", "Python"]
        })
        self.mock_mini_lm.generate.return_value = valid_output

        review = {"good_aspects": ["good"], "missing_aspects": ["tests"], "score": 8}
        result = self.generator.generate_next_task("dev123", "intermediate", "Build a calculator", review)

        self.assertEqual(result['difficulty'], "intermediate")
        self.assertIn("REST API", result['task_description'])

    def test_generate_next_task_invalid_corrected(self):
        invalid_output = 'Task: {"task_description": "Debug code", "requirements": ["Find bugs"]}'
        self.mock_mini_lm.generate.return_value = invalid_output

        review = {"score": 6}
        result = self.generator.generate_next_task("dev123", "beginner", "Simple app", review)

        self.assertEqual(result['task_description'], "Debug code")

if __name__ == '__main__':
    unittest.main()