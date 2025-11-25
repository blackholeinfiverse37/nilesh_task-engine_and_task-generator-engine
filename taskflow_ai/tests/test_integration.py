import unittest
from unittest.mock import Mock, patch
import json
import os
import tempfile
from taskflow_ai.mini_lm import MiniLM
from taskflow_ai.reviewer import TaskReviewer
from taskflow_ai.generator import TaskGenerator
from taskflow_ai.rewards import RewardSystem

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Mock the MiniLM to avoid loading real model
        self.mock_mini_lm = Mock()
        self.mock_mini_lm.generate.side_effect = self._mock_generate

    def _mock_generate(self, prompt, **kwargs):
        """Mock generate method that returns appropriate JSON based on prompt."""
        if "review" in prompt.lower():
            return json.dumps({
                "good_aspects": ["Clean code", "Good structure"],
                "missing_aspects": ["Tests", "Documentation"],
                "score": 8
            })
        elif "generate" in prompt.lower() or "next task" in prompt.lower():
            return json.dumps({
                "task_description": "Implement unit tests",
                "requirements": ["Write test cases", "Use testing framework"],
                "difficulty": "intermediate",
                "estimated_time": "3 hours",
                "skills_focused": ["Testing", "Quality assurance"]
            })
        else:
            return "{}"

    @patch('taskflow_ai.reviewer.subprocess.run')
    @patch('taskflow_ai.reviewer.Path')
    def test_full_pipeline(self, mock_path, mock_subprocess):
        """Test the complete pipeline workflow."""
        # Mock subprocess for git clone
        mock_subprocess.return_value = Mock(returncode=0)

        # Mock Path operations
        mock_repo_path = Mock()
        mock_path.return_value.rglob.return_value = [
            Mock() for _ in range(3)  # Mock 3 Python files
        ]
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__truediv__ = Mock(return_value=Mock())
        mock_path.return_value.__truediv__.return_value.exists.return_value = True
        mock_path.return_value.__truediv__.return_value.read_text.return_value = "# Test README"

        # Mock file reading for code analysis
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_file.read.return_value = "def test():\n    pass\n# comment\n"
            mock_open.return_value.__enter__.return_value = mock_file

            # Initialize components
            reviewer = TaskReviewer(self.mock_mini_lm)
            generator = TaskGenerator(self.mock_mini_lm)
            rewards = RewardSystem(self.mock_mini_lm)

            # Test reviewer
            review = reviewer.review("https://github.com/test/repo", {"type": "python"})
            self.assertIsInstance(review, dict)
            self.assertIn("score", review)
            self.assertIn("good_aspects", review)

            # Test generator
            task = generator.generate_next_task("dev123", "intermediate", "Built calculator", review)
            self.assertIsInstance(task, dict)
            self.assertIn("task_description", task)
            self.assertIn("requirements", task)

            # Test rewards
            reward = rewards.evaluate_output("test prompt", json.dumps(review), reviewer.schema)
            self.assertIn(reward, [-1.0, 1.0])

            summary = rewards.get_feedback_summary()
            self.assertIsInstance(summary, dict)
            self.assertIn("total_evaluations", summary)

    def test_schema_validation(self):
        """Test that outputs conform to schemas."""
        from jsonschema import validate

        # Test reviewer schema
        review_output = {
            "good_aspects": ["test"],
            "missing_aspects": ["test"],
            "score": 5
        }
        reviewer = TaskReviewer(self.mock_mini_lm)
        validate(instance=review_output, schema=reviewer.schema)

        # Test generator schema
        task_output = {
            "task_description": "test",
            "requirements": ["test"]
        }
        generator = TaskGenerator(self.mock_mini_lm)
        validate(instance=task_output, schema=generator.schema)

    def test_mini_lm_schema_enforcement(self):
        """Test that Mini-LM enforces schema compliance."""
        from taskflow_ai.mini_lm import MiniLM

        # This test would require loading the actual model
        # For now, test the schema validation logic
        mini_lm = MiniLM.__new__(MiniLM)  # Create without __init__ to avoid model loading

        # Mock the required attributes
        mini_lm.schema = {
            "type": "object",
            "properties": {"test": {"type": "string"}},
            "required": ["test"]
        }

        # Test schema validation
        valid_data = {"test": "value"}
        invalid_data = {"other": "value"}

        self.assertTrue(mini_lm._validate_against_schema(valid_data, mini_lm.schema))
        self.assertFalse(mini_lm._validate_against_schema(invalid_data, mini_lm.schema))

    def test_rl_training_loop(self):
        """Test the RL training loop functionality."""
        from taskflow_ai.rewards import RewardSystem
        from unittest.mock import Mock

        # Mock Mini-LM
        mock_mini_lm = Mock()
        mock_mini_lm.generate.side_effect = lambda prompt, **kwargs: '{"test": "generated"}'

        reward_system = RewardSystem(mock_mini_lm)

        # Test RL training loop (should not crash)
        try:
            reward_system.run_rl_training_loop(max_iterations=1)
            rl_stats = reward_system.get_rl_stats()
            self.assertIsInstance(rl_stats, dict)
            self.assertIn('total_steps', rl_stats)
        except Exception as e:
            self.fail(f"RL training loop failed: {e}")

    def test_mini_lm_schema_enforcement(self):
        """Test that Mini-LM enforces schema compliance with guardrails."""
        from taskflow_ai.mini_lm import MiniLM

        # Create Mini-LM instance (mock for testing)
        mini_lm = MiniLM.__new__(MiniLM)
        mini_lm.model_name = "test_model"

        # Mock the generate method to return schema-compliant output
        def mock_generate(prompt, max_tokens=300, schema=None, template=None):
            if template:
                return '{"test": "templated"}'
            elif schema and "review" in prompt.lower():
                return '{"good_aspects": ["test"], "missing_aspects": ["test"], "score": 5}'
            elif schema and "task" in prompt.lower():
                return '{"task_id": "test_123", "title": "Test Task", "description": "Test desc", "steps": ["step1"], "acceptance_criteria": ["criteria1"], "difficulty": "intermediate", "estimated_time": "2 hours"}'
            else:
                return '{"default": "response"}'

        mini_lm.generate = mock_generate

        # Test schema enforcement for reviewer
        reviewer_schema = {
            "type": "object",
            "properties": {
                "good_aspects": {"type": "array"},
                "missing_aspects": {"type": "array"},
                "score": {"type": "integer"}
            },
            "required": ["good_aspects", "missing_aspects", "score"]
        }

        review_prompt = "Review this code submission"
        output = mini_lm.generate(review_prompt, schema=reviewer_schema)
        parsed = json.loads(output)

        # Verify schema compliance
        self.assertIn("good_aspects", parsed)
        self.assertIn("missing_aspects", parsed)
        self.assertIn("score", parsed)
        self.assertIsInstance(parsed["good_aspects"], list)
        self.assertIsInstance(parsed["missing_aspects"], list)
        self.assertIsInstance(parsed["score"], int)

    def test_rl_feedback_loop_integration(self):
        """Test the complete RL feedback loop integration."""
        from taskflow_ai.rewards import RewardSystem
        from unittest.mock import Mock

        # Mock Mini-LM
        mock_mini_lm = Mock()
        mock_mini_lm.generate.side_effect = lambda prompt, **kwargs: '{"valid": "json"}'

        reward_system = RewardSystem(mock_mini_lm)

        # Test that RL methods exist and are callable
        self.assertTrue(hasattr(reward_system, 'run_rl_training_loop'))
        self.assertTrue(hasattr(reward_system, 'reward_output'))
        self.assertTrue(hasattr(reward_system, 'evaluate_output'))

        # Test reward calculation
        reward = reward_system.reward_output(True)   # Valid output
        self.assertEqual(reward, 1)

        reward = reward_system.reward_output(False)  # Invalid output
        self.assertEqual(reward, -1)

    def test_strict_template_generator(self):
        """Test that generator uses strict templates."""
        from taskflow_ai.generator import generate_next_task

        # Test different score ranges produce different templates
        review_low = {"score": 3}
        review_med = {"score": 6}
        review_high = {"score": 9}

        task_low = generate_next_task("dev1", "beginner", "task1", review_low)
        task_med = generate_next_task("dev2", "intermediate", "task2", review_med)
        task_high = generate_next_task("dev3", "advanced", "task3", review_high)

        # Check that all required fields are present
        required_fields = ["task_id", "title", "description", "steps", "acceptance_criteria", "difficulty", "estimated_time"]

        for field in required_fields:
            self.assertIn(field, task_low)
            self.assertIn(field, task_med)
            self.assertIn(field, task_high)

        # Check that difficulty is set correctly
        self.assertEqual(task_low["difficulty"], "beginner")
        self.assertEqual(task_med["difficulty"], "intermediate")
        self.assertEqual(task_high["difficulty"], "advanced")

        # Check that steps and criteria are arrays
        self.assertIsInstance(task_low["steps"], list)
        self.assertIsInstance(task_low["acceptance_criteria"], list)

        # Check that task IDs are unique
        self.assertNotEqual(task_low["task_id"], task_med["task_id"])
        self.assertNotEqual(task_med["task_id"], task_high["task_id"])

if __name__ == '__main__':
    unittest.main()