import unittest
import json
from taskflow_ai.rewards import RewardSystem
from unittest.mock import Mock

class TestRewardSystem(unittest.TestCase):
    def setUp(self):
        self.mock_mini_lm = Mock()
        self.reward_system = RewardSystem(self.mock_mini_lm)

    def test_evaluate_output_valid(self):
        schema = {
            "type": "object",
            "properties": {"score": {"type": "integer"}},
            "required": ["score"]
        }
        output = json.dumps({"score": 8})
        reward = self.reward_system.evaluate_output("prompt", output, schema)
        self.assertEqual(reward, 1.0)

    def test_evaluate_output_invalid(self):
        schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
        output = "invalid json"
        reward = self.reward_system.evaluate_output("prompt", output, schema)
        self.assertEqual(reward, -1.0)

    def test_get_feedback_summary(self):
        schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
        self.reward_system.evaluate_output("p1", json.dumps({"score": 8}), schema)
        self.reward_system.evaluate_output("p2", "invalid", schema)
        summary = self.reward_system.get_feedback_summary()
        self.assertEqual(summary["total_evaluations"], 2)
        self.assertEqual(summary["positive_rewards"], 1)
        self.assertEqual(summary["accuracy"], 0.5)

if __name__ == '__main__':
    unittest.main()