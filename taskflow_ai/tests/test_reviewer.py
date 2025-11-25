import unittest
from unittest.mock import Mock, patch
import json
import os
from taskflow_ai.reviewer import TaskReviewer

class TestTaskReviewer(unittest.TestCase):
    def setUp(self):
        self.mock_mini_lm = Mock()
        self.reviewer = TaskReviewer(self.mock_mini_lm)

    @patch('taskflow_ai.reviewer.requests.get')
    def test_review_valid_output(self, mock_get):
        # Mock GitHub API
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'content': 'VGVzdCBSRUFETUU='}  # 'Test README' in base64
        mock_get.return_value = mock_response

        # Mock valid JSON output
        valid_output = json.dumps({
            "good_aspects": ["Clear code structure", "Good documentation"],
            "missing_aspects": ["Unit tests", "Error handling"],
            "score": 7
        })
        self.mock_mini_lm.generate.return_value = valid_output

        result = self.reviewer.review("https://github.com/user/repo", {"task_type": "coding"})
        self.assertEqual(result['score'], 7)
        self.assertIn("Clear code structure", result['good_aspects'])

    @patch('taskflow_ai.reviewer.requests.get')
    def test_review_invalid_output_corrected(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'content': 'VGVzdA=='}  # 'Test' in base64
        mock_get.return_value = mock_response

        # Mock invalid output that can be corrected
        invalid_output = 'Some text {"good_aspects": ["good"], "missing_aspects": ["bad"], "score": 5} more text'
        self.mock_mini_lm.generate.return_value = invalid_output

        result = self.reviewer.review("https://github.com/user/repo", {})
        self.assertEqual(result['score'], 5)

    @patch('taskflow_ai.reviewer.requests.get')
    def test_review_api_failure(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError):
            self.reviewer.review("https://github.com/user/repo", {})

if __name__ == '__main__':
    unittest.main()