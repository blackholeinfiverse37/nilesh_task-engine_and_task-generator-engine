import json
import jsonschema


class RewardSystem:
    """
    Reward system for evaluating JSON outputs against schemas and applying RL feedback.
    """

    def __init__(self):
        self.feedback_data = []

    # -----------------------------
    # JSON Validation
    # -----------------------------
    def _is_valid_json(self, output, schema_path):
        """
        Validates a JSON object using the provided schema.
        """
        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)

            jsonschema.validate(instance=output, schema=schema)
            return True

        except Exception as e:
            print(f"[Validation Error] {e}")
            return False

    # -----------------------------
    # Reward Function
    # -----------------------------
    def reward_output(self, is_valid):
        """
        Returns +1 if JSON is valid, -1 otherwise.
        """
        return 1 if is_valid else -1

    # -----------------------------
    # Evaluation Wrapper
    # -----------------------------
    def evaluate_output(self, prompt, output, schema_path):
        """
        Validates LM output and records feedback.
        """
        is_valid = self._is_valid_json(output, schema_path)
        reward = self.reward_output(is_valid)

        self.feedback_data.append({
            "prompt": prompt,
            "output": output,
            "reward": reward
        })

        return reward

    # -----------------------------
    # RL Training Loop (Simplified)
    # -----------------------------
    def train_model(self, model, prompt, expected_schema_path, num_iterations=5):
        """
        Trains the model using RL-style feedback.
        This is lightweight and uses the model's built-in fine-tuning method.
        """
        print("\n[RL Training Started]")

        for iteration in range(num_iterations):

            print(f"Iteration {iteration + 1}/{num_iterations}")

            # Generate output
            output = model.generate(
                prompt=prompt,
                schema_path=expected_schema_path
            )

            # Evaluate
            reward = self.evaluate_output(
                prompt=prompt,
                output=output,
                schema_path=expected_schema_path
            )

            print(f"Reward: {reward}")

            # Apply RL fine-tuning step in model
            model.fine_tune_rl(reward)

        print("[RL Training Complete]\n")
