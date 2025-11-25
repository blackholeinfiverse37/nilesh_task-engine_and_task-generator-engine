import json
import os
from jsonschema import validate, ValidationError
from .mini_lm import MiniLM

def reward_output(is_valid):
    return 1 if is_valid else -1

class RewardSystem:
    def __init__(self, mini_lm: MiniLM):
        self.mini_lm = mini_lm
        self.feedback_data = []  # List of (prompt, output, reward)
        self.learning_rate = 0.01  # Simple learning rate for RL simulation
        self.performance_history = []

    def evaluate_output(self, prompt: str, output: str, schema: dict) -> float:
        """
        Evaluate the generated output against the schema and return a reward.
        Applies RL update based on the reward.

        Args:
            prompt (str): The input prompt
            output (str): The generated output
            schema (dict): JSON schema to validate against

        Returns:
            float: Reward value (+1 for valid, -1 for invalid)
        """
        is_valid = self._is_valid_json(output, schema)
        reward = self.reward_output(is_valid)

        # Store for potential retraining
        self.feedback_data.append((prompt, output, reward))

        # Apply RL update
        self.apply_rl_update(reward)

        return reward

    def reward_output(self, is_valid):
        return 1 if is_valid else -1

    def _is_valid_json(self, output: str, schema: dict) -> bool:
        """
        Check if output is valid JSON matching schema.
        """
        try:
            parsed = json.loads(output)
            validate(instance=parsed, schema=schema)
            return True
        except (json.JSONDecodeError, ValidationError):
            return False

    def get_feedback_summary(self):
        """
        Get a summary of feedback data.

        Returns:
            dict: Summary statistics
        """
        if not self.feedback_data:
            return {"total_evaluations": 0, "positive_rewards": 0, "negative_rewards": 0, "accuracy": 0.0}

        total = len(self.feedback_data)
        positive = sum(1 for _, _, r in self.feedback_data if r > 0)
        accuracy = positive / total

        return {
            "total_evaluations": total,
            "positive_rewards": positive,
            "negative_rewards": total - positive,
            "accuracy": accuracy
        }

    def run_rl_training_loop(self, max_iterations=10, train_batch_size=4):
        """
        Run the COMPLETE RL FEEDBACK LOOP with actual model training.

        This implements: output → reward → update model → repeat

        Args:
            max_iterations (int): Maximum number of RL training iterations
            train_batch_size (int): Batch size for training updates
        """
        print("🚀 Starting RL Training Loop (Output → Reward → Model Update)")
        print("=" * 60)

        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        # Setup optimizer for RL fine-tuning
        rl_optimizer = AdamW(self.mini_lm.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        rl_criterion = nn.CrossEntropyLoss()

        for iteration in range(max_iterations):
            print(f"\n📈 RL Training Iteration {iteration + 1}/{max_iterations}")

            # Step 1: Generate outputs using current model
            sample_prompts = [
                "Generate a task for improving code documentation",
                "Create a task for adding unit tests",
                "Generate a refactoring task",
                "Create a task for performance optimization"
            ]

            iteration_losses = []
            iteration_rewards = []

            for prompt in sample_prompts:
                print(f"🎯 Processing: {prompt[:40]}...")

                # Generate output
                try:
                    output = self.mini_lm.generate(prompt, max_tokens=150)
                    print(f"✅ Output: {output[:80]}...")

                    # Step 2: Calculate reward
                    is_valid = self._is_valid_json(output, {})
                    reward = self.reward_output(is_valid)
                    iteration_rewards.append(reward)

                    # Store feedback for training
                    self.feedback_data.append((prompt, output, reward))

                    print(f"🎖️  Reward: {reward}")

                    # Step 3: RL Update - Fine-tune model based on reward
                    if reward > 0:  # Positive reward = reinforce good behavior
                        loss = self._rl_fine_tune_step(prompt, output, rl_optimizer, rl_criterion, reward)
                        iteration_losses.append(loss)
                        print(f"🔄 RL Update Loss: {loss:.4f}")

                except Exception as e:
                    print(f"❌ Failed: {e}")
                    reward = -1
                    iteration_rewards.append(reward)
                    self.apply_rl_update(reward)

            # Calculate iteration statistics
            avg_reward = sum(iteration_rewards) / len(iteration_rewards)
            avg_loss = sum(iteration_losses) / len(iteration_losses) if iteration_losses else 0.0

            print(f"📊 Iteration {iteration + 1} - Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
            print(f"🧠 Learning Rate: {self.learning_rate:.6f}")

            # Step 4: Check performance and decide next action
            if avg_reward < 0.3 and iteration > 2:
                print("🔄 Low performance - triggering full retraining...")
                self.retrain_if_needed(threshold_accuracy=0.5, min_samples=len(iteration_rewards))

            if avg_reward > 0.8 and iteration > 3:
                print("🎉 High performance achieved!")
                break

        print("\n🏁 RL Training Loop Completed")
        final_stats = self.get_rl_stats()
        print(f"📈 Final RL Stats: {final_stats}")

    def _rl_fine_tune_step(self, prompt, output, optimizer, criterion, reward):
        """
        Perform one RL fine-tuning step on the model.

        Args:
            prompt (str): Input prompt
            output (str): Generated output
            optimizer: PyTorch optimizer
            criterion: Loss function
            reward (float): Reward value

        Returns:
            float: Training loss
        """
        import torch

        # Tokenize prompt + output as training sequence
        full_text = f"{prompt}\n{output}"
        tokens = self.mini_lm.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)

        # Create labels (shifted input_ids for causal LM)
        labels = tokens['input_ids'].clone()

        # RL: Scale loss by reward (higher reward = stronger learning)
        reward_scale = max(0.1, reward)  # Ensure positive scaling

        optimizer.zero_grad()

        outputs = self.mini_lm.model(**tokens, labels=labels)
        loss = outputs.loss * reward_scale  # RL: Reward-weighted loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mini_lm.model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    def apply_rl_update(self, reward: float):
        """
        Apply reinforcement learning update based on reward.

        Args:
            reward (float): Reward value (+1 or -1)
        """
        # Simple RL: adjust learning rate based on performance
        if reward > 0:
            self.learning_rate *= 1.01  # Increase learning rate for positive rewards
        else:
            self.learning_rate *= 0.99  # Decrease learning rate for negative rewards

        # Keep learning rate in reasonable bounds
        self.learning_rate = max(0.001, min(0.1, self.learning_rate))

        # Track performance for analysis
        self.performance_history.append(reward)

    def get_rl_stats(self):
        """
        Get RL performance statistics.

        Returns:
            dict: RL statistics
        """
        if not self.performance_history:
            return {"total_steps": 0, "average_reward": 0.0, "learning_rate": self.learning_rate}

        avg_reward = sum(self.performance_history) / len(self.performance_history)
        return {
            "total_steps": len(self.performance_history),
            "average_reward": avg_reward,
            "learning_rate": self.learning_rate,
            "recent_performance": self.performance_history[-10:]  # Last 10 rewards
        }

    def retrain_if_needed(self, threshold_accuracy=0.8, min_samples=10):
        """
        Trigger retraining if accuracy is below threshold and enough samples.
        Includes RL-based decision making.

        Args:
            threshold_accuracy (float): Minimum accuracy to avoid retraining
            min_samples (int): Minimum feedback samples before retraining
        """
        summary = self.get_feedback_summary()
        rl_stats = self.get_rl_stats()

        # RL-enhanced decision: consider recent performance
        recent_avg = sum(rl_stats.get("recent_performance", [])) / max(1, len(rl_stats.get("recent_performance", [])))

        should_retrain = (
            summary["total_evaluations"] >= min_samples and
            (summary["accuracy"] < threshold_accuracy or recent_avg < 0.5)
        )

        if should_retrain:
            # Prepare data for fine-tuning: use positive examples
            positive_pairs = [{"input": p, "output": o} for p, o, r in self.feedback_data if r > 0]
            if positive_pairs:
                print(f"[RL] Retraining model with {len(positive_pairs)} positive examples...")
                print(f"[RL] Current learning rate: {self.learning_rate:.4f}")
                print(f"[RL] Recent average reward: {recent_avg:.2f}")

                self.mini_lm.fine_tune(positive_pairs)
                # Clear feedback data after retraining
                self.feedback_data = []
                print("[RL] Retraining completed. Model updated.")
            else:
                print("[RL] No positive examples available for retraining.")