#!/usr/bin/env python3
"""
COMPLETE RL TRAINING LOOP IMPLEMENTATION
Includes: output → reward → model update → repeat cycle
"""

import json
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import logging
from mini_lm import MiniLM
from rewards import RewardSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_rl_training_loop(mini_lm, reward_system, num_iterations=10):
    """
    COMPLETE RL TRAINING LOOP: output → reward → model update → repeat

    This implements the full RL cycle:
    1. Generate outputs using current model
    2. Check schema compliance and calculate rewards
    3. Update model based on rewards
    4. Repeat for multiple iterations

    Args:
        mini_lm (MiniLM): The model to train with RL
        reward_system (RewardSystem): RL reward system
        num_iterations (int): Number of RL training iterations

    Returns:
        dict: RL training statistics
    """
    logger.info("🚀 Starting COMPLETE RL Training Loop (Output → Reward → Model Update)")

    # Setup optimizer for RL fine-tuning
    rl_optimizer = AdamW(mini_lm.model.parameters(), lr=0.0001, weight_decay=0.01)
    rl_criterion = nn.CrossEntropyLoss()

    rl_stats = {
        'iterations': num_iterations,
        'total_rewards': [],
        'avg_rewards': [],
        'training_losses': [],
        'model_updates': 0
    }

    for iteration in range(num_iterations):
        logger.info(f"\n📈 RL Iteration {iteration + 1}/{num_iterations}")

        # STEP 1: Generate outputs using current model
        sample_prompts = [
            "Generate a task for improving code documentation",
            "Create a task for adding unit tests",
            "Generate a refactoring task",
            "Create a task for performance optimization"
        ]

        iteration_rewards = []
        training_losses = []

        for prompt in sample_prompts:
            logger.info(f"🎯 Processing: {prompt[:40]}...")

            # Generate output using current model
            try:
                output = mini_lm.generate(prompt, max_tokens=150)
                logger.info(f"✅ Generated: {output[:80]}...")

                # STEP 2: Check schema compliance and calculate reward
                is_valid = reward_system._is_valid_json(output, {})
                reward = reward_system.reward_output(is_valid)
                iteration_rewards.append(reward)

                # Store feedback for future training
                reward_system.feedback_data.append((prompt, output, reward))

                logger.info(f"🎖️  Reward: {reward}")

                # STEP 3: RL Model Update - Fine-tune based on reward
                if reward > 0:  # Only update on positive rewards
                    loss = perform_rl_update(mini_lm, prompt, output, rl_optimizer, rl_criterion, reward)
                    training_losses.append(loss)
                    rl_stats['model_updates'] += 1
                    logger.info(f"🔄 RL Update Loss: {loss:.4f}")

            except Exception as e:
                logger.error(f"❌ Failed: {e}")
                reward = -1
                iteration_rewards.append(reward)
                reward_system.apply_rl_update(reward)

        # Calculate iteration statistics
        avg_reward = sum(iteration_rewards) / len(iteration_rewards)
        avg_loss = sum(training_losses) / len(training_losses) if training_losses else 0.0

        rl_stats['total_rewards'].extend(iteration_rewards)
        rl_stats['avg_rewards'].append(avg_reward)
        rl_stats['training_losses'].extend(training_losses)

        logger.info(f"📊 Iteration {iteration + 1} - Avg Reward: {avg_reward:.2f}, Training Loss: {avg_loss:.4f}")

        # Update learning rate based on performance
        reward_system.apply_rl_update(avg_reward)

        # Early stopping on good performance
        if avg_reward > 0.8 and iteration > 3:
            logger.info("🎉 High performance achieved!")
            break

    logger.info("🏁 RL Training Loop Completed")
    final_stats = reward_system.get_rl_stats()
    logger.info(f"📈 Final RL Stats: {final_stats}")

    return rl_stats

def perform_rl_update(mini_lm, prompt, output, optimizer, criterion, reward):
    """
    Perform one RL fine-tuning update step.

    Args:
        mini_lm (MiniLM): Model to update
        prompt (str): Input prompt
        output (str): Generated output
        optimizer: PyTorch optimizer
        criterion: Loss function
        reward (float): Reward value

    Returns:
        float: Training loss
    """
    # Tokenize prompt + output for training
    full_text = f"{prompt}\n{output}"
    tokens = mini_lm.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)

    # Create labels for causal LM training
    labels = tokens['input_ids'].clone()

    # RL: Scale loss by reward (higher reward = stronger learning)
    reward_scale = max(0.1, reward)  # Ensure positive scaling

    # Forward pass
    outputs = mini_lm.model(**tokens, labels=labels)
    loss = outputs.loss * reward_scale  # Reward-weighted loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mini_lm.model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

def load_dataset(file_path):
    """Load training data from JSONL file."""
    data_pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data_pairs.append(item)
    return data_pairs

def train_model(mini_lm, data_pairs, reward_system=None, num_epochs=3, batch_size=4, learning_rate=5e-5):
    """
    Complete training loop with loss function, optimizer, and RL integration.

    Args:
        mini_lm (MiniLM): The model to train
        data_pairs (list): Training data
        reward_system (RewardSystem, optional): RL system for reward integration
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate

    Returns:
        dict: Training statistics
    """
    logger.info("Setting up training...")

    # Create dataset and dataloader
    dataset = TaskDataset(data_pairs, mini_lm.tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer and loss function
    optimizer = AdamW(mini_lm.model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Calculate total training steps for scheduler
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Loss function (Cross-entropy for causal LM)
    loss_fn = nn.CrossEntropyLoss()

    # Training statistics
    stats = {
        'epochs': num_epochs,
        'total_steps': total_steps,
        'losses': [],
        'learning_rates': [],
        'rl_updates': 0
    }

    logger.info(f"Starting training for {num_epochs} epochs...")
    mini_lm.model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move batch to device (assuming CPU for now)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # Forward pass
            outputs = mini_lm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # RL reward integration - modify loss based on recent performance
            if reward_system:
                rl_stats = reward_system.get_rl_stats()
                if rl_stats['total_steps'] > 0:
                    # Adjust loss based on recent RL performance
                    reward_factor = max(0.5, min(2.0, 1.0 + rl_stats['average_reward'] * 0.5))
                    loss = loss * reward_factor
                    stats['rl_updates'] += 1
                    logger.debug(f"RL-adjusted loss: {loss.item():.4f} (factor: {reward_factor:.2f})")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(mini_lm.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Log progress
            if num_batches % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {num_batches}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

                stats['losses'].append(loss.item())
                stats['learning_rates'].append(current_lr)

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

        # RL integration: run RL training loop after each epoch
        if reward_system:
            logger.info("Running RL training loop after epoch...")
            reward_system.run_rl_training_loop(max_iterations=2)

    return stats

def save_model(mini_lm, output_dir="./models/fine_tuned"):
    """
    Save the trained model and tokenizer.

    Args:
        mini_lm (MiniLM): Trained model
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving model to {output_dir}...")
    mini_lm.model.save_pretrained(output_dir)
    mini_lm.tokenizer.save_pretrained(output_dir)

    # Save training metadata
    metadata = {
        "model_name": mini_lm.model_name,
        "vocab_size": mini_lm.tokenizer.vocab_size,
        "max_position_embeddings": getattr(mini_lm.model.config, 'max_position_embeddings', None),
        "trained_with_rl": True
    }

    with open(os.path.join(output_dir, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved successfully!")

def main():
    """Main RL training function - COMPLETE RL LOOP."""
    logger.info("🚀 Starting COMPLETE RL Training System")

    # Initialize components
    mini_lm = MiniLM()
    reward_system = RewardSystem(mini_lm)

    # STEP 1: Optional supervised pre-training
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'train.jsonl')
    if os.path.exists(dataset_path):
        logger.info("📚 Loading supervised training dataset...")
        data_pairs = load_dataset(dataset_path)
        logger.info(f"✅ Loaded {len(data_pairs)} training examples")

        # Optional: Quick supervised warm-up (commented out for pure RL focus)
        # logger.info("🔥 Performing supervised warm-up...")
        # train_model(mini_lm, data_pairs, num_epochs=1, batch_size=4)

    # STEP 2: MAIN RL TRAINING LOOP - output → reward → model update → repeat
    logger.info("🎯 Starting MAIN RL Training Loop...")
    rl_stats = run_rl_training_loop(mini_lm, reward_system, num_iterations=10)

    # STEP 3: Save the RL-trained model
    save_model(mini_lm)

    # STEP 4: Final evaluation and summary
    logger.info("🏁 RL Training completed!")
    logger.info(f"📊 RL Training Stats: {rl_stats}")
    logger.info(f"💾 Model saved to: ./models/fine_tuned/")
    logger.info("🎉 RL training system complete!")

if __name__ == "__main__":
    main()