# TaskFlow AI

A modular micro-system for reviewing submitted tasks, scoring them using a fixed rubric, identifying gaps, and generating the next task. Powered by a small in-house fine-tunable mini-LM with rule-bound outputs and RL-style feedback loops.

## Overview

TaskFlow AI is designed as a standalone core engine that can be plugged into larger BHIV modules. It provides:

- **Structured Task Reviews**: Analyzes GitHub repositories and provides rubric-based evaluations
- **Personalized Task Generation**: Creates next tasks tailored to developer skill levels and progress
- **Constrained AI Generation**: Uses guardrails and schemas to ensure predictable, rule-bound outputs
- **Continuous Improvement**: RL feedback loop nudges the model toward better performance

## Features

- **Task Reviewer Module**: Input GitHub repo URL + metadata, output structured review (good/missing aspects, score/10)
- **Task Generator Module**: Input developer info + last review, output next task in tight instruction format
- **In-House Mini-LM**: Tiny LLM (TinyLlama-1.1B-Chat-v1.0) with fine-tuning support and strict JSON schema guardrails
- **RL Feedback Loop**: Rewards accurate outputs (+1), penalties for violations (-1), triggers retraining when needed
- **Validation Layer**: Automatic schema checks and correction attempts
- **Pure Python**: No external architecture assumptions, easily integrable

## Installation

```bash
pip install -r taskflow_ai/requirements.txt
```

## Quick Start

```python
from taskflow_ai.mini_lm import MiniLM
from taskflow_ai.reviewer import TaskReviewer
from taskflow_ai.generator import TaskGenerator
from taskflow_ai.rewards import RewardSystem

# Initialize the system
mini_lm = MiniLM()  # Loads TinyLlama-1.1B-Chat-v1.0 by default
reviewer = TaskReviewer(mini_lm)
generator = TaskGenerator(mini_lm)
reward_system = RewardSystem(mini_lm)

# Review a submitted task (GitHub repo)
review = reviewer.review(
    "https://github.com/example/task-repo",
    {"task_type": "web app", "framework": "React"}
)

print(f"Review Score: {review['score']}/10")
print(f"Good aspects: {review['good_aspects']}")
print(f"Missing aspects: {review['missing_aspects']}")

# Generate next task for developer
next_task = generator.generate_next_task(
    developer_id="dev_123",
    current_skill="intermediate",
    last_task="Built a todo app",
    review=review
)

print(f"Next Task: {next_task['task_description']}")
print(f"Difficulty: {next_task['difficulty']}")
print(f"Estimated Time: {next_task['estimated_time']}")

# Evaluate and improve
reward = reward_system.evaluate_output(
    "generation prompt",
    str(next_task),
    generator.schema
)
print(f"Reward: {reward}")

# Check if retraining needed
reward_system.retrain_if_needed()
```

## Training the Model

### Fine-tuning from Scratch

```bash
cd taskflow_ai
python train.py
```

### Custom Training Data

Add JSONL entries to `dataset/train.jsonl` in the format:
```json
{"input": "Review this Python code with 5 files, 200 lines...", "output": "{\"good_aspects\": [\"Clean code\"], \"missing_aspects\": [\"Tests\"], \"score\": 8}"}
```

### Fine-tuning Programmatically

```python
from taskflow_ai.mini_lm import MiniLM

mini_lm = MiniLM()
training_data = [
    {"input": "Review this code...", "output": '{"good_aspects": [...], "score": 8}'},
    # ... more pairs
]
mini_lm.fine_tune(training_data, num_epochs=3)
```

## Running Tests

```bash
cd taskflow_ai
python -m unittest discover tests/
```

## Running the Pipeline

### End-to-End Pipeline

Run the complete system with a GitHub repository:

```bash
cd taskflow_ai
python main.py https://github.com/microsoft/vscode dev123 intermediate "Built a calculator app"
```

### Real GitHub Analysis Demo

Run the demo that actually clones and analyzes a real GitHub repository:

```bash
cd taskflow_ai
python demo_real.py
```

This will:
- Clone the octocat/Hello-World repository
- Analyze its file structure and languages
- Check for README existence
- Generate a rubric score
- Create a personalized next task
- Demonstrate the reward system

### Mock Demo

Run the demo with mocked components:

```bash
python -m taskflow_ai.demo
```

## Project Structure

```
taskflow_ai/
├── __init__.py         # Package initialization
├── main.py             # Main end-to-end pipeline script
├── mini_lm.py          # MiniLM class for language model operations
├── reviewer.py         # TaskReviewer class for reviewing submissions
├── generator.py        # TaskGenerator class for creating next tasks
├── rewards.py          # RewardSystem for RL feedback
├── pipeline.py         # Alternative pipeline script
├── train.py            # Model training script
├── demo.py             # Demonstration script
├── requirements.txt    # Python dependencies
├── config/
│   └── config.yaml     # Configuration settings
├── dataset/
│   └── train.jsonl     # Training data for fine-tuning
├── schemas/
│   ├── reviewer_schema.json    # Schema for review outputs
│   └── generator_schema.json   # Schema for task generation outputs
└── tests/
    ├── test_reviewer.py
    ├── test_generator.py
    ├── test_rewards.py
    └── test_integration.py
```

## API Reference

### MiniLM
- `MiniLM(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")`: Initialize with optional model
- `fine_tune(data_pairs, output_dir, num_epochs)`: Fine-tune on input-output pairs
- `generate(prompt, max_length, schema)`: Generate text with optional schema validation
- `load_fine_tuned(model_path)`: Load a fine-tuned model

### TaskReviewer
- `TaskReviewer(mini_lm)`: Initialize with MiniLM instance
- `review(repo_url, metadata)`: Review a GitHub repo and return structured evaluation

### TaskGenerator
- `TaskGenerator(mini_lm)`: Initialize with MiniLM instance
- `generate_next_task(dev_id, skill, last_task, review)`: Generate next personalized task

### RewardSystem
- `RewardSystem(mini_lm)`: Initialize with MiniLM instance
- `evaluate_output(prompt, output, schema)`: Evaluate output and return reward
- `get_feedback_summary()`: Get statistics on feedback
- `retrain_if_needed(threshold, min_samples)`: Trigger retraining if performance is poor

## Dependencies

- `transformers`: For language model operations
- `torch`: PyTorch for model training/inference
- `requests`: For GitHub API calls
- `jsonschema`: For JSON schema validation

## License

MIT License - feel free to use in BHIV modules or other projects.

## Contributing

Built by Nilesh for the BHIV ecosystem. Contributions welcome!