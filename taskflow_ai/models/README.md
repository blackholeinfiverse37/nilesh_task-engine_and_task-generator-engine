# Trained Models Directory

This directory contains saved model checkpoints after training.

## Expected Files After Training:
- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration
- `training_metadata.json` - Training information
- `training_args.bin` - Training arguments

## Training Command:
```bash
cd taskflow_ai
python train.py
```

This will create the fine-tuned model files in `models/fine_tuned/` subdirectory.