# AI Model Structure

This directory contains all the code related to AI models used in the game.

## Directory Structure

- `model_definition/` - Contains all model architectures
  - `enemy_movement_model.py` - Neural network for enemy movement predictions
  - `simple_missile_model.py` - Neural network for missile control
  - `__init__.py` - Package initialization
- `train.py` - Unified script for training all models
- `missile_dataset.py` - Dataset class for missile training data
- `enemy_dataset.py` - Dataset class for enemy training data

## Training Models

You can train models using the centralized `train.py` script:

```bash
# Train the missile model with default parameters
python -m ai_platform_trainer.ai_model.train --model missile

# Train the enemy model with custom parameters
python -m ai_platform_trainer.ai_model.train --model enemy --epochs 100 --batch_size 64 --learning_rate 0.001
```

### Command Line Arguments

- `--model`: Required. Which model to train (`missile` or `enemy`)
- `--data_file`: Optional. JSON file with training data (default: latest in data/raw)
- `--epochs`: Optional. Number of training epochs (default: model-specific)
- `--batch_size`: Optional. Training batch size (default: 32)
- `--learning_rate`: Optional. Learning rate (default: 0.001)
- `--hidden_size`: Optional. Size of hidden layers (default: model-specific)
- `--dropout_prob`: Optional. Dropout probability (for enemy model only, default: 0.3)
- `--no_save_best`: Optional. Don't save best model during training (default: save best)
- `--models_dir`: Optional. Directory to save trained models (default: models)

## Adding New Models

To add a new model:

1. Add the model definition in `model_definition/`
2. Create a dataset class if needed
3. Add a training function in `train.py`
4. Update the main function in `train.py` to handle your new model
5. Add the model to `utils/model_manager.py`
