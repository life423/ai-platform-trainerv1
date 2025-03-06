# AI Platform Trainer

This application provides an environment for training AI in a 2D game setting, with particular focus on missile evasion behaviors and enemy movement patterns.

## Features

- Interactive game environment with player and AI-controlled enemies
- Training mode for collecting data and supervised learning
- Play mode to test AI models in real-time
- Missile evasion AI for realistic enemy behavior
- Configurable headless mode for faster training
- Built-in data logging and model management

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- Pygame

### Installation

```
pip install -r requirements.txt
```

### Running the Application

#### Using the Interactive Launcher

For a menu of options, run:

```
run_game_examples.bat
```

#### Command Line Options

The application supports various command-line arguments:

```
python -m ai_platform_trainer.main [OPTIONS]
```

Available options:

- `--headless`: Run without graphical display (for training on servers)
- `--mode [train|play]`: Start directly in training or play mode
- `--batch-logging`: Enable batch logging (save data at end of session)
- `--training-speed X`: Run training at X times normal speed (e.g., 2.0)
- `--no-log`: Disable data logging
- `--fullscreen`: Run in fullscreen mode

### Examples

#### Regular Game
```
python -m ai_platform_trainer.main
```

#### Headless Training (2x speed)
```
python -m ai_platform_trainer.main --headless --mode train --training-speed 2.0
```

#### Start Directly in Play Mode
```
python -m ai_platform_trainer.main --mode play
```

## Project Structure

- `ai_platform_trainer/`: Main package
  - `ai_model/`: Neural network models and training code
  - `core/`: Core utilities (logging, config, launcher)
  - `entities/`: Game entities (player, enemy, missiles)
  - `gameplay/`: Game mechanics (collision, rendering, modes)
  - `utils/`: Utility functions and helpers
- `data/`: Training data storage
- `models/`: Trained model storage
- `tests/`: Unit tests

## Development

### CI/CD Pipeline

The project uses GitHub Actions for:
- Linting with flake8
- Formatting checks with black
- Headless smoke tests
- Windows executable builds

### Model Management

The `model_manager.py` utility provides:
- Versioned model saving with timestamps
- Loading models by version or latest
- Model validation and testing

### Contributing

Contributions are welcome! Please ensure:
- Code passes linting with flake8
- New features include tests
- Documentation is updated

## License

This project is licensed under the MIT License - see the LICENSE file for details.
