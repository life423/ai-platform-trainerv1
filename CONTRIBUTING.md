# Contributing to AI Platform Trainer

Thank you for your interest in contributing to AI Platform Trainer! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-platform-trainerv1.git
   cd ai-platform-trainerv1
   ```

2. Create a virtual environment:
   ```bash
   # Using conda (recommended)
   conda env create -f config/environment-cpu.yml
   conda activate ai-platform-cpu
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Project Structure

The project is organized as follows:

```
ai_platform_trainer/
├── ai/                # AI models and training
├── core/              # Core engine components
├── engine/            # Game engine
├── entities/          # Game entities
├── gameplay/          # Game mechanics
└── utils/             # Utility functions
```

## Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all modules, classes, and functions
- Keep functions small and focused on a single task
- Use meaningful variable and function names

## Testing

Run tests using pytest:

```bash
python -m pytest
```

Add tests for new features in the `tests/` directory.

## Pull Request Process

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with descriptive commit messages:
   ```bash
   git commit -m "feat: add new feature"
   ```

3. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request on GitHub

5. Wait for code review and address any feedback

## Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding or modifying tests
- `chore:` for maintenance tasks

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.