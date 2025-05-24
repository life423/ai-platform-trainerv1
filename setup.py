from setuptools import setup, find_packages

setup(
    name="ai_platform_trainer",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ai-trainer = unified_launcher:main"
        ]
    },
    install_requires=[
        "pygame>=2.1.0",
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "gymnasium>=0.28.0",
        "stable-baselines3>=1.6.0",
        "noise>=1.2.2",
    ],
    extras_require={
        "dev": [
            "black>=23.1.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pre-commit>=3.0.0",
        ]
    },
    python_requires=">=3.8",
)