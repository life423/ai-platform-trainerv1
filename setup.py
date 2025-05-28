from setuptools import setup, find_packages
import os

setup(
    name="ai_platform_trainer",
    version="0.1.0",
    author="AI Platform Trainer Team",
    description="AI Platform Game Trainer",
    long_description=(
        open("README.md").read() if os.path.exists("README.md") else ""
    ),
    long_description_content_type="text/markdown",
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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment",
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
