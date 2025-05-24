from setuptools import setup, find_packages

setup(
    name="ai_platform_trainer",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ai-trainer = ai_platform_trainer.main:main"
        ]
    },
)