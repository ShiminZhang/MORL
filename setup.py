"""
Setup script for MORL package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="morl",
    version="0.1.0",
    description="Multi-Objective Reinforcement Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MORL Team",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8",
)

