"""
Setup configuration for torch_compile_utils.

A high-level compilation orchestration framework for PyTorch 2.0+.
"""

from setuptools import setup, find_packages




# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torch_compile_utils",
    version="0.1.0",
    author="Neil Tan",
    description="High-level compilation orchestration framework for PyTorch 2.0+",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neil-tan/torch_compile_utils",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    keywords="pytorch torch compile optimization cuda-graphs transformers",
)
