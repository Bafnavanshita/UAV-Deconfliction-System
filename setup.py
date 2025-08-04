#!/usr/bin/env python3
"""
Setup script for UAV Strategic Deconfliction System
FlytBase Robotics Assignment 2025
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="uav-deconfliction-system",
    version="1.0.0",
    author="FlytBase Robotics Team",
    author_email="assignment@flytbase.com",
    description="A comprehensive 4D deconfliction system for UAV missions in shared airspace",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flytbase/uav-deconfliction-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "docs": [
            "sphinx>=6.2.1",
            "sphinx-rtd-theme>=1.2.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "uav-deconfliction=main:main",
            "uav-test=tests.test_system:run_test_suite",
        ],
    },
    package_data={
        "src": ["*.py"],
        "tests": ["*.py"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="uav drone deconfliction airspace safety aviation robotics",
    project_urls={
        "Bug Reports": "https://github.com/flytbase/uav-deconfliction-system/issues",
        "Source": "https://github.com/flytbase/uav-deconfliction-system",
        "Documentation": "https://github.com/flytbase/uav-deconfliction-system/wiki",
    },
)