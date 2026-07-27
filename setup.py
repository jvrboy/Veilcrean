"""
Setup script for the Veilcrean Python brain.

This is *not* a hard requirement — you can also just `pip install -r
requirements.txt` and run `python -m python_brain.main`. But having
setup.py means the brain can be installed in editable mode (`pip
install -e .`) and imported cleanly from anywhere.
"""
from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).resolve().parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="veilcrean",
    version="1.0.0",
    description="Veilcrean — Adaptive AI trading bot (Python brain)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Veilcrean",
    license="MIT",
    python_requires=">=3.11",
    packages=find_packages(exclude=["mt5_ea", "tests", "docs", "scripts", "models", "data", "logs"]),
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.1",
        "torch>=2.1",
        "pyzmq>=25.1",
        "sqlalchemy>=2.0",
        "loguru>=0.7",
        "requests>=2.31",
    ],
    extras_require={
        "viz": ["matplotlib>=3.8", "plotly>=5.17", "dash>=2.14"],
        "alerts": ["python-telegram-bot>=20.7"],
        "dev": ["pytest>=7.4", "pytest-cov>=4.1", "black>=23.10", "flake8>=6.1", "mypy>=1.6"],
    },
    entry_points={
        "console_scripts": [
            "veilcrean=python_brain.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
