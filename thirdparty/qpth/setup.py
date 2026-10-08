from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="qpth",
    version="0.0.19",
    description="A fast and differentiable QP solver for modern PyTorch.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Brandon Amos",
    author_email="bamos@cs.cmu.edu",
    platforms=["any"],
    license="Apache-2.0",
    url="https://github.com/locuslab/qpth",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0,<3",
        "torch>=2.0",
    ],
    extras_require={
        "cvxpy": ["cvxpy>=1.6"],
        "test": ["pytest>=8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
