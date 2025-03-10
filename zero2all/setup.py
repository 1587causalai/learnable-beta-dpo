#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="zero2all",
    version="0.1.0",
    description="从零开始建立自己的大语言模型",
    author="zero2all",
    author_email="example@domain.com",
    url="https://github.com/yourusername/zero2all",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "regex>=2022.1.18",
        "tqdm>=4.62.0",
        "sentencepiece>=0.1.96",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
) 