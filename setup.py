#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="deepmc",
    version="0.0.1",
    description="Deep Marker Based Motion Capture Solving",
    author="",
    author_email="",
    url="https://github.com/beyaldiz/DeepMC",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
