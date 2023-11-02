#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='src',
    version='0.0.2',
    description='Medical Imaging Development Pipeline',
    author='Muhammad Haritsah Mukhlis',
    author_email='muhammad.mukhlis@kalbecorp.com',
    url='https://github.com/KalbeDigitalLab/AI-Vision-Lightning-Pipeline',
    install_requires=['pytorch-lightning', 'hydra-core'],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        'console_scripts': [
            'train_command = src.train:main',
            'eval_command = src.eval:main',
        ]
    },
)
