#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='bert4vector',
    version='v0.0.2.post2',
    description='an elegant bert4vector',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT Licence',
    url='https://github.com/Tongjilibo/bert4vector',
    author='Tongjilibo',
    install_requires=['torch>1.6', 'bert4torch', 'loguru', 'numpy'],
    packages=find_packages()
)