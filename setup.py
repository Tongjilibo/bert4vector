#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

extra_require = {
    "bert4torch": ["bert4torch"],
    "transformers": ["transformers"],
    "sentence_transformers": ["sentence_transformers"]
}

setup(
    name='bert4vector',
    version='v0.0.6',
    description='an elegant bert4vector',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT Licence',
    url='https://github.com/Tongjilibo/bert4vector',
    author='Tongjilibo',
    install_requires=['loguru', 'numpy', 'torch4keras'],
    extras_require=extra_require,
    packages=find_packages()
)