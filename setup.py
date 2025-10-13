#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

def get_requires() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines

extra_require = {
    "bert4torch": ["bert4torch"],
    "transformers": ["transformers"],
    "sentence_transformers": ["sentence_transformers"]
}

setup(
    name='bert4vector',
    version='v0.0.7.post2',
    description='an elegant bert4vector',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT Licence',
    url='https://github.com/Tongjilibo/bert4vector',
    author='Tongjilibo',
    author_email='tongjilibo@163.com',
    install_requires=get_requires(),
    extras_require=extra_require,
    packages=find_packages(),
    include_package_data=True
)