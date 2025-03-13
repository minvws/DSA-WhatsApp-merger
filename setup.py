import os

import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()


setuptools.setup(
    name="chatmerger",
    version="1.0.8",
    author='Erik van Egmond',
    author_email= 'data-science-nc19@minvws.nl',
    description="A package to merge exports from different people and moments in time into one conversation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    test_suite="tests",
    packages=setuptools.find_packages(include=['chatmerger', 'chatmerger.*']),
    # package_dir={"": "chatmerger"},
    package_data={'chatmerger': ['font/*']},
    python_requires='>=3.9',
)