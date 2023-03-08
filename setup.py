#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()


def parse_requirements(path):
    with open(path) as requirements_file:
        requirements_base = requirements_file.read().splitlines()

    requirements = [
        f"{r.split('#egg=')[-1]}@{r}" for r in requirements_base if r.startswith("git+")
    ]
    requirements += [r for r in requirements_base if not r.startswith("git+")]
    return requirements


requirements = parse_requirements("requirements.txt")

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Fahdi Kanavati",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Code for sarcopenia AI tools",
    entry_points={
        'console_scripts': [
            'sarco_slice_detection_trainer=sarcopenia_ai.apps.slice_detection.trainer:main',
            'sarco_detect_slice=sarcopenia_ai.apps.slice_detection.predict:main',
            'sarco_seg_trainer=sarcopenia_ai.apps.segmentation.trainer:main',
            'sarco_predict=sarcopenia_ai.apps.slice_detection.predict:main'
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords='sarcopenia_ai',
    name='sarcopenia_ai',
    packages=find_packages(include=['sarcopenia_ai']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='',
    version='0.1.0',
    zip_safe=False,
)
