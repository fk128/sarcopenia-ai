#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()


with open('requirements.txt') as file:
    requirements = file.read().splitlines()

dependency_links = []
for i, req in enumerate(requirements):
    if 'git' in req:
        dependency_links.append(req)
        requirements[i] = req.split('/')[-1].split('.')[0]


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
    dependency_links=dependency_links,
    url='',
    version='0.1.0',
    zip_safe=False,
)
