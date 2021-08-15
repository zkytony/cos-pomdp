#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension

setup(name='cos-pomdp',
      packages=['cospomdp'],
      version='0.1',
      description='COS-POMDP',
      python_requires='>3.6',
      install_requires=[
          'ai2thor==3.3.4',
          'pomdp_py',
          'tqdm',
          'torch>=1.8.0',
          'prettytable',
          'pytz',
          'pandas',
          'seaborn'
      ],
      author='Kaiyu Zheng',
      author_email='kaiyutony@gmail.com')
