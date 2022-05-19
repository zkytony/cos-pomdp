#!/usr/bin/env python

# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup
from distutils.extension import Extension

setup(name='cos-pomdp',
      packages=['cospomdp', 'cospomdp_apps'],
      version='0.1',
      description='COS-POMDP',
      python_requires='>3.6',
      install_requires=[
          'ai2thor==3.3.4',
          'pomdp-py',
          'tqdm',
          'torch>=1.8.0',
          'prettytable',
          'pytz',
          'pandas',
          'seaborn',
          'sciex==0.3',
      ],
      author='Kaiyu Zheng',
      author_email='kaiyutony@gmail.com')
