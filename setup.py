from setuptools import setup

setup(
   name='torch_ops',
   version='1.0',
   description='Backward Propagation function provided as functional calls',
   author='Chris Rae',
   author_email='crae@ed.ac.uk',
   packages=['torch_ops'],  #same as name
   install_requires=['torch'], #external packages as dependencies
)
