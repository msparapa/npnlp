from setuptools import setup
import os

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

modules = []
tests = []

dir_setup = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_setup, 'npnlp', 'release.py')) as f:
    exec(f.read())

setup(
    name='npnlp',
    version='0.0.1',
    packages=['npnlp'] + modules + tests,
    url='https://github.com/msparapa/npnlp',
    license='MIT',
    author='msparapa',
    author_email='msparapa@purdue.edu',
    description='NumPy-based nonlinear programming and optimization.',
    install_requires=requirements
)