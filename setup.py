import io
import os
from setuptools import find_packages, setup, Command
import setuptools.command.build_py

import sys
from setuptools.command.test import test as TestCommand


def get_version():
    try:
        import subprocess
        CWD = os.path.dirname(os.path.abspath(__file__))
        rev = subprocess.check_output("git rev-parse --short HEAD".split(), cwd=CWD)
        version = "0.0+" + str(rev.strip().decode('utf-8'))
        return version
    except Exception:
        return "0.0"


def long_description():
    here = os.path.abspath(os.path.dirname(__file__))

    # Import the README and use it as the long-description.
    # Note: this will only work if 'README.md' is present in your MANIFEST.in file!
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return '\n' + f.read()


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        # import sys, os
        # myPath = os.path.dirname(os.path.abspath(__file__))
        # sys.path.insert(0, myPath)
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='combinators',
    version=get_version(),
    description= \
      'Compositional operators for the design and' + \
      'training of deep probabilistic programs',
    long_description=long_description(),
    url='https://github.com/probtorch/combinators',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'probtorch',
        'flatdict',
    ],
    tests_require=['pytest'],
    cmdclass = {'test': PyTest},
    include_package_data=True,
    license='MIT',
)
