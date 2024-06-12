# Copyright 2021-2024 Northeastern University
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
# Borrow from: https://github.com/navdeep-G/setup.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
import setuptools.command.build_py

from shutil import rmtree
from setuptools import find_packages, setup, Command
from setuptools.command.test import test as TestCommand


# Package meta-data.
NAME = "combinators"
DESCRIPTION = (
    "Compositional operators for the design and training of deep probabilistic programs"
)
URL = "https://github.com/probtorch/combinators"
EMAIL = "fnz@fgvgrf.vb (cipher:rot13)"
AUTHOR = "Sam Stites"
REQUIRES_PYTHON = ">=3.7.0"


def get_version():
    try:
        import subprocess

        CWD = os.path.dirname(os.path.abspath(__file__))
        rev = subprocess.check_output("git rev-parse --short HEAD".split(), cwd=CWD)
        version = "0.1.0+" + str(rev.strip().decode("utf-8"))
        return version
    except Exception:
        return "0.1.0"


VERSION = get_version()

# What packages are required for this module to be executed?
REQUIRED = [
    "probtorch",
    "flatdict",
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.org"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


def long_description():
    """
    Import the README and use it as the long-description.
    Note: this will only work if 'README.md' is present in your MANIFEST.in file!
    """
    here = os.path.abspath(os.path.dirname(__file__))

    with io.open(os.path.join(here, "README.org"), encoding="utf-8") as f:
        return "\n" + f.read()


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest

        # import sys, os
        # myPath = os.path.dirname(os.path.abspath(__file__))
        # sys.path.insert(0, myPath)
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description(),
    long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    tests_require=["pytest"],
    include_package_data=True,
    license="MIT",
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    # $ setup.py publish support.
    cmdclass={
        "test": PyTest,
        "upload": UploadCommand,
    },
)
