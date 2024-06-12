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
#!/usr/bin/env python3
"""
This file is intended to be a verbose "bootstrap" script used in conjunction with a jupyter notebook.
From <git root>/experiments/annealing/notebooks/my-notebook.ipynb, invoke:

    %run ../../startup.py

And the following commands will run (verbosely).
"""

import sys
import subprocess

gitroot = (
    subprocess.check_output("git rev-parse --show-toplevel", shell=True)
    .decode("utf-8")
    .rstrip()
)
sys.path.append(gitroot)

import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.info(gitroot + " appended to python path")

from IPython import get_ipython

ipython = get_ipython()
ipython.magic("load_ext autoreload")
logging.info("%load_ext autoreload")
ipython.magic("autoreload 2")
logging.info("%autoreload 2")

from IPython.core.debugger import set_trace

logging.info("from IPython.core.debugger import set_trace")
from IPython.core.display import display, HTML

logging.info("from IPython.core.display import display, HTML")

try:
    import torch

    logging.info("import torch")
    import numpy as np

    logging.info("import numpy as np")
    import scipy as sp

    logging.info("import scipy as sp")
except:
    logging.debug("expected science imports failed")

try:
    import matplotlib

    logging.info("import matplotlib")
    import matplotlib.pyplot as plt

    logging.info("import matplotlib.pyplot as plt")
    ipython.magic("matplotlib inline")
    logging.info("%matplotlib inline")
    # ipython.magic("config InlineBackend.figure_format = 'retina'"); logging.info("%config InlineBackend.figure_format = 'retina'")
except:
    logging.debug("matplotlib import failed")

try:
    import seaborn as sns

    logging.info("import seaborn as sns")
    sns.set_context("poster")
    sns.set(rc={"figure.figsize": (16, 9.0)})
    sns.set_style("whitegrid")
except:
    logging.debug("seaborn import failed")

try:
    import pandas as pd

    logging.info("import pandas as pd")
    pd.set_option("display.max_rows", 120)
    pd.set_option("display.max_columns", 120)
except:
    logging.debug("pandas import failed")
