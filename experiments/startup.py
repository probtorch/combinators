#!/usr/bin/env python3

import sys
import subprocess
gitroot = subprocess.check_output('git rev-parse --show-toplevel', shell=True).decode("utf-8").rstrip()
sys.path.append(gitroot)

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.info(gitroot + " appended to python path")

from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext autoreload"); logging.info("%load_ext autoreload")
ipython.magic("autoreload 2"); logging.info("%autoreload 2")

from IPython.core.debugger import set_trace; logging.info("from IPython.core.debugger import set_trace")
from IPython.core.display import display, HTML; logging.info("from IPython.core.display import display, HTML")

try:
    import torch      ; logging.info("import torch")
    import numpy as np; logging.info("import numpy as np")
    import scipy as sp; logging.info("import scipy as sp")
except:
    logging.debug("expected science imports failed")

try:
    import matplotlib;                  logging.info("import matplotlib")
    import matplotlib.pyplot as plt;    logging.info("import matplotlib.pyplot as plt")
    ipython.magic("matplotlib inline"); logging.info("%matplotlib inline")
    # ipython.magic("config InlineBackend.figure_format = 'retina'"); logging.info("%config InlineBackend.figure_format = 'retina'")
except:
    logging.debug("matplotlib import failed")

try:
    import seaborn as sns; logging.info("import seaborn as sns")
    sns.set_context("poster")
    sns.set(rc={'figure.figsize': (16, 9.)})
    sns.set_style("whitegrid")
except:
    logging.debug("seaborn import failed")

try:
    import pandas as pd; logging.info("import pandas as pd")
    pd.set_option("display.max_rows", 120)
    pd.set_option("display.max_columns", 120)
except:
    logging.debug("pandas import failed")
