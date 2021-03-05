#!/usr/bin/env python3

import os
from combinators.utils import git_root

def datapaths(data_dir=f'{git_root()}/data/bshape/', subfolder=''):
    return [os.path.join(data_dir, subfolder, f) for f in os.listdir(data_dir)]
