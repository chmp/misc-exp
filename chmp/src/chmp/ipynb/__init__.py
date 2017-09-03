"""Helper for working with jupyter notebooks.
"""

import importlib


def notebook_preamble():
    """Add common code
    """
    get_ipython().set_next_input(_notebook_preamble, replace=True)


_notebook_preamble = '''# from chmp.ipynb import notebook_preamble; notebook_preamble()

%matplotlib inline
# disable rescaling the figure, to gain tighter control over the result
%config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

import logging
logging.basicConfig(level=logging.INFO)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
'''


def reload(module_or_module_name):
    if isinstance(module_or_module_name, str):
        module_or_module_name = importlib.import_module(module_or_module_name)

    return importlib.reload(module_or_module_name)
