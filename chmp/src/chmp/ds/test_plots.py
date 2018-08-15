import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from chmp.ds import mpl_set


def test_mpl_set_xscale():
    mpl_set(xscale="log")
    assert plt.gca().get_xscale() == "log"


def test_mpl_set_yscale():
    mpl_set(yscale="log")
    assert plt.gca().get_yscale() == "log"
