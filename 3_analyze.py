#!/usr/bin/env python3

"""Analyze and plot results."""

import argparse
import sys
import os
import datetime
import subprocess
import time
import functools
import random
import re
from multiprocessing import Pool
from subprocess import check_output

import numpy as np
import matplotlib as mpl

DEFAULT_FIGURE_SIZE = 0.9

# Function to calculate figure size in LaTeX document
def figsize(scale=DEFAULT_FIGURE_SIZE, extra_width=0.0, extra_height=0.0):
    """Determine a good size for the figure given a scale."""
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    # Aesthetic ratio (you could change this)
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    if scale < 0.7:
        golden_mean *= 1.2
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width + extra_width, fig_height + extra_height]
    return fig_size

figcount = 1
def figsave(file, include_count=True):
    global figcount
    if not os.path.exists('figures'):
        os.makedirs('figures')
    if include_count:
        file = f'figures/{figcount:02}_{file}'
        figcount += 1
    else:
        file = f'figures/{file}'
    plt.savefig(file + '.svg')
    plt.savefig(file + '.pdf')

tblcount = 1
def tblsave(file, data, include_count=False):
    global tblcount
    if not os.path.exists('tables'):
        os.makedirs('tables')
    if include_count:
        file = f'tables/{tblcount:02}_{file}.tex'
        tblcount += 1
    else:
        file = f'tables/{file}.tex'
    with open(file, 'w') as fp:
        fp.write(data.to_latex())

# pgf settings for use in LaTeX
latex = {  # setup matplotlib to use latex for output
    "font.family": "serif",
    "axes.labelsize":  10,
    "font.size":       10,
    "legend.fontsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
mpl.rcParams.update(latex)
# print(mpl.rcParams.find_all)
import matplotlib.pyplot as plt
import math
import re
import sys
import pandas as pd
import seaborn as sns
import os
import json_tricks as json
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
pd.options.display.max_columns = None
#pd.options.display.max_rows = None

title_size = 16

# %matplotlib inline
# %matplotlib notebook

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Description""")
    args = parser.parse_args(argv[1:])

    # Todo: Add Graphing logic.

    print("Done")


if __name__ == '__main__':
    main(sys.argv)
