#!/usr/bin/env python3

import importlib
import argparse
import datastore
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import subprocess
import sys
import timeit
from itertools import repeat
from multiprocessing import Pool
from tabulate import tabulate
from pprint import pprint

# Todo: update output directory
OUTPUT_DIR = 'out'

def run(command, verbose=True):
    """Print command then run command"""

    if verbose:
        print(command)
    return_val = subprocess.check_output(command, shell=True).decode()
    if verbose and return_val:
        print(return_val)


def execute(output_file, *args, **kwargs):
    """Execute the task."""
    # Todo: This function is overwritten with the desired behavior.

    print(f"Running Task args:{args} kwargs:{kwargs}")
    with open(output_file, 'w') as fd:
        fd.write(f"Running Task args:{args} kwargs:{kwargs}")


def run_task(argv):
    """Parse the arguments and call execute."""
    # Parse the arguments
    # Todo: Change the parser to expose the single task variables.
    parser = argparse.ArgumentParser(description="""Description""")
    parser.add_argument('-n', '--number', help='integer value', type=int, default=0)
    parser.add_argument('positional', metavar='p', type=str, nargs='*')
    args = parser.parse_args(argv[1:])
    
    # Todo: Change to expected output file.
    run(f'mkdir -p {OUTPUT_DIR}', False)
    output_file = f"{OUTPUT_DIR}/run_{args.number}.txt"

    datastore.execute_if_missing(output_file, execute, output_file, *args.positional, **vars(args))


if __name__ == '__main__':
    run_task(sys.argv)