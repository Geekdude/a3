#!/usr/bin/env python3

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

# Todo: update results filename
RESULTS_FILE = 'results.txt'

def main(argv):
    # Read data from output directory
    data = datastore.read_all_data_from_folder(OUTPUT_DIR)
    
    # Todo: add any data processing
    data = [item + '\n' for sublist in data for item in sublist]

    # Write out results.
    with open(RESULTS_FILE, 'w') as fd:
        fd.writelines(data)

    print('Done')


if __name__ == '__main__':
    main(sys.argv)