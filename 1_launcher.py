#!/usr/bin/env python3
"""
Author: Aaron Young

Script to launch multiple single tasks.
"""

# Todo: Update email to send results
MAIL_TO = None

# Todo: Update number of concurent jobs to run.
JOBS = 8

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

# Import the single script.
single = importlib.import_module('0_single')


def run(command):
    """Print command then run command"""
    print(command)
    return_val = subprocess.check_output(command, shell=True).decode()
    if return_val:
        print(return_val)


def send_mail(subject, body):
    if MAIL_TO != None:
        run(f'echo "{body}" | mail -s "{subject}" {MAIL_TO}')


def main(argv):

    # Define Jobs
    # Todo: Update matrix of tasks to run.
    jobs = [
        f'-n 0 first function',
        f'-n 1 second function',
        f'-n 2 third function'
    ]

    # Convert string into args
    jobs = [[''] + i.split() for i in jobs]

    # Run jobs using single.run_task
    try:
        with Pool(8) as p:
            p.map(single.run_task, jobs)

        print('Done')
        send_mail("Data Collection Completed", os.getcwd())

    except Exception as e:
        print('Error', file=sys.stderr)
        send_mail("Data Collection Error", f'{os.getcwd()}: {e}')
        raise e


if __name__ == '__main__':
    main(sys.argv)
