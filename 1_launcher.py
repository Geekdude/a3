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
import tqdm
from itertools import repeat
from multiprocessing import Pool
from tabulate import tabulate
from pprint import pprint

# Import the single script.
single = importlib.import_module('0_single')


# From https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


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
        with Pool(JOBS) as p:
            for _ in tqdm.tqdm(p.imap(single.run_task, jobs), total=len(jobs)):
                pass

        print('Done')
        send_mail("Data Collection Completed", os.getcwd())

    except Exception as e:
        print('Error', file=sys.stderr)
        send_mail("Data Collection Error", f'{os.getcwd()}: {e}')
        raise e


if __name__ == '__main__':
    main(sys.argv)
