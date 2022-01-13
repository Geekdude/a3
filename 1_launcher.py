#!/usr/bin/env python3
"""
A3: https://github.com/Geekdude/a3
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
from contextlib import contextmanager

# Import the single script.
single = importlib.import_module('0_single')


# From https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


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


def run(command, verbose=True, noop=False):
    """Print command then run command"""
    return_val = ''

    if verbose:
        print(command)
    if not noop:
        try:
            return_val = subprocess.check_output(command, shell=True, stderr=subprocess.PIPE).decode()
        except subprocess.CalledProcessError as e:
            err_mesg = f'{os.getcwd()}: {e}\n\n{traceback.format_exc()}\n\n{e.returncode}\n\n{e.stdout.decode()}\n\n{e.stderr.decode()}'
            print(err_mesg, file=sys.stderr)
            with open('err.txt', 'w') as fd:
                fd.write(err_mesg)
            raise e
        except Exception as e:
            err_mesg = f'{os.getcwd()}: {e}\n\n{traceback.format_exc()}'
            print(err_mesg, file=sys.stderr)
            with open('err.txt', 'w') as fd:
                fd.write(err_mesg)
            raise e
        if verbose and return_val:
            print(return_val)

    return return_val


def single_no_stdout(job):
    with suppress_stdout():
        single.run_task(job)


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

    print('Running Jobs:')
    pprint(jobs)

    # Convert string into args
    jobs = [[''] + i.split() for i in jobs]

    # Run jobs using single.run_task
    try:
        with Pool(JOBS) as p:
            for _ in tqdm.tqdm(p.imap(single_no_stdout, jobs), total=len(jobs)):
                pass

        send_mail("Data Collection Completed", os.getcwd())
        print('Done')

    except Exception as e:
        send_mail("Data Collection Error", f'{os.getcwd()}: {e}')
        print('Error', file=sys.stderr)
        raise e


if __name__ == '__main__':
    main(sys.argv)
