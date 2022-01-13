#!/usr/bin/env python3
"""
A3: https://github.com/Geekdude/a3
Author: Aaron Young

Script to launch a single task.
"""

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
import time
import traceback
from itertools import repeat
from multiprocessing import Pool
from tabulate import tabulate
from pprint import pprint

# Todo: update output directory
OUTPUT_DIR = 'out'
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


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


def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""
    import subprocess, os
    pipe = subprocess.Popen("bash -c 'source %s > /dev/null; env'" % script, stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0].decode()
    env = {}
    for line in output.splitlines():
        try:
            key, val = line.split("=", 1)
        except ValueError as e:
            pass
        env[key] = val
    os.environ.update(env)


def execute(output_file, *args, **kwargs):
    """Execute the task."""
    # Todo: This function is overwritten with the desired behavior.

    print(f"Running Task args:{args} kwargs:{kwargs}")
    with open(output_file, 'w') as fd:
        fd.write(f"Running Task args:{args} kwargs:{kwargs}")

def InitParser(parser):
    # Todo: Change the parser to expose the single task variables.
    parser.add_argument('-n', '--number', help='integer value', type=int, default=0)
    parser.add_argument('-d', '--dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('positional', metavar='p', type=str, nargs='*')

def run_task(argv):
    """Parse the arguments and call execute."""
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Description""")
    InitParser(parser)
    args = parser.parse_args(argv[1:])

    # Todo: Change to expected output file.
    run(f'mkdir -p {args.dir}', False)
    output_file = f"{args.dir}/run_{args.number}.txt"

    datastore.execute_if_missing(output_file, execute, output_file, *args.positional, **vars(args))


if __name__ == '__main__':
    run_task(sys.argv)

