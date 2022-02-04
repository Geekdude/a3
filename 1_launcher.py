#!/usr/bin/env python3
"""
A3: https://github.com/Geekdude/a3
Author: Aaron Young

Script to launch multiple single tasks.
"""

# Todo: Update email to send results
_MAIL_TO = None

# Todo: Update number of concurent jobs to run.
JOBS = 1

# Todo: update output directory
OUTPUT_DIR = 'out'

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
import traceback
import multiprocessing.pool as mpp
from itertools import repeat
from multiprocessing import Pool
from tabulate import tabulate
from pprint import pprint
from contextlib import contextmanager

# Import the single script.
single = importlib.import_module('0_single')

# Locate the script.
script_path = os.path.dirname(os.path.realpath(__file__))
script = os.path.join(script_path, '0_single.py')


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

mpp.Pool.istarmap = istarmap


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


# GIT_ROOT = run('git rev-parse --show-toplevel', False).strip()


def single_start(job, args, unknown):
    global script

    # Convert string into argument array
    job_array = [''] + unknown + job.split()

    # Parse dir
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', type=str, required=True)
    a, u = p.parse_known_args(job_array)

    # Extend dir
    a.dir = os.path.join(args.dir, a.dir)
    u.insert(1, f'--dir={a.dir}')
    
    # Update Job Array
    job_array = u
    job_str = " ".join(job_array)

    # Add space to source
    if args.source:
        args.source += '; '

    # if args.ssh_remote:
    #     ssh_command = f'ssh {args.ssh_remote}'
    #     run(f'{ssh_command} bash -c "{args.source}{script}{job_str}"', verbose = not args.quiet, noop=args.noop)

    if args.slurm_remote:
        run(f'mkdir -p {a.dir}', verbose = not args.quiet, noop=args.noop)

        slurm_name = args.sname if args.sname else args.name
        slurm_command = f'srun --job-name {slurm_name} --output {a.dir}/sbuild_stdout.txt --error {a.dir}/sbuild_stderr.txt --partition {args.slurm_remote} --ntasks 1 --cpus-per-task {args.threads}'
        if args.slurm_remote_exclude:
            slurm_command += f' --exclude {args.slurm_remote_exclude}'
        if args.mem:
            slurm_command += f' --mem {args.mem}'
        run(f'{slurm_command} {args.source}{script}{job_str}', verbose = not args.quiet, noop=args.noop)

    # Local
    else:
        if not args.noop:
            if args.quiet:
                with suppress_stdout():
                    single.run_task(job_array)
            else:
                single.run_task(job_array)
        else:
            print(f'Running Task {job_array}')


def send_mail(subject, body):
    if MAIL_TO != None:
        run(f'echo "{body}" | mail -s "{subject}" {MAIL_TO}')


class Jobs():
    def __init__(self, args):
        self._job_runner = {
            'test': Jobs.test_jobs,
        }
        self.args = args

    def get_jobs(self, name):
        if name not in self._job_runner.keys():
            print(f"{name} not a valid sweep name", file=sys.stderr)
            exit(1)
        return self._job_runner[name](self)

    def test_jobs(self):
        # Todo: Update matrix of tasks to run.
        jobs = [
            f'-n 0 -d 0 first function',
            f'-n 1 -d 1 second function',
            f'-n 2 -d 2 third function'
        ]
        return jobs


def InitParser(parser):
    # Launch parameters
    parser.add_argument('-n', '--name', type=str, help='Name of tests to run', default='test')
    parser.add_argument('--noop', action='store_true', help='Print jobs, but do no run')
    parser.add_argument('--mem', type=str, help='Memory required per build (Used for slurm launch only).', default=None)
    parser.add_argument('--quiet', action='store_true', help='reduce output')
    parser.add_argument('--mail', type=str, help='Address to email the launch results to.', default=_MAIL_TO)
    parser.add_argument('-j', '--jobs', type=int, help='Number of jobs to run in parallel', default=JOBS)
    parser.add_argument('--source', type=str, help='Source needed to run 0_single.py', default='')

    # Duplicate parameters
    parser.add_argument('-d', '--dir', type=str, default=OUTPUT_DIR, help='Output directory for the run, This overwrites --output.')
    parser.add_argument('-t', '--threads', type=int, help='Number of threads to use per job.', default=1)

    # Remote parameters
    remote_run = parser.add_mutually_exclusive_group()
    # remote_run.add_argument('--ssh_remote', type=str, metavar='connection string', help='Run remotely using ssh.', default=None)
    remote_run.add_argument('--slurm_remote', type=str, metavar='partition', help='Run remotely using slurm.', default=None)
    parser.add_argument('--slurm_remote_exclude', type=str, help='Do no run on these slurm nodes.', default=None)
    parser.add_argument('--sname', type=str, help='Override default slurm job name', default=None)


_quiet = False
def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Launch single jobs""")
    InitParser(parser)
    args, unknown = parser.parse_known_args(argv[1:])

    # Add back in duplicate parameters.
    unknown.extend(['--threads', f'{args.threads}'])
    if args.noop:
        unknown.extend(['--noop'])

    # Override Mailto
    global MAIL_TO
    MAIL_TO = args.mail

    # Override Quite
    global _quiet
    _quiet = args.quiet

    # Define Jobs
    jobs = Jobs(args).get_jobs(args.name)

    print(f'Running {len(jobs)} Jobs:')
    pprint(jobs)
    print()

    # Create output directory
    run(f'mkdir -p {args.dir}', verbose = not args.quiet, noop=args.noop)

    # Run jobs using single.run_task
    try:
        with Pool(args.jobs) as p:
            for _ in tqdm.tqdm(p.istarmap(single_start, zip(jobs, repeat(args, len(jobs)), repeat(unknown, len(jobs)) )), total=len(jobs)):
                pass

        send_mail("Data Collection Completed", f'{os.getcwd()} {" ".join(argv)}')
        print('Done')

    except Exception as e:
        send_mail("Data Collection Error", f'{os.getcwd()} {" ".join(argv)}: {e}\n\n{traceback.format_exc()}')
        print(f'Error {e}', file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv)
