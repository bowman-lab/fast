# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential

#######################################################################
# imports
#######################################################################

import numpy as np
import os
import subprocess as sp
import time
from enspara.util import array as ra
from .. import tools
from ..base import base
from ..exception import UnexpectedResult

#######################################################################
# code
#######################################################################


def _make_bsub_lines(kwargs):
    """makes keyword lines in an sbatch script from kwargs"""
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    additions = "\n".join(
        [
            '#BSUB -' + i[0] + ' ' + i[1] \
            for i in np.transpose([keys, values])])
    return additions


def _gen_header(
        queue, n_tasks, max_time, job_name, kwargs):
    """Generates an sbatch header file"""
    header = '#!/bin/bash\n\n'
    header += '# specify resources\n' + \
        '#BSUB -n %d\n' % n_tasks
    header += '\n# max wallclock time\n' + \
        '#BSUB -ptl %d:00\n' % max_time
    header += '\n# queue\n' + \
        '#BSUB -q %s\n' % queue
    header += '\n# name and output\n' + \
        '#BSUB -J %s\n' % job_name
    header += '#BSUB -o lsf_output-%J.log\n'
    header += '#BSUB -e lsf_output-%J.log\n'
    additions = _make_bsub_lines(kwargs)
    header += '\n# additional specs\n'
    header += additions + '\n'
    header += '\n'
    return header


def get_running_jobs():
    """Finds jobs that are currently running"""
    try:
        bjobs_output = tools.run_commands('bjobs', supress=True)[0]
        if bjobs_output == '':
            running_jobs = ['']
        else:
            job_listing_information = bjobs_output.split("\n")[:-1]
            running_jobs = ra.RaggedArray(
                [
                    s.split() for s in 
                    job_listing_information])[:,0]
            if running_jobs[0] != 'JOBID':
                raise UnexpectedResult(
                    'LSF queue wrapper failed to parse jobs!')
            else:
                running_jobs = running_jobs[1:]
    except:
        logger.log("an error has occured with finding jobs...")
        logger.log("for error checking purposes: ")
        logger.log(bjobs_output)
        logger.log(job_listing_information)
        logger.log(running_jobs)
        raise UnexpectedResult(
            'lsf queue wrapper failed to parse jobs!')
    return np.array(running_jobs)


class LSFWrap(base):
    """Wrapper for LSF checking and waiting for jobs

    Parameters
    ----------
    max_n_procs : int, default = np.inf,
        The maximum number of jobs to be running at a time.
    """
    def __init__(self, max_n_procs=np.inf):
        self.max_n_procs = max_n_procs

    @property
    def class_name(self):
        return "LSFWrap"

    @property
    def config(self):
        return {
            'max_n_procs': self.max_n_procs
        }

    def wait_for_pids(self, pids, wait_time=2, wait_for_all=False):
        # if waiting for all, the maximum number of procs running
        # should be zero
        if wait_for_all:
            max_n_procs = 0
        else:
            max_n_procs = self.max_n_procs
        wait = True
        # while waiting, check is pids are still running
        while wait:
            running_jobs = get_running_jobs()
            wait = False
            n_running_jobs = 0
            for pid in pids:
                # if job is still running, add 1 to n_running_jobs
                if len(np.where(running_jobs == pid)[0]) > 0:
                    n_running_jobs += 1
                # if running jobs exceeds the maximum allowed,
                # wait longer
                if n_running_jobs > max_n_procs:
                    wait = True
                    time.sleep(wait_time)
                    break
        return

    def get_submission_names(self, pids):
        """Returns the submission file name"""
        if type(pids) is str:
            pids = [pids]
        names = ['lsf_output-%d.log' % int(pid) for pid in pids]
        return names


class LSFSub(base):
    """LSF submission wrapper.

    Parameters
    ----------
    queue : str,
        The queue to submit.
    n_tasks : int, default=1,
        The number of tasks for the submission job.
    max_time : int, default = 1500,
        The maximum time for submission job in hours.
    job_name : str, default = None,
        The name of the submission job.
    """
    def __init__(
            self, queue, n_tasks=1, max_time=1500, job_name=None, **kwargs):
        self.queue = str(queue)
        self.n_tasks = n_tasks
        self.max_time = max_time
        if job_name is None:
            self.job_name = 'LSF_Sub'
        else:
            self.job_name = str(job_name)
        self.kwargs = kwargs

    @property
    def class_name(self):
        return "LSFSub"

    @property
    def config(self):
        config_dict = {
            'queue': self.queue,
            'n_tasks': self.n_tasks,
            'max_time': self.max_time,
            'job_name': self.job_name}
        config_dict.update(self.kwargs)
        return config_dict

    def run(self, cmds, output_dir=None, output_name=None):
        # generate header file
        header = _gen_header(
            self.queue, self.n_tasks, self.max_time, self.job_name,
            self.kwargs)
        # add commands
        if type(cmds) is str:
            sub_file = header + cmds
        else:
            sub_file = header
            for cmd in cmds:
                sub_file += cmd
        # catalog home dir and switch to output dir
        home_dir = os.path.abspath("./")
        if output_dir is None:
            output_dir = os.path.abspath("./")
        if output_name is None:
            output_name = 'lsf_submission'
        os.chdir(output_dir)
        # write submission file
        f = open(output_name, 'w')
        f.write(sub_file)
        f.close()
        # run submission file
        job_sub = tools.run_commands('bsub < ' + output_name)[0]
        job_id = job_sub.split()[1].split("<")[-1].split(">")[0]
        os.chdir(home_dir)
        return job_id
        
