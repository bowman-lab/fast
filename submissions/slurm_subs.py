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
from ..exceptions import UnexpectedResult

#######################################################################
# code
#######################################################################


def _make_sbatch_lines(kwargs):
    """makes keyword lines in an sbatch script from kwargs"""
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    additions = "\n".join(
        [
            '#SBATCH --' + i[0] + '=' + i[1] \
            for i in np.transpose([keys, values])])
    return additions


def _gen_header(
        queue, n_tasks, n_cpus, exclusive, email, max_time, job_name, kwargs):
    """Generates an sbatch header file"""
    header = '#!/bin/bash\n\n'
    header += '# specify resources\n' + \
        '#SBATCH --ntasks=' + n_tasks + '\n'\
        '#SBATCH --cpus-per-task=' + n_cpus + '\n'
    if exclusive:
        header += '#SBATCH --exclusive\n'
    header += '\n# max wallclock time\n' + \
        '#SBATCH --time=' + str(max_time) + ':00:00\n'
    header += '\n# jobname\n' + \
        '#SBATCH --job-name=' + job_name + '\n'
    header += '\n# queue\n' + \
        '#SBATCH --partition=' + queue + '\n'
    if email is not None:
        header += '\n# mail alert' + \
            '#SBATCH --mail-type=ALL\n' + \
            '#SBATCH --mail-user=' + email + '\n'
    additions = _make_sbatch_lines(kwargs)
    header += '\n# additional specs\n'
    header += additions + '\n'
    header += '\n'
    return header


def get_running_jobs():
    """Finds jobs that are currently running"""
    try:
        squeue_output = tools.run_commands('squeue')[0]
        job_listing_information = squeue_output.split("\n")[:-1]
        running_jobs = ra.RaggedArray(
            [
                s.split() for s in 
                job_listing_information])[:,0]
        if running_jobs[0] != 'JOBID':
            raise UnexpectedResult(
                'slurm queue wrapper failed to parse jobs!')
        else:
            running_jobs = running_jobs[1:]
    except:
        logger.log("an error has occured with finding jobs...")
        logger.log("for error checking purposes: ")
        logger.log(squeue_output)
        logger.log(job_listing_information)
        logger.log(running_jobs)
        raise UnexpectedResult(
            'slurm queue wrapper failed to parse jobs!')
    return np.array(running_jobs)


class SlurmWrap(base):
    """Wrapper for slurm checking and waiting for jobs

    Parameters
    ----------
    max_n_procs : int, default = np.inf,
        The maximum number of jobs to be running at a time.
    """
    def __init__(self, max_n_procs=np.inf):
        self.max_n_procs = max_n_procs

    @property
    def class_name(self):
        return "SlurmWrap"

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
        names = ['slurm-' + str(pid) + '.out' for pid in pids]
        return names


class SlurmSub(base):
    """Slurm submission wrapper.

    Parameters
    ----------
    queue : str,
        The queue to submit.
    n_tasks : int, default=1,
        The number of tasks for the submission job.
    n_cpus : int, default = 1,
        Number of cpus to use.
    exclusive : bool, default = False,
        To request exclusive use of a node.
    email : str, default = None,
        Email address to optionally email updates.
    max_time : int, default = 1500,
        The maximum time for submission job in hours.
    job_name : str, default = None,
        The name of the submission job.
    """
    def __init__(
            self, queue, n_tasks=1, n_cpus=1, exclusive=False, email=None,
            max_time=1500, job_name=None, **kwargs):
        self.queue = str(queue)
        self.n_tasks = str(n_tasks)
        self.n_cpus = str(n_cpus)
        self.exclusive = exclusive
        self.email = email
        self.max_time = str(max_time)
        if job_name is None:
            self.job_name = 'SlurmSub'
        else:
            self.job_name = str(job_name)
        self.kwargs = kwargs

    @property
    def class_name(self):
        return "SlurmSub"

    @property
    def config(self):
        config_dict = {
            'queue': self.queue,
            'n_tasks': self.n_tasks,
            'n_cpus': self.n_cpus,
            'exclusive': self.exclusive,
            'email': self.email,
            'max_time': self.max_time,
            'job_name': self.job_name}
        config_dict.update(self.kwargs)
        return config_dict

    def run(self, cmds, output_dir=None, output_name=None):
        # generate header file
        header = _gen_header(
            self.queue, self.n_tasks, self.n_cpus, self.exclusive, self.email,
            self.max_time, self.job_name, self.kwargs)
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
            output_name = 'slurm_submission'
        os.chdir(output_dir)
        # write submission file
        f = open(output_name, 'w')
        f.write(sub_file)
        f.close()
        # run submission file
        job_sub = tools.run_commands('sbatch ' + output_name)[0].split()[-1]
        os.chdir(home_dir)
        return job_sub
        
