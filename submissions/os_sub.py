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
from .. import tools
from ..base import base


#######################################################################
# code
#######################################################################


def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        running = False
    else:
        running = True
    return running


class OSWrap(base):
    """Wrapper for the linux operating system."""
    def __init__(self, max_n_procs=np.inf):
        self.max_n_procs = max_n_procs

    @property
    def class_name(self):
        return "OSWrap"

    @property
    def config(self):
        return {
            'max_n_procs': self.max_n_procs
        }

    def get_submission_names(self, pid=None):
        return ['os_output.txt']

    def wait_for_pids(self, pids, wait_time=2, wait_for_all=False):
        # if waiting for all, the maximum number of procs running
        # should be zero
        if wait_for_all:
            max_n_procs = 0
        else:
            max_n_procs = self.max_n_procs
        wait = True
        # while waiting, check if pids are still running
        while wait:
            wait = False
            n_running_jobs = 0
            for pid in pids:
                # if job is still running, add 1 to n_running_jobs
                if check_pid(pid):
                    n_running_jobs += 1
                # if running jobs exceeds the maximum allows,
                # wait longer
                if n_running_jobs > max_n_procs:
                    wait = True
                    time.sleep(wait_time)
                    break
        return


class SPSub(base):
    """Submission wrapper using subprocessing.

    Parameters
    ----------
    wait : bool, default = False,
        Optionally submit and wait for each job. When False, will
        submit a job and return while it is still running.
    """
    def __init__(
            self, wait=False, **kwargs):
        self.wait = wait
        self.kwargs = kwargs

    @property
    def class_name(self):
        return "SPSub"

    @property
    def config(self):
        return {
            'wait': self.wait
        }

    def run(self, cmds, output_dir=None, output_name=None):
        cmds = np.array(cmds).reshape(-1)
        # set home and output dir
        home_dir = os.path.abspath("./")
        if output_dir is None:
            output_dir = os.path.abspath("./")
        os.chdir(output_dir)
        if  output_name is None:
            output_name = 'os_submission'
        f = open(output_name, 'w')
        f.write("\n".join(cmds))
        f.close()
        # submit
        p = sp.Popen(
            "bash %s &> os_output.txt" % output_name, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        # optionally wait for job to finish
        if self.wait:
            out,err = p.communicate()
            print(out.decode('utf-8'))
            print(err.decode('utf-8'))
            job_sub = p.pid
        else:
            job_sub = p.pid
        # return home
        os.chdir(home_dir)
        return job_sub

