# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential

#######################################################################
# imports
#######################################################################

import mdtraj as md
import numpy as np
import os
import mdtraj_upside as mu
from .. import tools
from .. import submissions
from ..base import base

#######################################################################
# code
#######################################################################


class UpsideProcessing(base):
    """Generates gromacs commands for aligning a trajectory and
    determining output coordinates."""
    def __init__(self, align=True):
        self.align = align

    @property
    def class_name(self):
        return "UpsideProcessing"

    @property
    def config(self):
        return {'align': self.align}

    def run(self, input_file, output_file):
        process_cmd = "/home/mizimmer/programs/fast/md_gen/process_upside.py " + \
            " --input_file " + input_file + \
            " --output_file " + output_file
        if self.align:
            process_cmd += " --align"
        return process_cmd + "\n"


class Upside(base):
    """Upside wrapper for running simulations

    Parameters
    ----------
    fasta_file : str,
        The fasta filename.
    output_name : str, default = "simulation.up",
        The output name for upside simulations.
    output_basename : str, default = "upside_sim",
        The base name for output files from upside.
    upside_dir : str, default = None,
        The directory containing upside install. Currently defaults
        to Max's install.
    processing_obj : object, default = None,
        Object that when run will output commands for processing
        trajectory. Look at GromaxProcessing.
    submission_obj : object,
        Submission object used for running the simulation. Look into
        SlurmSub or OSSub.
    """
    def __init__(
            self, fasta_file, output_name="simulation.up",
            output_basename='upside_sim', upside_dir=None, processing_obj=None,
            submission_obj=None, duration='1e7', frame_interval='1e2', temperature=0.5):
        self.fasta_file = os.path.abspath(fasta_file)
        self.output_name = output_name
        self.output_basename = output_basename
        if upside_dir is None:
            upside_dir = "/home/mizimmer/programs/upside-md/"
        self.upside_py_dir = os.path.abspath(upside_dir) + "/py"
        self.upside_param_dir = os.path.abspath(upside_dir) + "/parameters"
        self.upside_obj = os.path.abspath(upside_dir) + "/obj/upside"
        self.duration = str(duration)
        self.frame_interval = str(frame_interval)
        self.temperature = str(temperature)
        self.processing_obj = processing_obj
        self.submission_obj = submission_obj

    @property
    def class_name(self):
        return "Upside"

    @property
    def config(self):
        return {
            'fasta_file': self.fasta_file,
            'output_name': self.output_name,
            'upside_py_dir': self.upside_py_dir,
            'upside_param_dir': self.upside_param_dir
        }

    def setup_run(self, struct, output_dir=None):
        # set output directory
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = "./"
        self.output_dir = os.path.abspath(self.output_dir)
        # generate directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            tools.run_commands('mkdir ' + self.output_dir)
        # determine starting structure filename
        if isinstance(struct, str):
            if struct[-3:] == 'gro':
                struct = md.load(struct)
        if type(struct) is md.Trajectory:
            struct.save_pdb(self.output_dir + '/start.pdb')
            self.start_name = self.output_dir + '/start.pdb'
        elif (struct is None) or (struct == 'None'):
            self.start_name = None
        else:
            self.start_name = os.path.abspath(struct)
        return

    def run(self, struct, output_dir=None):
        # setup_run
        self.setup_run(struct=struct, output_dir=output_dir)
        # if starting structure is available, process it
        upside_config_cmd = self.upside_py_dir + "/upside_config.py" \
            " --output " + self.output_name + \
            " --fasta " + self.fasta_file + \
            " --hbond-energy $(cat " + self.upside_param_dir + "/ff_1/hbond)" + \
            " --dynamic-rotamer-1body " + \
            " --rotamer-placement " + self.upside_param_dir + "/ff_1/sidechain.h5" + \
            " --rotamer-interaction " + self.upside_param_dir + "/ff_1/sidechain.h5" + \
            " --environment " + self.upside_param_dir + "/ff_1/environment.h5" + \
            " --rama-library " + self.upside_param_dir + "/common/rama.dat" + \
            " --rama-sheet-mixing-energy $(cat " + self.upside_param_dir + "/ff_1/sheet)" + \
            " --reference-state-rama " + self.upside_param_dir + "/common/rama_reference.pkl"
        # generate mdrun command
        run_cmd = self.upside_obj + \
            " --duration " + self.duration + \
            " --frame-interval " + self.frame_interval + \
            " --temperature " + self.temperature + \
            " --seed $RANDOM " + \
            self.output_name + "\n"
        if self.start_name is not None:
            pdb_process_cmd = self.upside_py_dir + \
                "/PDB_to_initial_structure.py " + self.start_name + " " + \
                self.output_basename + "\n"
            upside_config_cmd += " --initial-structure " + self.output_basename + ".initial.pkl\n"
            cmds = [pdb_process_cmd, upside_config_cmd, run_cmd]
        else:
            cmds = [upside_config_cmd+"\n", run_cmd]
        # combine commands and submit to submission object
        try:
            cmds.append(self.processing_obj.run(self.output_name, "frame0_aligned.xtc"))
            cmds.append(self.processing_obj.run(self.output_name, "frame0_masses.xtc"))
        except:
            pass
#        print(cmds)
        job_id = self.submission_obj.run(cmds, output_dir=output_dir)
        return job_id
        

