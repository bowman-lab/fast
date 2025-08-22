# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential

# Rewritten by: Gavin N. John <gavinnjohn@gmail.com>
# Copyright (C) 2025, University of Pennsylvania AND California Institute of Technology (Joint)
# Licensed under the terms of the GNU Lesser General Public License, version 2.1 or later

#######################################################################
# imports
#######################################################################


import mdtraj as md
import numpy as np
import os
from .. import tools
from .. import submissions
from ..base import base


#######################################################################
# code
#######################################################################


class GromacsProcessing(base):
    """Generates gromacs commands for aligning a trajectory and
    determining output coordinates."""

    def __init__(
        self,
        *,
        align_group=None,
        output_group=None,
        center_group=None,
        pbc="mol",
        ur="compact",
        index_file=None,
    ):
        self.align_group = str(align_group)
        self.output_group = str(output_group)
        self.center_group = str(center_group)
        self.pbc = pbc
        self.ur = ur
        self.index_file = os.path.abspath(index_file) if index_file else None

    @property
    def class_name(self):
        return "GromacsProcessing"

    @property
    def config(self):
        return {
            "align_group": self.align_group,
            "output_group": self.output_group,
            "pbc": self.pbc,
            "ur": self.ur,
        }

    def build_trjconv_command(
        self, input_xtc, output_xtc, groups, tpr="md.tpr", tag="align"
    ):
        base_cmd = f"gmx trjconv -f {input_xtc} -o {output_xtc} -s {tpr} -center -pbc {self.pbc} -ur {self.ur}"
        if self.index_file:
            base_cmd += f" -n {self.index_file}"
        return f"echo '{groups}' | {base_cmd}"

    def run(self):
        align_groups, output_groups = self._group_params_by_pbc()
        cmd_align = self.build_trjconv_command(
            "frame0.xtc", "frame0_aligned.xtc", align_groups
        )
        cmd_output = self.build_trjconv_command(
            "frame0.xtc", "frame0_masses.xtc", output_groups
        )
        return f"{cmd_align}\n{cmd_output}"

    def _group_params_by_pbc(self):
        match self.pbc:
            case "cluster":
                return (
                    f"{self.align_group} {self.center_group} 0",
                    f"{self.align_group} {self.center_group} {self.output_group}",
                )
            case "nojump":
                return (
                    f"{self.center_group} 0",
                    f"{self.center_group} {self.output_group}",
                )
            case "mol":
                return (
                    f"{self.align_group} 0",
                    f"{self.align_group} {self.output_group}",
                )
            case _:
                raise ValueError(f"Unknown pbc: {self.pbc}")

class Gromacs(base):
    """Gromacs wrapper for running md simulations or minimizing
    structures

    Parameters
    ----------
    top_file : str,
        Gromacs topology filename.
    mdp_file : str,
        Gromacs mdp file that specifies simulation parameters.
    n_cpus : int, default = 1,
        The number of cpus to use with the simulation.
    n_gpus : int, default = None,
        The number of gpus to use with the simulation. If None, will
        only use cpus.
    processing_obj : object, default = None,
        Object that when run will output commands for processing
        trajectory. Look at GromacsProcessing.
    index_file : str, default = None,
        Optionally supply an index file.
    itp_files : list, default = None,
        Optionally supply a list of itp files that go along with the
        topology file.
    submission_obj : object,
        Submission object used for running the simulation. Look into
        SlurmSub or OSSub.
    max_warn : int, default = 2,
        Maximum number of gromacs warnings to allow.
    min_run : bool, default = False,
        Is this a minimization run? Helps with output naming.
    setup_path : str, default = None,
        The path to a file to source before running Gromacs.
        For example, if using singularity, it should alias gmx
        to `singularity run ...`
        As another example, if using Gromacs normally with GPU
        acceleration but with an unknown GPU type, this script
        can detect the GPU type and load the correct module and
        GMXRC.
    """
    def __init__(
        self,
        *,
        top_file,
        mdp_file,
        n_cpus=1,
        n_gpus=None,
        processing_obj,
        index_file=None,
        itp_files=None,
        submission_obj = None,
        max_warn=2,
        min_run=False,
        setup_path=None,
        **kwargs,
    ):
        self.top_file = os.path.abspath(top_file)
        self.mdp_file = os.path.abspath(mdp_file)
        self.n_cpus = n_cpus
        self.n_gpus = n_gpus
        self.index_file = os.path.abspath(index_file) if index_file else None
        self.processing_obj = processing_obj
        self.submission_obj = submission_obj if submission_obj else GromacsProcessing(
            index_file=index_file,
            **kwargs
        )
        self.itp_files = (
            np.array([os.path.abspath(f) for f in itp_files]) if itp_files else None
        )
        self.max_warn = str(max_warn)
        self.min_run = min_run
        self.setup_path = os.path.abspath(setup_path) if setup_path else None
        self.kwargs = kwargs

    @property
    def class_name(self):
        return "Gromacs"

    @property
    def config(self):
        return {
            "top_file": self.top_file,
            "mdp_file": self.mdp_file,
            "n_cpus": self.n_cpus,
            "n_gpus": self.n_gpus,
            "processing_obj": self.processing_obj,
            "index_file": self.index_file,
            "itp_files": self.itp_files,
            "submission_obj": self.submission_obj,
            "max_warn": self.max_warn,
            "min_run": self.min_run,
            "setup_path": self.setup_path,
        }

    def setup_run(self, struct, output_dir=None):
        self.output_dir = os.path.abspath(output_dir or "./")
        os.makedirs(self.output_dir, exist_ok=True)

        if isinstance(struct, md.Trajectory):
            struct.save_gro(os.path.join(self.output_dir, "start.gro"))
            self.start_name = os.path.join(self.output_dir, "start.gro")
        else:
            self.start_name = os.path.abspath(struct)

        if self.itp_files is not None:
            tools.run_commands(
                [f"cp {file} {self.output_dir} -r" for file in self.itp_files]
            )

    def generate_grompp_cmd(self, base_output):
        cmd = f"gmx grompp -f {self.mdp_file} -c {self.start_name} -p {self.top_file} -o {base_output} -maxwarn {self.max_warn}"
        if self.index_file:
            cmd += f" -n {self.index_file}"
        return cmd

    def generate_mdrun_cmd(self, base_output):
        cmd = (
            f"gmx mdrun -cpi state -g md -s {base_output} -o {base_output} "
            f"-c after_{base_output} -v -nt {self.n_cpus} "
            "-nb gpu -bonded gpu -pme gpu"
        )
        if not self.min_run:
            cmd += " -x frame0"
        if self.n_gpus:
            if self.n_cpus % self.n_gpus != 0:
                raise ValueError("CPU count must be divisible by GPU count")
            cmd += f" -ntmpi {self.n_gpus} -ntomp {self.n_cpus // self.n_gpus}"
        cmd += " " + " ".join([f"-{k} {v}" for k, v in self.kwargs.items()])
        return cmd
    
    def run(self, struct, output_dir=None, check_continue=True):
        self.setup_run(struct, output_dir)
        base_output = "em" if self.min_run else "md"
        grompp_cmd = self.generate_grompp_cmd(base_output)
        mdrun_cmd = self.generate_mdrun_cmd(base_output)

        if self.setup_path:
            setup_cmd = f"source {self.setup_path}\n"
        else:
            setup_cmd = ""

        if check_continue:
            tpr_check = os.path.join(self.output_dir, "md.tpr")
            grompp_cmd = (
                f'if [ ! -f "{tpr_check}" ]; then\n'
                '    echo "Didn\'t find md.tpr, running grompp..."\n'
                f"    {grompp_cmd}\n"
                'else\n    echo "Found md.tpr, skipping grompp"\nfi\n'
            )

        cmds = [
            setup_cmd,
            "\n\n",
            grompp_cmd,
            "\n\n",
            mdrun_cmd,
            "\n\n",
            self.processing_obj.run()
        ]
        return self.submission_obj.run(cmds, output_dir=self.output_dir)
