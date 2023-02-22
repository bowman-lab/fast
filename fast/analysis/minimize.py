# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential


#######################################################################
# imports
#######################################################################


import glob
import itertools
import mdtraj as md
import numpy as np
import os
from .base_analysis import base_analysis
from .. import tools
from ..md_gen.gromax import Gromax
from ..submissions.os_sub import SPSub, OSWrap
from multiprocessing import Pool


#######################################################################
# code
#######################################################################


def _get_filenames(msm_dir):
    """Returns pdb filenames"""
    pdb_filenames = glob.glob(msm_dir + "/centers_masses/State*.pdb")
    pdb_filenames_full = np.array(
        [os.path.abspath(filename) for filename in np.sort(pdb_filenames)])
    return pdb_filenames_full


def _get_state_nums(pdb_filenames):
    """Determines the unique state numbers from pdb filenames"""
    state_nums = np.unique(
        np.array(
            [
                filename.split("State")[-1].split("-")[0]
                for filename in pdb_filenames]))
    return state_nums


def _minimize_energy(minimize_info):
    """multiprocessing helper. Minimizes a structure in its own
    direcory. First trjconv's it to a gro, then minimizes it with the
    specified minimize wrapper"""
    # unpack data
    minimize_obj, pdb_filename, output_folder = minimize_info
    pdb_base_name = pdb_filename.split("/")[-1].split(".pdb")[0]
    gro_output = output_folder + "/" + pdb_base_name + ".gro"
    # setup directory
    cmd0 = 'mkdir ' + output_folder
    # source gromacs file if applicable. Must add to line before
    # gromacs command
    if minimize_obj.source_file is not None:
        cmd1 = 'source ' + minimize_obj.source_file + '\n'
    else:
        cmd1 = ''
    # editconf command
    cmd1 += 'gmx editconf -f ' + pdb_filename + ' -o ' + gro_output
    cmds = [cmd0, cmd1]
    _ = tools.run_commands(cmds, supress=True)
    pid = minimize_obj.run(gro_output, output_dir=output_folder)
    return


def minimize_energies(minimize_obj, pdb_filenames, output_folder, n_cpus):
    """Minimizes a set of pdb files.

    Inputs
    ----------
    minimize_obj : object,
        Minimization wrapper.
    pdb_filenames : list,
        List of pdb filenames to minimize.
    output_folder : str,
        The folder to generate output data.
    n_cpus : int,
        The number of processes to use.
    """
    state_names = np.array(
        [filename.split("/")[-1].split("-")[0] for filename in pdb_filenames])
    output_folders = np.array(
        [output_folder + "/" + state_name for state_name in state_names])
    minimize_info = list(
        zip(
            itertools.repeat(minimize_obj), pdb_filenames, output_folders))
    pool = Pool(processes=n_cpus)
    _ = pool.map(_minimize_energy, minimize_info)
    pool.terminate()
    return


def _parse_log_for_energy(file_info):
    """Searches file for potential energy"""
    filename, _ = file_info
    f = open(filename, "r")
    f_data = f.readlines()
    f.close()
    energy = None
    for line in f_data:
        if line.split()[:3] == ['Potential', 'Energy', '=']:
            energy = float(line.split()[-1])
            break
    return energy


def parse_logs_for_energies(output_dir, n_cpus=1):
    """Searches through output directory for log files and parses them
    for potential energies."""
    # get log file names
    log_files = np.sort(glob.glob(output_dir + "/*/md.log"))
    # parallelize the parsing
    file_info = list(zip(log_files, np.arange(len(log_files))))
    pool = Pool(processes=n_cpus)
    energies = pool.map(_parse_log_for_energy, file_info)
    pool.terminate()
    return energies


class MinimizeWrap(base_analysis):
    """Analysis wrapper for minimizing structures and returning a
    potential energy.

    Parameters
    ----------
    top_file : str,
        Filename of the gromacs topology file to be used with each
        minimization.
    mdp_file : str,
        The gromacs parameter file to be used with minimization.
    n_cpus : int,
        The number of cpus to use for minimization. This is NOT per
        minimization, but the total cpus available (each minimization
        uses 1 cpu, but is parallelized).
    build_full : bool,
        Flag to either minimize all structures or to continue previous
        minimizations

    Attributes
    ----------
    msm_dir : str,
        The MSM and adaptive sampling analysis directory.
    output_folder : str,
        The directory within the msm_dir that contains minimizations.
    output_name : str,
        The filename of the final rankings.
    """
    def __init__(
            self, top_file, mdp_file, n_cpus=1, build_full=True, **kwargs):
        self.top_file = top_file
        self.mdp_file = mdp_file
        self.n_cpus = n_cpus
        self.g_obj = Gromax(
            top_file=top_file, mdp_file=mdp_file, n_cpus=1, n_gpus=None,
            submission_obj=SPSub(wait=True), min_run=True, **kwargs)
        self.build_full = build_full

    @property
    def class_name(self):
        return "MinimizeWrap"

    @property
    def config(self):
        return {
            'top_file': self.top_file,
            'mdp_file': self.mdp_file,
            'n_cpus': self.n_cpus,
            'build_full': self.build_full,
            'g_obj': self.g_obj
        }   

    @property
    def analysis_folder(self):
        return "gromax_minimize"

    @property
    def base_output_name(self):
        return "energy_per_state"

    def run(self):
        # determine if analysis was already done
        if os.path.exists(self.output_name):
            pass
        else:
            # get the pdb filenames
            pdb_filenames = _get_filenames(self.msm_dir)
            # optionally minimize all structures
            if self.build_full:
                cmd = ['mkdir ' + self.output_folder]
                _ = tools.run_commands(cmd)
                minimize_energies(
                    self.g_obj, pdb_filenames, self.output_folder,
                    self.n_cpus)
            # minimize non-processed states
            else:
                n_processed_states = len(
                    glob.glob(self.output_folder + "/State*"))
                minimize_energies(
                    self.g_obj, pdb_filenames[n_processed_states:],
                    self.output_folder, self.n_cpus)
            # parses log files for energies and saves them
            energies = parse_logs_for_energies(
                self.output_folder, n_cpus=self.n_cpus)
            np.save(self.output_name, energies)
        

