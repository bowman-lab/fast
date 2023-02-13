# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential


#######################################################################
# imports
#######################################################################


import gc
import glob
import mdtraj as md
import numpy as np
import os
from .base_analysis import base_analysis
from .. import tools


#######################################################################
# code
#######################################################################


class RMSDWrap(base_analysis):
    """Analysis wrapper for calculating state rmsds

    Parameters
    ----------
    base_struct : str or md.Trajectory,
        The base structure to compare for native contacts. This
        topology must match the structures to analyse. Can be provided
        as a pdb location or an md.Trajectory object.
    atom_indices : str or array,
        The atom indices to use for computing RMSD. Can be
        provided as a data file to load or an array.
    atom_indices_target : str or array,
        The atom indices to use for computing RMSD for the target structure.

    Attributes
    ----------
    output_name : str,
        The file containing rankings.
    """
    def __init__(
            self, base_struct, atom_indices=None, atom_indices_target=None):

        # determine base_struct
        self.base_struct = base_struct
        if type(base_struct) is md.Trajectory:
            self.base_struct_md = self.base_struct
        else:
            self.base_struct_md = md.load(base_struct)

        # determine atom indices
        self.atom_indices = atom_indices
        if type(atom_indices) is str:
            self.atom_indices_vals = np.loadtxt(atom_indices, dtype=int)
        else:
            self.atom_indices_vals = self.atom_indices

        # determine atom indices target
        self.atom_indices_target = atom_indices_target
        if type(atom_indices_target) is str:
            self.atom_indices_target_vals = np.loadtxt(atom_indices_target, dtype=int)
        else:
            self.atom_indices_target_vals = self.atom_indices_target

    @property
    def class_name(self):
        return "RMSDWrap"

    @property
    def config(self):
        return {
            'base_struct': self.base_struct,
            'atom_indices': self.atom_indices,
            'atom_indices_target': self.atom_indices_target,
        }

    @property
    def analysis_folder(self):
        return None

    @property
    def base_output_name(self):
        return "rmsd_per_state"

    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load(
                "./data/full_centers.xtc", top=self.base_struct_md,
                atom_indices=self.atom_indices_vals)
            # get subset if necessary
            if self.atom_indices_ref_vals is None:
                struct_sub = self.base_struct_md
            else:
                struct_sub = self.base_struct_md.atom_slice(self.atom_indices_ref_vals)
            # calculate and save rmsds
            rmsds = md.rmsd(centers, struct_sub)
            np.save(self.output_name, rmsds)
        
