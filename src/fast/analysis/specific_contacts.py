# Author: Artur Meller <ameller@wustl.edu>, Maxwell Zimmerman <mizimmer@wustl.edu>
# Modified from code in contacts.py
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


#######################################################################
# code
#######################################################################


def best_hummer_q(traj, native, atom_pairs):
    """Compute the fraction of contacts present based on predefined list
    Best, Hummer and Eaton [1].
    Adapted from: 'http://mdtraj.org/latest/examples/native-contact.html'

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used
    atom_pairs : np.array
        Pairs of atoms for which to determine if contacts are still present

    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`

    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """

    BETA_CONST = 10  # 1/nm, decreased to decrease slope
    LAMBDA_CONST = 2.5  # increased to encourage large changes

    # compute these distances for the whole trajectory
    r = md.compute_distances(traj, atom_pairs)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], atom_pairs)
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)

    print("total number of contacts is: %s" % atom_pairs.shape[0])
    print("value of q: %s" % q)
    return q


class SpecificContactsWrap(base_analysis):
    """Analyses the fraction of native contacts present based on a predefined
       list of contacts in the starting conformation.

    Parameters
    ----------
    base_struct : str or md.Trajectory,
        The base structure to compare for native contacts. This
        topology must match the structures to analyse. Can be provided
        as a pdb location or an md.Trajectory object.
    atom_pairs : str or array,
        The atom indice pairs to use for computing native contacts. Can be
        provided as a data file to load or an array.

    Attributes
    ----------
    output_name : str,
        The file containing rankings.
    """

    def __init__(self, base_struct, atom_pairs):
        # determine base_struct
        self.base_struct = base_struct
        if type(base_struct) is md.Trajectory:
            self.base_struct_md = self.base_struct
        else:
            self.base_struct_md = md.load(base_struct)
        # determine atom indices
        self.atom_pairs = atom_pairs
        if type(atom_pairs) is str:
            self.atom_pairs_vals = np.loadtxt(atom_pairs, dtype=int)
        else:
            self.atom_pairs_vals = self.atom_pairs

    @property
    def class_name(self):
        return "SpecificContactsWrap"

    @property
    def config(self):
        return {
            "base_struct": self.base_struct,
            "atom_pairs": self.atom_pairs,
        }

    @property
    def analysis_folder(self):
        return None

    @property
    def base_output_name(self):
        return "specific_contacts_per_state"

    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load("./data/full_centers.xtc", top=self.base_struct_md)
            contacts = best_hummer_q(centers, self.base_struct_md, self.atom_pairs_vals)
            np.save(self.output_name, contacts)
