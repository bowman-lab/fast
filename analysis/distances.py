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

class DistWrap(base_analysis):
    """Analysis wrapper for calculating distances between atom pairs.

    Parameters
    ----------
    atom_pairs : array, shape=(n_pairs, 2),
         The list of atom-pairs to use for calculating distances.
    p_norm : int, default=1,
        The p-norm to use when processing distance pairs. i.e.
        ||x||_p := sum(|x_i|^p)^(1/p)
    set_points : array, shape=(n_pairs,), default=None,
        A list of reference distances for each atom pair. If provided,
        reports deviation from these distances.

    Attributes
    ----------
    output_name : str,
        The file containing rankings.
    """
    def __init__(
            self, atom_pairs, p_norm=1, set_points=None):
        # set attributes
        self.atom_pairs = atom_pairs
        if type(self.atom_pairs) is str:
            self.atom_pairs = np.loadtxt(self.atom_pairs, dtype=int)
            if len(self.atom_pairs.shape) == 1:
                self.atom_pairs = [self.atom_pairs]
        self.p_norm = p_norm
        self.set_points = np.array(set_points)
        # check set points
        if self.set_points is not None:
            if len(self.set_points) != len(self.atom_pairs):
                raise # number of set points does not match atom-pairs!!

    @property
    def class_name(self):
        return "DistWrap"

    @property
    def config(self):
        return {
            'atom_pairs': self.atom_pairs,
            'p_norm': self.p_norm,
            'set_points': self.set_points
        }

    @property
    def analysis_folder(self):
        return None

    @property
    def base_output_name(self):
        return "distance_per_state"


    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load(
                "./data/full_centers.xtc", top="./prot_masses.pdb")
            # get distances of atom pairs
            distances = md.compute_distances(centers, atom_pairs=self.atom_pairs)
            if self.set_points is not None:
                diffs = np.abs(distances - self.set_points)
            else:
                diffs = np.abs(distances)
            norm_dists = np.sum(diffs**self.p_norm, axis=1)**(1/self.p_norm)
            np.save(self.output_name, norm_dists)
        
