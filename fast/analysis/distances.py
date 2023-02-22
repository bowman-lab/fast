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
import itertools
import mdtraj as md
import numpy as np
import os
import sys
from .base_analysis import base_analysis
from .. import tools


#######################################################################
# code
#######################################################################


def load_domain_indices(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    domain0, domain1 = [], []
    for l in data:
        dat = l.split()
        if len(dat) == 2:
            domain0.append(dat[0])
            domain1.append(dat[1])
        if len(dat) == 1:
            domain0.append(dat[0])
    domain0 = np.array(domain0, dtype=int)
    domain1 = np.array(domain1, dtype=int)
    atom_indices = np.array(
        list(
            itertools.zip_longest(
                domain0, domain1, fillvalue=None)))
    return atom_indices


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
    center_of_mass : bool, default=False,
        Optionally calculate distance between center of mass between
        pairs of atoms. Specifically, calculates the distance between
        the center of mass of the first column of atoms and the center
        of mass of the second column of atoms.

    Attributes
    ----------
    output_name : str,
        The file containing rankings.
    """
    def __init__(
            self, atom_pairs, p_norm=1, set_points=None, center_of_mass=False):
        # set attributes
        self.atom_pairs = atom_pairs
        if type(self.atom_pairs) is str:
            try:
                self.atom_pairs = np.loadtxt(self.atom_pairs, dtype=int)
            except:
                self.atom_pairs = load_domain_indices(self.atom_pairs)
        if len(self.atom_pairs.shape) == 1:
            self.atom_pairs = [self.atom_pairs]
        self.p_norm = p_norm
        # check set points
        self.set_points = set_points
        if self.set_points is not None:
            self.set_points = np.array(set_points)
            if len(self.set_points) != len(self.atom_pairs):
                raise # number of set points does not match atom-pairs!!
        self.center_of_mass = center_of_mass

    @property
    def class_name(self):
        return "DistWrap"

    @property
    def config(self):
        return {
            'atom_pairs': self.atom_pairs,
            'p_norm': self.p_norm,
            'set_points': self.set_points,
            'center_of_mass': self.center_of_mass
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
            # optionall calculate center of mass between pairs
            if self.center_of_mass:
                # slice domains
                iis_domain0 = self.atom_pairs[:,0]
                iis_domain0 = np.array(
                    iis_domain0[np.where(iis_domain0)], dtype=int)
                iis_domain1 = self.atom_pairs[:,1]
                iis_domain1 = np.array(
                    iis_domain1[np.where(iis_domain1)], dtype=int)
                domain0 = centers.atom_slice(iis_domain0)
                domain1 = centers.atom_slice(iis_domain1)
                # obtain masses
                center_of_mass_domain0 = md.compute_center_of_mass(domain0)
                center_of_mass_domain1 = md.compute_center_of_mass(domain1)
                # obtain distances
                diffs = np.abs(
                    center_of_mass_domain0 - center_of_mass_domain1)
                distances = np.sqrt(
                    np.einsum('ij,ij->i', diffs, diffs))[:,None]
            else:
                # get distances of atom pairs
                distances = md.compute_distances(centers, atom_pairs=self.atom_pairs)
            if self.set_points is not None:
                diffs = np.abs(distances - self.set_points)
            else:
                diffs = np.abs(distances)
            norm_dists = np.sum(diffs**self.p_norm, axis=1)**(1/self.p_norm)
            np.save(self.output_name, norm_dists)    
