# Author: Artur Meller <ameller@wustl.edu>, Maxwell I. Zimmerman <mizimmer@wustl.edu>
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
import pickle


#######################################################################
# code
#######################################################################


def _score_per_interface_calculator(
    traj,
    interface_list,
    center_of_mass=True,
    verbose=False,
    beta=4.0,
    lambda_constant=1.2,
    base_struct=None,
):
    """Compute the distance between several interfaces, apply a contact function to reward
    partial dissociation, and returns a score for each interface.

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    interface_list : list of of list of numpy arrays
        List containing the interfaces for which to compute scores
        Follows the pattern:
        [interface1, interface2, ...] where
        interfaceN is a 2-element list of numpy arrays containing
        the atom indices of one half of the interface and the other half
        For example, the following would be a valid input:
        [[S2_heavy_atoms, BH_heavy_atoms], [BH_heavy_atoms, FH_heavy_atoms]]

    center_of_mass : boolean
        Determines if COM is used or if pairwise distances are used for distance
        calculation

    verbose : boolean
        Determines if intermediate scores are printed

    beta : double
        Sets the rate of growth in the scoring function. The lower the value the
        more gradual the growth in the score.

    lambda_constant : double
        Determines the change needed relative to r0 to reach a score of 0.5.
        When lambda is 1.2, a 20% increase in the distance will lead to a score of 0.5.

    base_struct: md.Trajectory
        The structure that is used to determine r0. Typically the starting
        structure for simulations. If None, we assume this is the first
        structure in centers.

    Returns
    -------
    score_per_interface : np.array, shape=(len(traj), len(interface_list))
        A dissociation score for each frame of `traj` and each interface
        in the interfacee list

    """
    score_per_interface = np.zeros((traj.n_frames, len(interface_list)))
    if center_of_mass:
        for i, (interface1_ais, interface2_ais) in enumerate(interface_list):
            # slice domains
            domain0 = traj.atom_slice(interface1_ais)
            domain1 = traj.atom_slice(interface2_ais)
            # obtain masses
            center_of_mass_domain0 = md.compute_center_of_mass(domain0)
            center_of_mass_domain1 = md.compute_center_of_mass(domain1)
            # obtain distances
            diffs = np.abs(center_of_mass_domain0 - center_of_mass_domain1)
            distances = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))[:, None]
            # derive scores from distances
            # assumes starting structure is center 0
            if base_struct is None:
                r0 = distances[0]
            else:
                ref_domain0 = base_struct.atom_slice(interface1_ais)
                ref_domain1 = base_struct.atom_slice(interface2_ais)

                ref_com_domain0 = md.compute_center_of_mass(ref_domain0)
                ref_com_domain1 = md.compute_center_of_mass(ref_domain1)

                diffs = np.abs(ref_com_domain0 - ref_com_domain1)
                r0 = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))[0]

            scores = 1.0 / (1.0 + np.exp(-beta * (distances - lambda_constant * r0)))
            if verbose:
                print(np.hstack((distances, scores)))
            score_per_interface[:, i] = scores.flatten()

        return score_per_interface

    else:
        POTENTIAL_CONTACT_CUTOFF = 1.0  # nm
        for i, (interface1_ais, interface2_ais) in enumerate(interface_list):
            # Determine potential contact points in starting structure
            atom_pairs = list(itertools.product(interface1_ais, interface2_ais))
            if base_struct is None:
                distances = md.compute_distances(traj[0], atom_pairs)[0]
            else:
                distances = md.compute_distances(base_struct, atom_pairs)[0]
            potential_contacts = np.array(atom_pairs)[
                distances < POTENTIAL_CONTACT_CUTOFF
            ]

            # Compute distances for all neighbors for all centers
            distances = md.compute_distances(traj, potential_contacts)

            # derive scores from distances
            if base_struct is None:
                r0 = distances[0]
            else:
                r0 = md.compute_distances(base_struct, potential_contacts)
            scores = np.mean(
                1.0 / (1.0 + np.exp(-beta * (distances - lambda_constant * r0))), axis=1
            )
            if verbose:
                print(np.hstack((distances, scores)))
            score_per_interface[:, i] = scores.flatten()

        return score_per_interface


def calculate_dissociation_score(
    traj,
    interface_list,
    center_of_mass=True,
    verbose=False,
    beta=4.0,
    lambda_constant=1.2,
    base_struct=None,
):
    """Compute the distance between several interfaces, apply a contact function to reward
    partial dissociation, and sum scores over interfaces.

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    interface_list : list of of list of numpy arrays
        List containing the interfaces for which to compute scores
        Follows the pattern:
        [interface1, interface2, ...] where
        interfaceN is a 2-element list of numpy arrays containing
        the atom indices of one half of the interface and the other half
        For example, the following would be a valid input:
        [[S2_heavy_atoms, BH_heavy_atoms], [BH_heavy_atoms, FH_heavy_atoms]]

    center_of_mass : boolean
        Determines if COM is used or if pairwise distances are used for distance
        calculation

    verbose : boolean
        Determines if intermediate scores are printed

    beta : double
        Sets the rate of growth in the scoring function. The lower the value the
        more gradual the growth in the score.

    lambda_constant : double
        Determines the change needed relative to r0 to reach a score of 0.5.
        When lambda is 1.2, a 20% increase in the distance will lead to a score of 0.5.

    base_struct: md.Trajectory
        The structure that is used to determine r0. Typically the starting
        structure for simulations. If None, we assume this is the first
        structure in centers.

    Returns
    -------
    dissociation_score : np.array, shape=(len(traj),)
        A dissociation score for each frame of `traj`

    """
    score_per_interface = _score_per_interface_calculator(
        traj,
        interface_list,
        center_of_mass,
        verbose,
        beta,
        lambda_constant,
        base_struct,
    )

    return score_per_interface.sum(axis=1)


def calculate_constrained_target_dissociation_score(
    traj,
    interface_list,
    already_open,
    center_of_mass=True,
    verbose=False,
    beta=4.0,
    lambda_constant=1.2,
    base_struct=None,
):
    """Compute the distance between several interfaces, apply a contact function to reward
    partial dissociation, and returns a score between 0 and n where n is the number of interfaces
    we wish to open. It sums over the interfaces we wish to open and subtracts a penalty for
    those interfaces that are already open.

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    interface_list : list of of list of numpy arrays
        List containing the interfaces for which to compute scores
        Follows the pattern:
        [interface1, interface2, ...] where
        interfaceN is a 2-element list of numpy arrays containing
        the atom indices of one half of the interface and the other half
        For example, the following would be a valid input:
        [[S2_heavy_atoms, BH_heavy_atoms], [BH_heavy_atoms, FH_heavy_atoms]]

    already_open : array of booleans
        Describes which of the interfaces are already open in the initial structure

    center_of_mass : boolean
        Determines if COM is used or if pairwise distances are used for distance
        calculation

    verbose : boolean
        Determines if intermediate scores are printed

    beta : double
        Sets the rate of growth in the scoring function. The lower the value the
        more gradual the growth in the score.

    lambda_constant : double
        Determines the change needed relative to r0 to reach a score of 0.5.
        When lambda is 1.2, a 20% increase in the distance will lead to a score of 0.5.

    base_struct: md.Trajectory
        The structure that is used to determine r0. Typically the starting
        structure for simulations. If None, we assume this is the first
        structure in centers.

    Returns
    -------
    dissociation_score : np.array, shape=(len(traj),)
        A dissociation score for each frame of `traj`

    """
    score_per_interface = _score_per_interface_calculator(
        traj,
        interface_list,
        center_of_mass,
        verbose,
        beta,
        lambda_constant,
        base_struct,
    )

    custom_scores = score_per_interface[:, ~already_open].sum(axis=1) - (
        1 - score_per_interface[:, already_open]
    ).sum(axis=1)

    # set any negative scores to 0
    custom_scores = [max(s, 0) for s in custom_scores]

    return custom_scores


class ConstrainedTargetInterfaceDissociationWrap(base_analysis):
    """Computes a dissociaiton score for a multi-interface dissociation simulation
         where one wishes to open a complex at one interface while maintaining an
         open state at another interface. This has proven to be advantageous
         when opening a complex that already has other interfaces dissociated.

     Parameters
     ----------
     interface_list : list of list of numpy arrays
         List containing the interfaces for which to compute contacts
         Follows the pattern:
         [interface1, interface2, ...] where
         interfaceN is a 2-element list of numpy arrays containing
         the atom indices of one half of the interface and the other half
         For example, the following would be a valid input:
         [[S2_heavy_atoms, BH_heavy_atoms], [BH_heavy_atoms, FH_heavy_atoms]]

     center_of_mass : boolean
         Determines if COM is used or if pairwise distances are used for distance
         calculation

     verbose : boolean
         Determines if intermediate scores are printed

     beta : double
         Sets the rate of growth in the scoring function. The lower the value the
         more gradual the growth in the score.

    lambda_constant : double
         Determines the change needed relative to r0 to reach a score of 0.5.
         When lambda is 1.2, a 20% increase in the distance will lead to a score of 0.5.

     base_struct: md.Trajectory or str
         The structure that is used to determine r0. Typically the starting
         structure for simulations. If None, we assume this is the first
         structure in centers.

     already_open: np.array of booleans
         Describes which of the interfaces are already open in the initial structure

     Attributes
     ----------
     output_name : str,
         The file containing rankings.
    """

    def __init__(
        self,
        interface_list,
        already_open,
        center_of_mass=True,
        verbose=False,
        beta=4.0,
        lambda_constant=1.2,
        base_struct=None,
    ):
        # load in interface list
        if type(interface_list) is str:
            with open(interface_list, "rb") as f:
                self.interface_list = pickle.load(f)
        else:
            self.interface_list = interface_list
        self.already_open = already_open
        self.center_of_mass = center_of_mass
        self.verbose = verbose
        self.beta = beta
        self.lambda_constant = lambda_constant
        if type(base_struct) is str:
            self.base_struct = md.load(base_struct)
        else:
            self.base_struct = base_struct

    @property
    def class_name(self):
        return "ConstrainedTargetInterfaceDissociationWrap"

    @property
    def config(self):
        return {
            "interface_list": self.interface_list,
        }

    @property
    def analysis_folder(self):
        return None

    @property
    def base_output_name(self):
        return "dissociation_score_per_state"

    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load("./data/full_centers.xtc", top="./prot_masses.pdb")
            # calculate and save contacts
            scores = calculate_constrained_target_dissociation_score(
                centers,
                self.interface_list,
                self.already_open,
                center_of_mass=self.center_of_mass,
                verbose=self.verbose,
                beta=self.beta,
                lambda_constant=self.lambda_constant,
                base_struct=self.base_struct,
            )
            np.save(self.output_name, scores)


class MultipleInterfaceDissociationWrap(base_analysis):
    """Computes a dissociaiton score for a multi-interface dissociation simulation.

     Parameters
     ----------
     interface_list : list of list of numpy arrays
         List containing the interfaces for which to compute contacts
         Follows the pattern:
         [interface1, interface2, ...] where
         interfaceN is a 2-element list of numpy arrays containing
         the atom indices of one half of the interface and the other half
         For example, the following would be a valid input:
         [[S2_heavy_atoms, BH_heavy_atoms], [BH_heavy_atoms, FH_heavy_atoms]]

     center_of_mass : boolean
         Determines if COM is used or if pairwise distances are used for distance
         calculation

     verbose : boolean
         Determines if intermediate scores are printed

     beta : double
         Sets the rate of growth in the scoring function. The lower the value the
         more gradual the growth in the score.

    lambda_constant : double
         Determines the change needed relative to r0 to reach a score of 0.5.
         When lambda is 1.2, a 20% increase in the distance will lead to a score of 0.5.

     base_struct: md.Trajectory or str
         The structure that is used to determine r0. Typically the starting
         structure for simulations. If None, we assume this is the first
         structure in centers.

     Attributes
     ----------
     output_name : str,
         The file containing rankings.
    """

    def __init__(
        self,
        interface_list,
        center_of_mass=True,
        verbose=False,
        beta=4.0,
        lambda_constant=1.2,
        base_struct=None,
    ):
        # load in interface list
        if type(interface_list) is str:
            with open(interface_list, "rb") as f:
                self.interface_list = pickle.load(f)
        else:
            self.interface_list = interface_list
        self.center_of_mass = center_of_mass
        self.verbose = verbose
        self.beta = beta
        self.lambda_constant = lambda_constant
        if type(base_struct) is str:
            self.base_struct = md.load(base_struct)
        else:
            self.base_struct = base_struct

    @property
    def class_name(self):
        return "MultipleInterfaceDissociationWrapper"

    @property
    def config(self):
        return {
            "interface_list": self.interface_list,
        }

    @property
    def analysis_folder(self):
        return None

    @property
    def base_output_name(self):
        return "dissociation_score_per_state"

    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load("./data/full_centers.xtc", top="./prot_masses.pdb")
            # calculate and save contacts
            scores = calculate_dissociation_score(
                centers,
                self.interface_list,
                center_of_mass=self.center_of_mass,
                verbose=self.verbose,
                beta=self.beta,
                lambda_constant=self.lambda_constant,
                base_struct=self.base_struct,
            )
            np.save(self.output_name, scores)
