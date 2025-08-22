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


def _calculate_number_interface_contacts(
    traj, interface_list, dist_cutoff, verbose=False
):
    """Compute the number of contacts at the specified interfaces

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    interface_list : list of of list of numpy arrays
        List containing the interfaces for which to compute scores
        Follows the pattern:
        [interface1, interface2, ...] where
        interfaceN is a 2-element list of numpy arrays containing
        the residue indices (zero indexed) of one half of the interface and the other half
        For example, the following would be a valid input:
        [[S2_rids, BH_rids], [BH_rids, FH_rids]]

    verbose : boolean
        Determines if intermediate scores are printed


    Returns
    -------
    contacts_per_interface : np.array, shape=(len(traj), len(interface_list))
        A contact count for each frame of `traj` and each interface
        in the interfacee list

    """
    contacts_per_interface = np.zeros((traj.n_frames, len(interface_list)), dtype=int)

    for i, (interface1_rids, interface2_rids) in enumerate(interface_list):
        contacts = np.zeros((traj.n_frames))
        # to avoid out of memory issues iterate over each residue at a time
        for j, rid in enumerate(interface2_rids):
            rid_pairs = list(itertools.product(interface1_rids, [rid]))

            # returns distances for each residue-residue contact in each frame of the trajectory
            # distances is shape=(n_frames, n_pairs)
            distances = md.compute_contacts(traj, rid_pairs, scheme="closest")[0]

            contacts = contacts + np.sum(distances < dist_cutoff, axis=1)

        contacts_per_interface[:, i] = contacts

    if verbose:
        print(contacts_per_interface)
        np.save("./data/contacts_per_interface.npy", contacts_per_interface)

    return contacts_per_interface.sum(axis=1)


def _calculate_number_residue_contacts(
    traj, interface_list, verbose=False, native_cutoff=0.45
):
    """Compute the total number of residue contacts between a collection of
    protein interfaces.

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    interface_list : list of of list of numpy arrays
        List containing the interfaces for which to compute contacts
        Follows the pattern:
        [interface1, interface2, ...] where
        interfaceN is a 2-element list of numpy arrays containing
        the atom indices of one half of the interface and the other half
        For example, the following would be a valid input:
        [[S2_heavy_atoms, BH_heavy_atoms], [BH_heavy_atoms, FH_heavy_atoms]]


    Returns
    -------
    contact_counts : np.array, shape=(len(traj),)
        The number of residue contacts in each frame of `traj`

    """

    contact_count_per_interface = np.zeros((traj.n_frames, len(interface_list)))

    for i, (interface1_ais, interface2_ais) in enumerate(interface_list):
        # to avoid computing a very large number of distances, we first narrow
        # down the list of atoms which make a contact

        adjacent_interface2_ais = md.compute_neighbors(
            traj, native_cutoff, interface1_ais, interface2_ais
        )

        unique_adjacent_interface2_ais = np.unique(
            np.concatenate(adjacent_interface2_ais)
        )

        adjacent_interface1_ais = md.compute_neighbors(
            traj, native_cutoff, unique_adjacent_interface2_ais, interface1_ais
        )
        unique_adjacent_interface1_ais = np.unique(
            np.concatenate(adjacent_interface1_ais)
        )

        atom_pairs = list(
            itertools.product(
                unique_adjacent_interface1_ais, unique_adjacent_interface2_ais
            )
        )
        dists = md.compute_distances(traj, atom_pairs)

        contact_count = []

        for cix in range(dists.shape[0]):
            contact_pairs = np.array(atom_pairs)[dists[cix] < native_cutoff]

            # Get unique (chain, residue) pair contacts
            contact_residues = np.unique(
                [
                    (
                        traj.top.atom(a1).residue.resSeq,
                        traj.top.atom(a1).residue.chain.index,
                        traj.top.atom(a2).residue.resSeq,
                        traj.top.atom(a2).residue.chain.index,
                    )
                    for (a1, a2) in contact_pairs
                ],
                axis=0,
            )

            contact_count.append(contact_residues.shape[0])

        contact_count_per_interface[:, i] = contact_count

    if verbose:
        print(contact_count_per_interface)

    contact_counts = contact_count_per_interface.sum(axis=1)

    return contact_counts


class InterfaceContactWrap(base_analysis):
    """Computes the number of total residue contacts at multiple interfaces.

    Parameters
    ----------
    interface_list : list of of list of numpy arrays
        List containing the interfaces for which to compute scores
        Follows the pattern:
        [interface1, interface2, ...] where
        interfaceN is a 2-element list of numpy arrays containing
        the residue indices (zero indexed) of one half of the interface and the other half
        For example, the following would be a valid input:
        [[S2_rids, BH_rids], [BH_rids, FH_rids]]

    verbose : boolean
        Determines if intermediate scores are printed

    dist_cutoff : double
        Cutoff distance that defines what constitutes a contact. Any two residues
        that are less than the dist_cutoff apart are considered a contact.


    Attributes
    ----------
    output_name : str,
        The file containing rankings.
    """

    def __init__(self, interface_list, dist_cutoff, verbose=False, base_struct=None):
        # load in interface list
        if type(interface_list) is str:
            with open(interface_list, "rb") as f:
                self.interface_list = pickle.load(f)
        else:
            self.interface_list = interface_list
        self.dist_cutoff = dist_cutoff
        self.verbose = verbose

    @property
    def class_name(self):
        return "InterfaceContactWrapper"

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
        return "contacts_per_state"

    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load("./data/full_centers.xtc", top="./prot_masses.pdb")
            # calculate and save contacts
            scores = _calculate_number_interface_contacts(
                centers, self.interface_list, self.dist_cutoff, verbose=self.verbose
            )
            np.save(self.output_name, scores)


class ContactCountWrap(base_analysis):
    """Analyses the number of residue contact pairs at predefined interfaces.

    Parameters
    ----------
    base_struct_md : str or md.Trajectory,
        Topology for loading centers. This topology must match the structures to analyse.
        Can be provided as a pdb location or an md.Trajectory object.

    interface_list : list of of list of numpy arrays
        List containing the interfaces for which to compute contacts
        Follows the pattern:
        [interface1, interface2, ...] where
        interfaceN is a 2-element list of numpy arrays containing
        the atom indices of one half of the interface and the other half
        For example, the following would be a valid input:
        [[S2_heavy_atoms, BH_heavy_atoms], [BH_heavy_atoms, FH_heavy_atoms]]

    verbose : boolean
        Determines if the count contact method prints intermediate values

    Attributes
    ----------
    output_name : str,
        The file containing rankings.
    """

    def __init__(self, base_struct_md, interface_list, verbose=False):
        # determine base_struct
        if type(base_struct_md) is md.Trajectory:
            self.base_struct_md = self.base_struct_md
        else:
            self.base_struct_md = md.load(base_struct_md)
        # load in interface list
        if type(interface_list) is str:
            with open(interface_list, "rb") as f:
                self.interface_list = pickle.load(f)
        else:
            self.interface_list = interface_list
        self.verbose = verbose

    @property
    def class_name(self):
        return "ContactCountWrap"

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
        return "contact_count_per_state"

    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load("./data/full_centers.xtc", top=self.base_struct_md)
            # calculate and save contacts
            contacts = calculate_number_residue_contacts(
                centers, self.interface_list, verbose=self.verbose
            )
            np.save(self.output_name, contacts)
