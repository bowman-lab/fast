# Author: Artur Meller <ameller@wustl.edu>, Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential


#######################################################################
# imports
#######################################################################

import itertools
import mdtraj as md
import numpy as np
import os
from .base_analysis import base_analysis
from .. import tools
import math

#######################################################################
# code
#######################################################################


def dot_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])


def norm(x):
    return np.sqrt(dot_product(x, x))


def normalize(x):
    return [x[i] / norm(x) for i in range(len(x))]


def project_onto_plane(x, n):
    d = dot_product(x, n) / norm(n)
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]


def angle(v1, v2):
    return np.arccos(
        np.sum(v1 * v2, axis=1)
        / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    )


def _calculate_axial_angle(
    traj,
    ref_struct1,
    ref_struct2,
    ref1_sel1,
    ref1_sel2,
    ref2_sel2,
    traj_sel1,
    traj_sel2,
):
    """
    traj: the trajectory being analyzed
    ref_struct1: structure 1 used to build plane (this will be the structure you compute the anlge relative to)
    ref_struct2 structure 2 used to build plane
    ref1_sel1 (etc): residues used to build a point of a vector (in the format of, for example 'residue 1 and name CA')
    """
    p1 = md.compute_center_of_mass(
        ref_struct1.atom_slice(ref_struct1.top.select(ref1_sel1))
    )
    p2 = md.compute_center_of_mass(
        ref_struct1.atom_slice(ref_struct1.top.select(ref1_sel2))
    )
    p3 = md.compute_center_of_mass(
        ref_struct2.atom_slice(ref_struct2.top.select(ref2_sel2))
    )

    v1 = p2 - p1
    v2 = p3 - p1

    cross = np.cross(v1, v2)

    p1_traj = md.compute_center_of_mass(traj.atom_slice(traj.top.select(traj_sel1)))
    p2_traj = md.compute_center_of_mass(traj.atom_slice(traj.top.select(traj_sel2)))
    vec_traj = p2_traj - p1_traj

    proj_vec = []
    for e in vec_traj:
        v = project_onto_plane(e, cross[0])
        proj_vec.append(v)

    angs = np.rad2deg(angle(np.array(proj_vec), v1))
    return angs


def _align_to_ref1(traj, ref1, alignment_sel_string_ref1, alignment_sel_string_traj):
    traj.superpose(
        ref1,
        ref_atom_indices=ref1.top.select(alignment_sel_string_ref1),
        atom_indices=traj.top.select(alignment_sel_string_traj),
    )
    return traj


class AxialAngleWrap(base_analysis):
    """Computes an axial angle for a selected region based on two reference structures.
    Parameters
    ----------
    ref_struct1: md.Trajectory or str
        One of two reference structure used to construct axial plane. This is the structure
        we will compute angle relative to.
    ref_struct2: md.Trajectory or str
        One of two referene structures used to construct axial plane.
    ref1_sel1: str
        Path to file containing mdtraj selection string used to build point 1 defining axial plane.
        These residues will be used to select center of mass for reference structure 1.
    ref1_sel2: str
        Path to file containing mdtraj selection string used to build point 2 defining axial plane.
        These residues will be used to select center of mass for reference structure 1.
    ref2_sel1: str
        Path to file containing mdtraj selection string used to build point 3 defining axial plane.
        These residues will be used to select center of mass for reference structure 2.
    traj_sel1: str
        Path to file containing mdtraj selection string used to define point 1 for trajectory being
        analyzed. We will project vector from point 2 to point 1 onto axial plane.
    traj_sel2: str
        Path to file containing mdtraj selection string used to define point 2 for trajectory being
        analyzed. We will project vector from point 2 to point 1 onto axial plane.
    alignment_sel_string_ref1: str
        Path to file containing mdtraj selection string for reference atoms to be used for aligning trajectory being analyzed
        to the reference structure 1.
    alignment_sel_string_traj: str
        Path to file containing mdtraj selection string for atoms to be used for aligning trajectory being analyzed
        to the reference structure 1. This selection string is specific to the analyzedd
        trajectory.
    Attributes
    ----------
    output_name : str,
        The file containing rankings.
    """

    def __init__(
        self,
        ref_struct1,
        ref_struct2,
        ref1_sel1,
        ref1_sel2,
        ref2_sel1,
        traj_sel1,
        traj_sel2,
        alignment_sel_string_ref1,
        alignment_sel_string_traj,
        base_struct=None,
    ):
        # load in reference structures
        if type(ref_struct1) is str:
            self.ref_struct1 = md.load(ref_struct1)
        else:
            self.ref_struct1 = ref_struct1
        if type(ref_struct2) is str:
            self.ref_struct2 = md.load(ref_struct2)
        else:
            self.ref_struct2 = ref_struct2
        # load in selection strings needed for defining points
        with open(ref1_sel1, "r") as f:
            self.ref1_sel1 = f.read()
        with open(ref1_sel2, "r") as f:
            self.ref1_sel2 = f.read()
        with open(ref2_sel1, "r") as f:
            self.ref2_sel1 = f.read()
        with open(traj_sel1, "r") as f:
            self.traj_sel1 = f.read()
        with open(traj_sel2, "r") as f:
            self.traj_sel2 = f.read()
        with open(traj_sel2, "r") as f:
            self.traj_sel2 = f.read()
        with open(alignment_sel_string_ref1, "r") as f:
            self.alignment_sel_string_ref1 = f.read()
        with open(alignment_sel_string_traj, "r") as f:
            self.alignment_sel_string_traj = f.read()

    @property
    def class_name(self):
        return "AxialAngleWrapper"

    @property
    def config(self):
        return {
            "ref_struct1": self.ref_struct1,
            "ref_struct2": self.ref_struct2,
        }

    @property
    def analysis_folder(self):
        return None

    @property
    def base_output_name(self):
        return "axial_angle_per_state"

    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load("./data/full_centers.xtc", top="./prot_masses.pdb")
            # align centers to ref struct 1
            _align_to_ref1(
                centers,
                self.ref_struct1,
                self.alignment_sel_string_ref1,
                self.alignment_sel_string_traj,
            )
            angles = _calculate_axial_angle(
                centers,
                self.ref_struct1,
                self.ref_struct2,
                self.ref1_sel1,
                self.ref1_sel2,
                self.ref2_sel1,
                self.traj_sel1,
                self.traj_sel2,
            )
            np.save(self.output_name, angles)
