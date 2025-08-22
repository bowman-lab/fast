# SPDX-License-Identifier: LGPL-2.1-or-later
# Author: Gavin N. John <gavinnjohn@gmail.com>
# Contributors:
# Copyright (C) 2025, University of Pennsylvania AND California Institute of Technology (Joint)

import numpy as np
import os
from .base_analysis import base_analysis

from pypocketminer.validate_performance_on_xtals import process_strucs

import tempfile

from scipy.spatial import QhullError, Voronoi
from scipy.spatial.distance import cdist
import trimesh
import mdtraj as md
import traceback

def get_backbone_residue_centers(traj):
    """Get geometric centers of protein residues, matching process_strucs() residue order."""
    prot_iis = traj.top.select("protein and (name N or name CA or name C or name O)")
    prot_bb = traj.atom_slice(prot_iis)
    
    residue_centers = []
    for res in prot_bb.top.residues:
        atom_indices = [atom.index for atom in res.atoms]
        coords = prot_bb.xyz[:, atom_indices, :]  # shape: (n_frames, n_atoms_in_res, 3)
        center = coords.mean(axis=1)  # mean over atoms
        residue_centers.append(center)
    
    return np.stack(residue_centers, axis=1)

def compute_3d_voronoi_cells(points: np.ndarray, nitpicky: bool = True) -> dict:
    """
    Approximates 3D Voronoi cells using Delaunay dual, returns trimesh meshes in-memory. Clips to CHull.

    Parameters:
        points (np.ndarray): Nx3 input point cloud.

    Returns:
        dict[int, trimesh.Trimesh]: Maps point index to its Voronoi cell mesh.
    """


    unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)
    unique_index_to_orig_indices = {}
    for orig_idx, unique_idx in enumerate(inverse_indices):
        unique_index_to_orig_indices.setdefault(unique_idx, []).append(orig_idx)
    
    bounding_points = []
    maxes = [np.max(unique_points[:, i]) for i in range(3)]
    mins = [np.min(unique_points[:, i]) for i in range(3)]
    D = max([maxes[i] - mins[i] for i in range(3)])
    for i in range(2 ** 3):
        bounding_points.append([
            maxes[0] + D if i & 0b001 == 0b001 else mins[0] - D,
            maxes[1] + D if i & 0b010 == 0b010 else mins[1] - D,
            maxes[2] + D if i & 0b100 == 0b100 else mins[2] - D,
        ])
    bounding_points = np.array(bounding_points)

    voro = Voronoi(np.vstack([unique_points, bounding_points]))

    unique_voronoi_cells = {}
    hull = trimesh.convex.convex_hull(unique_points)
    if not hull.is_volume:
        raise QhullError(f"CHull is degenerate!")
    for i in range(len(unique_points)):
        try:
            region_idx = voro.point_region[i]
            if region_idx < 0:
                raise RuntimeError(f"Failed to get region index for points {unique_index_to_orig_indices[i]}")
            region_vtx_idx = voro.regions[region_idx]
            if any([point < 0 for point in region_vtx_idx]):
                raise RuntimeError(f"Failed to get all points for region for points {unique_index_to_orig_indices[i]}")
            try:
                # Voronoi cells are convex, so taking the CHull of the vertices will get us
                # the same region more cheaply and without risking watertightness
                cell = trimesh.convex.convex_hull([voro.vertices[j] for j in region_vtx_idx])
                print(f"Initial mesh for points {unique_index_to_orig_indices[i]}: {cell}")
                print(f"{cell.vertices}")
                # Taking the chull of the intersection is fine because we're already convex
                # This serves to repair any issues with the intersection mesh (e.g. holes)
                # introduced by floating point artifacts and such
                final_mesh = cell.intersection(hull).convex_hull
                print(f"Final mesh for points {unique_index_to_orig_indices[i]}: {final_mesh}")
                print(f"{final_mesh.vertices}")
                if final_mesh.is_volume:
                    unique_voronoi_cells[i] = final_mesh
                else:
                    raise RuntimeError(f"Failed to get volume for points {unique_index_to_orig_indices[i]}")
            except Exception as e:
                if nitpicky:
                    raise
                else:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    # Set it to empty mesh
                    unique_voronoi_cells[i] = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        except Exception as e:
            print(f"Failed for points {unique_index_to_orig_indices[i]}")
            raise

    full_voronoi_cells = {}
    for unique_idx, orig_idxs in unique_index_to_orig_indices.items():
        mesh = unique_voronoi_cells.get(unique_idx)
        if mesh:
            for orig_idx in orig_idxs:
                full_voronoi_cells[orig_idx] = mesh

    return full_voronoi_cells


def physically_capped_voronoi_score(points, probs, power=1, epsilon=1e-3, nitpicky: bool = True):
    """
    Computes a geometry-aware score using Voronoi cells, capping infinite regions
    by intersecting with the Convex Hull.

    Parameters
    ----------
    points : (N, 3) np.ndarray
        3D coordinates of residues.
    probs : (N,) np.ndarray
        Per-residue probabilities.

    Returns
    -------
    score : float
        Weighted sum of Voronoi volumes restricted by the CHull.
    """

    if len(points) != len(probs):
        raise RuntimeError(f"Number of points ({len(points)}) does not match number of probs ({len(probs)})")

    if len(points) == 0:
        return 0.0

    try:
        cells = compute_3d_voronoi_cells(points, nitpicky=nitpicky)
    except QhullError as e:
        if nitpicky:
            raise
        else:
            traceback.print_exception(type(e), e, e.__traceback__)
            print("Points are coplanar, score is zero")
            return 0.0

    total_score = 0.0
    for i, cell_mesh in cells.items():
        total_score += pow(cell_mesh.volume * probs[i], power)

    return total_score


class PMExpectedVolumeWrap(base_analysis):
    """
    Runs PocketMiner on a trajectory or list of PDBs and returns
    the maximum per-residue likelihood for each structure.

    Parameters
    ----------
    model : TODO
    """

    def __init__(self, model, power=1):
        self.model = model
        self.power = power

    @property
    def class_name(self):
        return "PMExpectedVolumeWrap"

    @property
    def config(self):
        return {
            "model": self.model,
            "power": self.power
        }

    @property
    def base_output_name(self):
        return "pocketminer_expected_volume"

    @property
    def analysis_folder(self):
        return None

    def run(self):
        if os.path.exists(self.output_name):
            return

        # load centers
        traj = md.load(os.path.join(self.msm_dir, "data/full_centers.xtc"), top=os.path.join(self.msm_dir, "prot_masses.pdb"))

        # Calculate scores
        scores = np.zeros(traj.n_frames, dtype="float64")
        for i in range(traj.n_frames):
            X, S, mask = process_strucs([traj[i]])
            preds = np.array(self.model(X, S, mask, train=False, res_level=True))[0]
            scores[i] = physically_capped_voronoi_score(get_backbone_residue_centers(traj[i])[0], preds, power=self.power, nitpicky=False)

        np.save(self.output_name, scores)

class PMExpectedVolumeWrapWithBase(base_analysis):
    """
    Runs PocketMiner on a base structure and returns
    the maximum per-residue likelihood for each structure.

    Parameters
    ----------
    model : TODO
    """

    def __init__(self, model, power=1):
        self.model = model
        self.power = power

    @property
    def class_name(self):
        return "PMExpectedVolumeWrapWithBase"

    @property
    def config(self):
        return {
            "model": self.model,
            "power": self.power,
        }

    @property
    def base_output_name(self):
        return "pocketminer_expected_volume_with_base"

    @property
    def analysis_folder(self):
        return None

    def run(self):
        if os.path.exists(self.output_name):
            return

        # load base
        base_top = os.path.join(self.msm_dir, "prot_masses.pdb")
        traj_base = md.load(base_top, top=base_top)
        X, S, mask = process_strucs([traj_base[0]])
        preds = np.array(self.model(X, S, mask, train=False, res_level=True))[0]

        # load centers
        traj = md.load(os.path.join(self.msm_dir, "data/full_centers.xtc"), top=os.path.join(self.msm_dir, "prot_masses.pdb"))

        # Calculate scores
        scores = np.zeros(traj.n_frames, dtype="float64")
        for i in range(traj.n_frames):
            scores[i] = physically_capped_voronoi_score(get_backbone_residue_centers(traj[i])[0], preds, power=self.power, nitpicky=False)

        np.save(self.output_name, scores)

class PMLikelihoodSumWrap(base_analysis):
    def __init__(self, model, power=1):
        self.model = model
        self.power = power

    @property
    def class_name(self):
        return "PMLikelihoodSumWrap"

    @property
    def config(self):
        return {
            "model": self.model,
            "power": self.power,
        }

    @property
    def base_output_name(self):
        return "pocketminer_likelihood_sum"

    @property
    def analysis_folder(self):
        return None

    def run(self):
        if os.path.exists(self.output_name):
            return

        # load centers
        traj = md.load(os.path.join(self.msm_dir, "data/full_centers.xtc"), top=os.path.join(self.msm_dir, "prot_masses.pdb"))

        # Calculate scores
        scores = np.zeros(traj.n_frames, dtype="float64")
        for i in range(traj.n_frames):
            X, S, mask = process_strucs([traj[i]])
            preds = np.array(self.model(X, S, mask, train=False, res_level=True))[0]
            scores[i] = (preds ** self.power).sum()

        np.save(self.output_name, scores)
