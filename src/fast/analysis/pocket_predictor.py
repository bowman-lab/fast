# Authors: Maxwell I. Zimmerman <mizimmer@wustl.edu>, Artur Meller <ameller@wustl.edu>
# Michael Ward <mdward@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential


#######################################################################
# imports
#######################################################################


import itertools
import glob
import mdtraj as md
import numpy as np
import os
from .base_analysis import base_analysis
from .. import tools
from enspara.geometry import pockets
from functools import partial
from multiprocessing import Pool
import tensorflow as tf


#######################################################################
# code
#######################################################################


def _get_filenames(msm_dir):
    """Returns pdb filenames"""
    pdb_filenames = glob.glob(msm_dir + "/centers_masses/state*.pdb")
    pdb_filenames_full = np.array(
        [os.path.abspath(filename) for filename in np.sort(pdb_filenames)]
    )
    return pdb_filenames_full


def _save_pocket_element(save_info):
    """Helper function for parallelizing the saving the pdb of pocket
    elements and a data file containing a list of pocket volumes."""
    # gather save info
    pocket_element, state_name, output_folder = save_info
    # make state analysis directory and save pdb coords
    _ = tools.run_commands("mkdir " + output_folder)
    pok_output_name = output_folder + "/" + state_name + "_pockets.pdb"
    pocket_element.save_pdb(pok_output_name)
    # generate pocket size array (first element is the total size)
    pok_sizes = np.array(
        [len(list(resi.atoms)) for resi in list(pocket_element.top.residues)]
    )
    pok_sizes = np.append(pok_sizes.sum(), pok_sizes)
    # save pocket sizes
    pok_details_output_name = output_folder + "/pocket_sizes.dat"
    np.savetxt(pok_details_output_name, pok_sizes, fmt="%d")
    return


def save_pocket_elements(pocket_func, centers, pdb_filenames, output_folder, n_procs):
    """Function to calculate and save pocket elements / pocket info"""
    # determine the base state name and output folders tp create
    state_names = np.array(
        [filename.split("/")[-1].split("-")[0] for filename in pdb_filenames]
    )
    output_folders = np.array(
        [output_folder + "/" + state_name for state_name in state_names]
    )
    # calculate pockets
    pocket_elements = pocket_func(centers)
    # generate zipped info to send to helper
    save_info = list(zip(pocket_elements, state_names, output_folders))
    # paralellize
    pool = Pool(processes=n_procs)
    pool.map(_save_pocket_element, save_info)
    pool.terminate()
    return


def _parse_pocket_file(pocket_info):
    """Helper to parse pocket data file for a pocket volume."""
    # unpack info
    filename, pocket_num = pocket_info
    # open file and take value at position `pocket_num`
    pocket_sizes = np.loadtxt(filename, dtype=int)
    if pocket_num is None:
        psize = np.sum(pocket_sizes)
    else:
        psize = pocket_sizes[pocket_num]
    return psize


class TopPockets:
    """Reports pocket volume of a particular pocket.

    Parameters
    ----------
    pocket_number : int, default=None,
        The pocket number to report volume. If None, will provide the
        total pocket volume.
    """

    def __init__(self, pocket_number=None, n_cpus=1):
        self.pocket_number = pocket_number
        self.n_cpus = n_cpus

    def parse_pockets(self, pockets_dir):
        """Searches through output directory for pocket_size files and
        parses them for pocket sizes of a given pocket num."""
        pockets_dir = os.path.abspath(pockets_dir)
        # get data file names
        pocket_files = np.sort(glob.glob(pockets_dir + "/*/pocket_sizes.dat"))
        # parallelize the parsing
        file_info = list(zip(pocket_files, itertools.repeat(self.pocket_number)))
        pool = Pool(processes=self.n_cpus)
        pockets = pool.map(_parse_pocket_file, file_info)
        pool.terminate()
        return np.array(pockets)


class ResiduePockets:
    """Reports pocket volume around selected residues.

    Parameters
    ----------
    atom_indices : array-like, shape=(n_atoms, ),
        The atom indices to search around.
    distance_cutoff : float, default=1.0,
        The distance to search around atoms for pocket volumes.
    n_cpus : int, default=1,
        The number of cpus to use for determining pocket volumes.
    """

    def __init__(self, atom_indices, distance_cutoff=1.0, n_cpus=1):
        self.atom_indices = atom_indices
        if isinstance(self.atom_indices, (str)):
            try:
                self.atom_indices = np.loadtxt(self.atom_indices, dtype=int)
            except:
                self.atom_indices = np.array(np.load(self.atom_indices), dtype=int)
        self.distance_cutoff = distance_cutoff
        self.n_cpus = n_cpus

    def parse_pockets(self, pockets_dir):
        """Searches through output directory for pdbs and pockets and
        parses them for pocket sizes around selected residues."""
        pockets_dir = os.path.abspath(pockets_dir)
        # get data file names
        pocket_files = np.sort(glob.glob(pockets_dir + "/*/state*.pdb"))
        pdb_files = np.sort(glob.glob(pockets_dir + "/../centers_masses/state*.pdb"))
        # parallelize the parsing
        file_info = list(
            zip(
                pdb_files,
                pocket_files,
                itertools.repeat(self.atom_indices),
                itertools.repeat(self.distance_cutoff),
            )
        )
        pool = Pool(processes=self.n_cpus)
        pockets = pool.map(_determine_pocket_neighbors, file_info)
        pool.terminate()
        return np.array(pockets)


def _determine_pocket_neighbors(file_info):
    pdb_filename, pocket_filename, atom_indices, distance_cutoff = file_info
    pdb = md.load(pdb_filename)
    pdb_pockets = md.load(pocket_filename)
    pdb_xyz = pdb.xyz[0, atom_indices]
    pdb_pockets_xyz = pdb_pockets.xyz[0]
    close_iis = []
    for n in np.arange(pdb_xyz.shape[0]):
        diffs = np.abs(pdb_pockets_xyz - pdb_xyz[n])
        dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        close_iis.append(np.where(dists < distance_cutoff)[0])
    close_iis = np.unique(np.concatenate(close_iis))
    return len(close_iis)


def make_pocket_predictions(model, nn_path, centers, opt=tf.keras.optimizers.Adam()):
    """Predicts a pocket increase probability for each residue
    in each state in the centers

    model : object, MQAModel
        An object of class MQA Model which specifies the architecture
        of the network


    nn_path : str
        A path specifying the location of the input model's index
        and dat files. This path is passed to load_checkpoint

    centers : MDTraj trajectory
        A trajectory containing the structures for which to make a prediction
    """
    from models import MQAModel
    from util import load_checkpoint

    # Load model weights from checkpoint
    load_checkpoint(model, opt, nn_path)

    aa2index = {
        "ALA": 0,
        "ARG": 1,
        "ASN": 2,
        "ASP": 3,
        "CYS": 4,
        "CYM": 4,
        "GLU": 6,
        "GLN": 5,
        "GLY": 7,
        "HIS": 8,
        "ILE": 9,
        "LEU": 10,
        "LYS": 11,
        "MET": 12,
        "PHE": 13,
        "PRO": 14,
        "SER": 15,
        "THR": 16,
        "TRP": 17,
        "TYR": 18,
        "VAL": 19,
    }

    pdbs = []
    for s in centers:
        prot_iis = s.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = s.atom_slice(prot_iis)
        pdbs.append(prot_bb)

    B = len(centers)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)

    for i, prot_bb in enumerate(pdbs):
        l = prot_bb.top.n_residues
        xyz = prot_bb.xyz.reshape(l, 4, 3)
        seq = [r.name for r in prot_bb.top.residues]

        S[i, :l] = np.asarray([aa2index[a] for a in seq], dtype=np.int32)
        X[i] = np.pad(
            xyz, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,)
        )

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0
    X = np.nan_to_num(X)

    prediction = model(X, S, mask, train=False, res_level=True)
    return prediction


class PocketPredictionWrap(base_analysis):
    """Analysis wrapper for pocket analysis using ligsite and
        pocket prediction using the geometric vector perceptron.
        This method balances selecting states with high pocket
        volumes and those predicted to have high pocket volumes
        in the future.

     Parameters
     ----------
    model : object, MQAModel
         An object of class MQA Model which specifies the architecture
         of the network
     nn_path : str
         A path specifying the location of the input model's index
         and dat files. This path is passed to load_checkpoint
     alpha : float, default = 0.6
         Defines how to weight predctive component with the current
         pocket volume. This is the percent weight that is given
         to the predictive component.
         i.e. r_i = (1 - alpha) * current_score + alpha * prediction_score
     pocket_reporter : object, default = None,
         How to rank pocket volumes. A specific object that calls
         `parse_pockets` must be supplied. If None are specified,
         will use `TopPockets`, which reports on all pocket volumes.
     grid_spacing : float, default = 0.1,
         The spacing for grid around the protein.
     probe_radius : float, default = 0.14,
         The radius of the grid point to probe for pocket elements.
     min_rank : int, default = 4,
         Minimum rank for defining a pocket element. Ranges from 1-7, 1
         being very shallow and 7 being a fully enclosed pocket element.
     min_cluster_size : int, default = 0,
         The minimum number of adjacent pocket elements to consider a
         true pocket. Trims pockets smaller than this size.
     n_cpus : int,
         The number of cpus to use for pocket analysis.
     build_full : bool,
         Flag to either analyze all structures or to continue previous
         analysis.
     atom_indices : array-like, or string, default=None,
         The atom indices to use for calculating pocket volumes. Can be
         supplied as a path to a numpy file or a list of indices.

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
        self,
        model,
        nn_path,
        alpha=0.6,
        pocket_reporter=None,
        grid_spacing=0.1,
        probe_radius=0.14,
        min_rank=4,
        min_cluster_size=0,
        n_cpus=1,
        build_full=True,
        atom_indices=None,
        **kwargs
    ):
        self.model = model
        self.nn_path = nn_path
        self.pocket_reporter = pocket_reporter
        if self.pocket_reporter is None:
            self.pocket_reporter = TopPockets(n_cpus=n_cpus)
        self.grid_spacing = grid_spacing
        self.probe_radius = probe_radius
        self.min_rank = min_rank
        self.min_cluster_size = min_cluster_size
        self.n_cpus = n_cpus
        self.build_full = build_full
        self.atom_indices = atom_indices
        if isinstance(self.atom_indices, (str)):
            try:
                self.atom_indices = np.loadtxt(self.atom_indices, dtype=int)
            except:
                self.atom_indices = np.array(np.load(self.atom_indices), dtype=int)
        self.pocket_func = partial(
            pockets.get_pockets,
            grid_spacing=grid_spacing,
            probe_radius=probe_radius,
            min_rank=min_rank,
            min_cluster_size=min_cluster_size,
            n_procs=n_cpus,
        )

    @property
    def class_name(self):
        return "PocketPredictionWrap"

    @property
    def config(self):
        return {
            "pocket_reporter": self.pocket_reporter,
            "grid_spacing": self.grid_spacing,
            "probe_radius": self.probe_radius,
            "min_rank": self.min_rank,
            "min_cluster_size": self.min_cluster_size,
            "n_cpus": self.n_cpus,
            "build_full": self.build_full,
            "atom_indices": self.atom_indices,
        }

    @property
    def analysis_folder(self):
        return "pocket_analysis"

    @property
    def base_output_name(self):
        return "composite_pocket_rank_per_state"

    def run(self):
        # determine if analysis was already done
        if os.path.exists(self.output_name):
            pass
        else:
            # get pdb filenames
            pdb_filenames = _get_filenames(self.msm_dir)
            # get the pdb centers
            centers = md.load(
                self.msm_dir + "/data/full_centers.xtc",
                top=self.msm_dir + "/prot_masses.pdb",
            )
            if self.atom_indices is not None:
                centers = centers.atom_slice(self.atom_indices)
            # optionally determine pockets of all structures
            if self.build_full:
                cmd = ["mkdir " + self.output_folder]
                _ = tools.run_commands(cmd)
                save_pocket_elements(
                    self.pocket_func,
                    centers,
                    pdb_filenames,
                    self.output_folder,
                    self.n_cpus,
                )
            # determine pockets of all non-processed states
            else:
                n_processed_states = len(glob.glob(self.output_folder + "/state*"))
                save_pocket_elements(
                    self.pocket_func,
                    centers[n_processed_states:],
                    pdb_filenames[n_processed_states:],
                    self.output_folder,
                    self.n_cpus,
                )
            # parses log files for pockets and save them
            # pockets contains pocket volumes per state
            pockets = self.pocket_reporter.parse_pockets(self.output_folder)

            # make a prediction
            predictions = make_pocket_predictions(self.model, self.nn_path, centers)

            # feature scale
            # other option is just to use predictions
            np.save(self.output_name, predictions)
