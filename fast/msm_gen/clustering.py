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
import logging
import mdtraj as md
import numpy as np
import os
import time
from .save_states import save_states
from .. import tools
from ..base import base
from enspara import cluster
from enspara.util import array as ra
from enspara.util.load import load_as_concatenated
from functools import partial
from multiprocessing import Pool


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#######################################################################
# code
#######################################################################


def load_trjs(trj_filenames, n_procs=1, **kwargs):
    """Parallelize loading trajectories from msm directory."""
    # get filenames
    trj_filenames_test = np.array(
        [
            os.path.abspath(f) 
            for f in np.sort(np.array(glob.glob("trajectories/*.xtc")))])
    t0 = time.time()
    diffs = np.setdiff1d(trj_filenames, trj_filenames_test)
    while diffs.shape[0] != 0:
        t1 = time.time()
        logging.info(
            'waiting on nfs. missing %d files (%0.2f s)' % \
            (trj_filenames.shape[0]-trj_filenames_test.shape[0], t1-t0))
        time.sleep(15)
        _ = tools.run_commands('ls trajectories/*.xtc')
        trj_filenames_test = np.array(
            [
                os.path.abspath(f) 
                for f in np.sort(np.array(glob.glob("trajectories/*.xtc")))])
        diffs = np.setdiff1d(trj_filenames, trj_filenames_test)
    # parallelize load with **kwargs
    partial_load = partial(md.load, **kwargs)
    pool = Pool(processes=n_procs)
    trjs = pool.map(partial_load, trj_filenames)
    pool.terminate()
    return trjs


class ClusterWrap(base):
    """Clustering wrapper function

    Parameters
    ----------
    base_struct : str or md.Trajectory,
        A structure with the same topology as the trajectories to load.
    base_clust_obj : enspara.msm.MSM,
        enspara object to use with clustering.
    atom_indices : str or list,
        The atom indices of the base_struct to cluster with.
    build_full : bool, default = True,
        Flag for building from scratch.
    n_procs : int, default = 1,
        The number of processes to use when loading, clustering and
        saving conformations.
    mem_efficient : bool, default=False,
        optionally save memory by not loading all of the atoms of trajectories.
        Saving full cluster centers should be performed by save_states if this
        is set to True.
    """
    def __init__(
            self, base_struct, base_clust_obj=None, atom_indices=None,
            build_full=True, n_procs=1, mem_efficient=False):
        # determine base_struct
        self.base_struct = base_struct
        if type(base_struct) is md.Trajectory:
            self.base_struct_md = base_struct
        else:
            self.base_struct_md = md.load(base_struct)
        # determine base clustering object
        if base_clust_obj is None:
            self.base_clust_obj = cluster.KCenters(
                metric=md.rmsd, cluster_radius=1.0)
        else:
            self.base_clust_obj = base_clust_obj
        # determine atom indices
        self.atom_indices = atom_indices
        if type(atom_indices) is str:
            try:
                self.atom_indices_vals = np.loadtxt(atom_indices, dtype=int)
            except ValueError:
                print("\n")
                logging.warning(
                    ' Atom indices for clustering are not integers!'
                    ' Attempting to convert to integers\n')
                non_int_vals = np.loadtxt(atom_indices)
                self.atom_indices_vals = np.array(non_int_vals, dtype=int)
                # ensure no conversion error
                diffs = self.atom_indices_vals - non_int_vals
                assert np.all(diffs == np.zeros(non_int_vals.shape[0]))
        else:
            self.atom_indices_vals = atom_indices
        self.n_procs = n_procs
        self.build_full = build_full
        self.trj_filenames = None
        self.mem_efficient = mem_efficient

    def check_clustering(self, msm_dir, gen_num, n_kids, verbose=True):
        correct_clustering = True
        total_assignments = (gen_num + 1) * n_kids
        assignments = ra.load(msm_dir + '/data/assignments.h5')
        n_assignments = len(assignments) 
        if total_assignments != n_assignments:
            correct_clustering = False
            logging.info(
                "inconsistent number of trajectories between assignments and data!")
        return correct_clustering

    @property
    def class_name(self):
        return "ClusterWrap"

    @property
    def config(self):
        return {
        'base_struct': self.base_struct,
        'base_clust_obj': self.base_clust_obj,
        'atom_indices': self.atom_indices,
        'build_full': self.build_full,
        'n_procs': self.n_procs,
        'trj_filenames': self.trj_filenames,
        'mem_efficient': self.mem_efficient,
        }

    def set_filenames(self, msm_dir):
        self.trj_filenames = np.sort(
            np.array(glob.glob(msm_dir + "/trajectories/*.xtc")))
        return

    def run(self):
        # load and concat trjs
        if self.mem_efficient:
            trj_lengths, xyzs = load_as_concatenated(
                filenames=self.trj_filenames, processes=self.n_procs,
                top=self.base_struct_md, atom_indices=self.atom_indices_vals)
            trjs_sub = md.Trajectory(
                xyzs, self.base_struct_md.atom_slice(self.atom_indices_vals).topology)
        else:
            trj_lengths, xyzs = load_as_concatenated(
                filenames=self.trj_filenames, processes=self.n_procs,
                top=self.base_struct_md)
            trjs = md.Trajectory(xyzs, self.base_struct_md.topology)
            trjs_sub = trjs.atom_slice(self.atom_indices_vals)
        # determine if rebuilding all msm stuff
        if self.build_full:
            base_struct_centers = self.base_struct_md.atom_slice(
                self.atom_indices_vals)
            base_struct_centers.save_pdb("./centers.pdb")
            self.base_struct_md.save_pdb("./prot_masses.pdb")
            init_centers = None
        else:
            init_centers = md.load(
                "./data/centers.xtc", top="./centers.pdb")
        # fit data with base clustering object
        self.base_clust_obj.fit(
            trjs_sub, init_centers=init_centers)
        center_indices, distances, assignments, centers = \
            self.base_clust_obj.result_.partition(trj_lengths)
        # save data
        ra.save("./data/assignments.h5", assignments)
        ra.save("./data/distances.h5", distances)
        trjs_sub = trjs_sub[self.base_clust_obj.center_indices_]
        trjs_sub.superpose(trjs_sub[0])
        trjs_sub.save_xtc(
            "./data/centers.xtc")
        if not self.mem_efficient:
            full_centers = trjs[self.base_clust_obj.center_indices_]
            full_centers.superpose(self.base_struct_md)
            full_centers.save_xtc("./data/full_centers.xtc")
        # save states
        n_states = len(self.base_clust_obj.center_indices_)
        unique_states = np.arange(n_states)
        if init_centers is not None:
            unique_states = unique_states[-(n_states-len(init_centers)):]
        np.save("./data/unique_states.npy", unique_states)
