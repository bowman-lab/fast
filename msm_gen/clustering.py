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
from .. import tools
from ..base import base
from enspara import cluster
from enspara.util import array as ra
from functools import partial
from multiprocessing import Pool

#######################################################################
# code
#######################################################################


def load_trjs(n_procs=1, **kwargs):
    """Parallelize loading trajectories from msm directory."""
    # get filenames
    trj_filenames = np.sort(np.array(glob.glob("trajectories/*.xtc")))
    # parallelize load with **kwargs
    partial_load = partial(md.load, **kwargs)
    pool = Pool(processes=n_procs)
    trjs = pool.map(partial_load, trj_filenames)
    pool.terminate()
    return trjs


def _save_states(centers_info):
    """Save centers found within a single trajectory"""
    # get state, conf, and frame info. Also the filename and topology.
    states = centers_info['state']
    confs = centers_info['conf']
    frames = centers_info['frame']
    trj_filename = centers_info['trj_filename'][0]
    topology = centers_info['topology'][0]
    # load trajectories
    trj = md.load('./trajectories/' + trj_filename, top=topology)
    trj_full = md.load(
        './trajectories_full/' + trj_filename, top="restart.gro")
    # save each of the states within the trajectory
    for num in range(len(states)):
        # save center after processing
        pdb_filename = "./centers_masses/State" + ('%06d' % states[num]) + \
            '-' + ('%02d' % confs[num]) + ".pdb"
        center = trj[frames[num]]
        center.save_pdb(pdb_filename)
        # save center for restarting simulations
        pdb_filename = "./centers_restarts/State" + ('%06d' % states[num]) + \
            '-' + ('%02d' % confs[num]) + ".gro"
        center = trj_full[frames[num]]
        center.save_gro(pdb_filename)
    return


def unique_states(assignments):
    """Search assignments array and return a list of the state ids
    within.
    """
    state_nums = np.unique(assignments)
    state_nums = state_nums[np.where(state_nums != -1)]
    return state_nums


def save_states(
        assignments, distances, state_nums=None,
        largest_center=np.inf, n_confs=1, n_procs=1):
    """Saves specified state-numbers by searching through the
    assignments and distances. Can specify a largest distance to a
    cluster center to save computational time searching for min
    distances. If multiple conformations are saved, the center is saved
    as conf-0 and the rest are random conformations.
    """
    t0 = time.time()
    if state_nums is None:
        state_nums = unique_states(assignments)
    trj_filenames = np.sort(
        np.array(
            [s.split("/")[-1] for s in glob.glob("./trajectories/*.xtc")]))
    topology = "prot_masses.pdb"
    # reduce the number of conformations to search through
    reduced_iis = np.where((distances > -0.1)*(distances < largest_center))
    reduced_assignments = assignments[reduced_iis]
    reduced_distances = distances[reduced_iis]
    centers_location = []
    for state in state_nums:
        state_iis = np.where(reduced_assignments == state)
        nconfs_in_state = len(state_iis[0])
        if nconfs_in_state >= n_confs:
            center_picks = np.array([0])
            if n_confs > 1:
                center_picks = np.append(
                    center_picks,
                    np.random.choice(
                        range(1, nconfs_in_state), n_confs-1, replace=False))
        else:
            center_picks = np.array([0])
            center_picks = np.append(
                center_picks, np.random.choice(nconfs_in_state, n_confs - 1))
        state_centers = np.argsort(reduced_distances[state_iis])[center_picks]
        # Obtain information on conformation locations within trajectories
        trj_locations = reduced_iis[0][state_iis[0][state_centers]]
        frame_nums = reduced_iis[1][state_iis[0][state_centers]]
        for conf_num in range(n_confs):
            trj_num = trj_locations[conf_num]
            centers_location.append(
                (
                    state, conf_num, trj_num,
                    frame_nums[conf_num], trj_filenames[trj_num],
                    topology))
    if type(topology) == str:
        centers_location = np.array(
            centers_location, dtype=[
                ('state', 'int'), ('conf', 'int'), ('trj_num', 'int'),
                ('frame', 'int'), ('trj_filename', np.str_, 500),
                ('topology', np.str_, 500)])
    unique_trjs = np.unique(centers_location['trj_num'])
    partitioned_centers_info = []
    for trj in unique_trjs:
        partitioned_centers_info.append(
            centers_location[np.where(centers_location['trj_num'] == trj)])
    logging.info("  Saving states!")
    pool = Pool(processes=n_procs)
    pool.map(_save_states, partitioned_centers_info)
    pool.terminate()
#    gc.collect()
    t1 = time.time()
    logging.info("    Finished in "+str(t1-t0)+" sec")
    return


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
    """
    def __init__(
            self, base_struct, base_clust_obj=None, atom_indices=None,
            build_full=True, n_procs=1):
        # determine base_struct
        self.base_struct = base_struct
        if type(base_struct) is md.Trajectory:
            self.base_struct_md = base_struct
        else:
            self.base_struct_md = md.load(base_struct)
        # determine base clustering object
        if base_clust_obj is None:
            self.base_clust_obj = cluster.KCenters(
                md.rmsd, cluster_radius=1.0)
        else:
            self.base_clust_obj = base_clust_obj
        # determine atom indices
        self.atom_indices = atom_indices
        if type(atom_indices) is str:
            self.atom_indices_vals = np.loadtxt(atom_indices, dtype=int)
        else:
            self.atom_indices_vals = atom_indices
        self.n_procs = n_procs
        self.build_full = build_full

    def check_clustering(self, msm_dir, gen_num, n_kids, verbose=True):
        correct_clustering = True
        total_assignments = (gen_num + 1) * n_kids
        assignments = ra.load(msm_dir + '/data/assignments.h5')
        n_assignments = len(assignments) 
        if total_assignments != n_assignments:
            correct_clustering = False
            logging.info(
                "inconsistent number of trajectories between assignments and data!")
        unique_states = np.unique(assignments)
        n_states = len(unique_states)
        saved_states = glob.glob(msm_dir + '/centers_masses/*-00.pdb')
        if n_states != len(saved_states):
            correct_clustering = False
            logging.info(
                "number of states saved does not match those found in assignments!")
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
        }

    def run(self):
        # load and concat trjs
        trjs = load_trjs(n_procs=self.n_procs, top=self.base_struct_md)
        trj_lengths = [len(t) for t in trjs]
        trjs = md.join(trjs)
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
        centers.save_xtc("./data/centers.xtc")
        full_centers = trjs[self.base_clust_obj.center_indices_]
        full_centers.save_xtc("./data/full_centers.xtc")
        # save states
        n_states = len(self.base_clust_obj.center_indices_)
        unique_states = np.arange(n_states)
        if init_centers is not None:
            unique_states = unique_states[-(n_states-len(init_centers)):]
        np.save("./data/unique_states.npy", unique_states)
        save_states(
            assignments, distances, state_nums=unique_states,
            n_procs=self.n_procs,
            largest_center=self.base_clust_obj.cluster_radius)
