# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential

#######################################################################
# imports
#######################################################################

import glob
import logging
import mdtraj as md
import numpy as np
import time
from multiprocessing import Pool

#######################################################################
# code
#######################################################################

def unique_states(assignments):
    """Search assignments array and return a list of the state ids
    within.
    """
    state_nums = np.unique(assignments)
    state_nums = state_nums[np.where(state_nums != -1)]
    return state_nums


def _save_states(centers_info):
    """Save centers found within a single trajectory"""
    # get state, conf, and frame info. Also the filename and topology.
    states = centers_info['state']
    confs = centers_info['conf']
    frames = centers_info['frame']
    trj_filename = centers_info['trj_filename'][0]
    topology = centers_info['topology'][0]
    # load structs trajectories
    trj = md.load('./trajectories/' + trj_filename, top=topology)
    trj_full = md.load(
        './trajectories_full/' + trj_filename, top="restart.gro")
    for num in range(len(states)):
        # save center after processing
        pdb_filename = "./centers_masses/state" + ('%06d' % states[num]) + \
            '-' + ('%02d' % confs[num]) + ".pdb"
        center = trj[frames[num]]
        center.save_pdb(pdb_filename)
        # save center for restarting simulations
        pdb_filename = "./centers_restarts/state" + ('%06d' % states[num]) + \
            '-' + ('%02d' % confs[num]) + ".gro"
        center = trj_full[frames[num]]
        center.save_gro(pdb_filename)
    return


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
                ('frame', 'int'), ('trj_filename', np.str_, 800),
                ('topology', np.str_, 800)])
    unique_trjs = np.unique(centers_location['trj_num'])
    partitioned_centers_info = []
    for trj in unique_trjs:
        partitioned_centers_info.append(
            centers_location[np.where(centers_location['trj_num'] == trj)])
    logging.info("  Saving states!")
    if n_procs == 1:
        for pci in partitioned_centers_info:
            _save_states(pci)
    else:
        pool = Pool(processes=n_procs)
        pool.map(_save_states, partitioned_centers_info)
        pool.terminate()
    t1 = time.time()
    logging.info("    Finished in %0.2f seconds" % (t1-t0))
    return

