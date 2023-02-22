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
from enspara.util import array as ra
from enspara.util.load import load_as_concatenated
from multiprocessing import Pool
from ..base import base


#######################################################################
# code
#######################################################################


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _save_states(centers_info):
    """Save centers found within a single trajectory"""
    # get state, conf, and frame info. Also the filename and topology.
    states = centers_info['state']
    confs = centers_info['conf']
    frames = centers_info['frame']
    trj_filename = centers_info['trj_filename'][0]
    save_routine = centers_info['save_routine'][0]
    msm_dir = centers_info['msm_dir'][0]
    if save_routine == 'full':
        save_masses = True
        save_restarts = True
    elif save_routine == 'masses':
        save_masses = True
        save_restarts = False
    elif save_routine == 'restarts':
        save_masses = False
        save_restarts = True
    else:
        raise
    # load structs trajectories
    if save_masses:
        top = md.load(msm_dir+"/prot_masses.pdb")
        trj = md.load(
            msm_dir + '/trajectories/' + trj_filename,
            top=top)
    if save_restarts:
        trj_full = md.load(
            msm_dir + '/trajectories_full/' + trj_filename,
            top=msm_dir + "/restart.gro")
    for num in range(len(states)):
        if save_masses:
            # save center after processing
            pdb_filename = msm_dir + \
                "/centers_masses/state%06d-%02d.pdb" % \
                (states[num], confs[num])
            center = trj[frames[num]]
            center.superpose(top)
            center.save_pdb(pdb_filename)
        if save_restarts:
            # save center for restarting simulations
            pdb_filename = msm_dir + \
                "/centers_restarts/state%06d-%02d.gro" % \
                (states[num], confs[num])
            center = trj_full[frames[num]]
            center.save_gro(pdb_filename)
    if save_masses:
        del trj
    if save_restarts:
        del trj_full
    return


def save_states(
        assignments, distances, state_nums=None, save_routine='full',
        largest_center=np.inf, n_confs=1, n_procs=1, msm_dir='.'):
    """Saves specified state-numbers by searching through the
    assignments and distances and pulling single frames from
    trajectories. This is a special tailored helper function that has a
    directory structure hard-coded in. Can specify a largest distance to a
    cluster center to save computational time searching for min
    distances. If multiple conformations are saved, the center is saved
    as conf-0 and new conformations are sampled from the cluster.

    Inputs
    ----------
    assignments : array, shape=(n_trajectories, n_frames),
        Assigned cluster for each frame in each trajectory.
    distances : array, shape=(n_trajectories, n_frames),
        The distance to the cluster center for each frame
        in each trajectory.
    state_nums : array, shape=(n_states, ), default=None,
        The specific state numbers for saving. If None are supplied,
        will save every state.
    save_routine : str, default='full',
        The routine for saving states, either 'full', 'masses', or
        'restarts'. 'masses' will only save the processed cluster centers,
        'restarts' will only save the full system centers, and 'full'
        will save both.
    largest_center : float, default=np.inf,
        The largest expected distance from any frame to a cluster center.
        Specifying a small number can save in computational time. Defaults
        to np.inf, which will consider every frame when finding cluster
        centers.
    n_confs : int, default=1,
        The number of representative conformations to save of each cluster
        center. The first conformation is always the cluster center, and
        subsequent conformations are sampled randomly from frames clustered.
    n_procs : int, default=1,
        The number of processes to use when saving states.
    msm_dir : str, default='.',
        Location of the msm directory containing trajectories and folders
        for saving states.
    """
    if state_nums is None:
        try:
            state_nums = np.unique(assignments)
        except:
            state_nums = np.unique(np.concatenate(assignments))
    trj_filenames = np.sort(
        np.array(
            [
                s.split("/")[-1]
                for s in glob.glob(msm_dir + "/trajectories/*.xtc")]))
    topology = "prot_masses.pdb"
    # reduce the number of conformations to search through
    reduced_iis = ra.where((distances > -0.1)*(distances < largest_center))
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
                    save_routine, msm_dir))
    if type(topology) == str:
        centers_location = np.array(
            centers_location, dtype=[
                ('state', 'int'), ('conf', 'int'), ('trj_num', 'int'),
                ('frame', 'int'), ('trj_filename', np.str_, 800),
                ('save_routine', np.str_, 10), ('msm_dir', np.str_, 800)])
    unique_trjs = np.unique(centers_location['trj_num'])
    partitioned_centers_info = []
    for trj in unique_trjs:
        partitioned_centers_info.append(
            centers_location[np.where(centers_location['trj_num'] == trj)])
    if n_procs == 1:
        for pci in partitioned_centers_info:
            _save_states(pci)
    else:
        with Pool(processes=n_procs) as pool:
            pool.map(_save_states, partitioned_centers_info)
#        pool = Pool(processes=n_procs)
#        pool.map(_save_states, partitioned_centers_info)
#        pool.terminate()
    return


class SaveWrap(base):
    """Save states wrapping object

    Parameters
    ----------
    save_routine : str, default='full',
        The type of states to save. Three options: 1) 'masses' saves
        only in the centers_masses, 2) 'restarts' saves only the
        restarts, and 3) 'full' saves both.
    centers : str, default='auto',
        The indicator for the set of centers to save. Four options:
        1) 'all' will save every center, 2) 'none' will not save any centers,
        3) 'restarts' will only save the centers to use for
        restarting simulations, and 4) 'auto' will only save new states
        that were discovered in previous round of sampling.
    gen_num : int, default=0,
        The generation number of adaptive sampling. Only used if
        centers is set to 'restarts'.
    largest_center : float, default=np.inf,
        The largest distance to a cluster center expected. Can be used
        to speed up searching for cluster centers. A reasonable value
        if the distance cutoff used for clustering.
    save_xtc_centers : bool, default=False,
        Optionally save centers as an xtc in data.
    n_procs : int, default=1,
        The number of processes to use when saving states.
    """
    def __init__(
            self, save_routine='full', centers='auto',
            gen_num=0, largest_center=np.inf, save_xtc_centers=False,
            n_procs=1):
        self.save_routine = save_routine
        self.centers = centers
        self.gen_num = gen_num    
        self.largest_center = largest_center
        self.save_xtc_centers = save_xtc_centers
        self.n_procs = n_procs

    @property
    def class_name(self):
        return "SaveWrap"

    @property
    def config(self):
        return {
            'save_routine': self.save_routine,
            'centers': self.centers,
            'gen_num': self.gen_num,
            'largest_center': self.largest_center,
            'n_procs': self.n_procs,
            'save_xtc_centers': self.save_xtc_centers,
            }

    def check_save_states(self, msm_dir):
        assigns = ra.load(msm_dir + '/data/assignments.h5')
        unique_states = np.unique(assigns)
        n_states = unique_states.shape[0]
        correct_save = True
        save_masses = False
        save_restarts = False
        if (self.save_routine == 'masses') or (self.save_routine == 'full'):
            save_masses = True
        if (self.save_routine == 'restarts') or (self.save_routine == 'full'):
            save_restarts = True
        if (self.centers == 'none') or (self.centers == 'restarts'):
            pass
        else:
            if save_masses:
                n_masses = len(glob.glob(msm_dir + '/centers_masses/*.pdb'))
                if n_masses != n_states:
                    correct_save = False
            if save_restarts:
                n_restarts = len(glob.glob(msm_dir + '/centers_restarts/*.gro'))
                if n_restarts != n_states:
                    correct_save = False
        return correct_save

    def run(self, msm_dir='.'):
        if self.centers != 'none':
            assignments = ra.load(msm_dir + "/data/assignments.h5")
            distances = ra.load(msm_dir + "/data/distances.h5")
            if self.centers == 'auto':
                state_nums = np.load(msm_dir + "/data/unique_states.npy")
            elif self.centers == 'all':
                state_nums = None
            elif self.centers == 'restarts':
                states_to_simulate_file = \
                    msm_dir + "/rankings/states_to_simulate_gen" + \
                    str(self.gen_num) + ".npy"
                state_nums = np.load(states_to_simulate_file)
            save_states(
                assignments, distances, state_nums=state_nums,
                n_procs=self.n_procs, largest_center=self.largest_center,
                save_routine=self.save_routine, msm_dir=msm_dir)
        if self.save_xtc_centers:
            center_filenames = np.sort(glob.glob("%s/centers_masses/*.pdb" % msm_dir))
            trj_lengths, xyzs = load_as_concatenated(center_filenames, processes=self.n_procs)
            centers = md.Trajectory(xyzs, topology=md.load("%s/prot_masses.pdb" % msm_dir).top)
            centers.save_xtc("%s/data/full_centers.xtc" % msm_dir)
