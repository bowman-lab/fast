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
import itertools
import logging
import mdtraj as md
import numpy as np
import os
import pickle
import scipy.io
import subprocess as sp
import time
from . import rankings
from .. import tools
from ..base import base
from ..exception import DataInvalid, MissingData
from ..msm_gen import SaveWrap
from ..submissions import slurm_subs
from ..submissions import lsf_subs
from enspara.msm import builders, MSM
from enspara.util import array as ra
from functools import partial


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#######################################################################
# code
#######################################################################


def _setup_directories(output_dir):
    """Setup adaptive sampling directory structure"""
    # if output dir already exists raise an error
    if os.path.exists(output_dir):
        raise DataInvalid('output directory already exists!')
    msm_dir = output_dir + "/msm"
    cmd1 = 'mkdir ' + output_dir
    cmd2 = 'mkdir ' + msm_dir
    cmd3 = 'mkdir ' + msm_dir + '/data'
    cmd4 = 'mkdir ' + msm_dir + '/rankings'
    cmd5 = 'mkdir ' + msm_dir + '/trajectories'
    cmd6 = 'mkdir ' + msm_dir + '/trajectories_full'
    cmd7 = 'mkdir ' + msm_dir + '/centers_masses'
    cmd8 = 'mkdir ' + msm_dir + '/centers_restarts'
    cmd9 = 'mkdir ' + msm_dir + '/submissions'
    cmds = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9]
    out = tools.run_commands(cmds)
    return msm_dir


def _gen_initial_sims(
        base_dir, initial_struct, trj_obj, n_kids, q_check_obj):
    """Runs the first round of adaptive sampling. Currently runs
    simulations from a single structure.

    Inputs
    ----------
    base_dir : str,
        The base adaptive sampling directory that will contain gen
        directories and the msm directory.
    initial_struct : str or md.Trajectory,
        The initial structure to start a swarn of sims from.
    trj_obj : object,
        Simulation object used for simulations. See Gromax.
    n_kids : int,
        Number of children gen will have.
    q_check_obj : object,
        Queueing system wrapper to determine if simulations are
        still running.
    """
    t0 = time.time()
    # generate initial gen directory
    gen0_dir = base_dir + '/gen0'
    cmd = 'mkdir ' + gen0_dir
    _ = tools.run_commands(cmd)
    # Spawn simulations
    pids = []
    for kid in range(n_kids):
        # generate kid directory
        kid_dir = gen0_dir + '/kid'+str(kid)
        cmd = 'mkdir ' + kid_dir
        _ = tools.run_commands(cmd)
        # submit job based on trj_obj and retain pid
        pid = trj_obj.run(initial_struct, kid_dir)
        pids.append(pid)
        # wait for a job to finish if maximum number of simulations are
        #still running
        q_check_obj.wait_for_pids(np.array(pids))
    # gather all pids and wait for every simulation to finish
    pids = np.array(pids)
    q_check_obj.wait_for_pids(pids, wait_for_all=True)
    t1 = time.time()
    logging.info("simulations took %0.4f seconds" % (t1-t0))
    return pids


def _prop_sims(base_dir, trj_obj, gen_num, q_check_obj, new_states):
    """Runs the subsequent rounds of adaptive sampling. Looks for
    states in specified directory structure and uses them for
    respawning simulations.

    Inputs
    ----------
    base_dir : str,
        The base adaptive sampling directory that will contain gen
        directories and the msm directory.
    trj_obj : object,
        Simulation object used for simulations. See Gromax.
    gen_num : int,
        The generation of sampling to propagate new simulations.
    q_check_obj : object,
        Queueing system wrapper to determine if simulations are
        still running.
    new_states : list,
        List of state numbers to respawn simulations from
    """
    t0 = time.time()
    # generate gen directory
    gen_dir = base_dir + '/gen' + str(gen_num)
    cmd = 'mkdir ' + gen_dir
    _ = tools.run_commands(cmd)
    pids = []
    # propagate simulations
    for kid in range(len(new_states)):
        # generate kid directory
        kid_dir = gen_dir + '/kid'+str(kid)
        cmd = 'mkdir ' + kid_dir
        _ = tools.run_commands(cmd)
        # run simulation and gather pid
        filename = base_dir + '/msm/centers_restarts/state' + \
            ('%06d' % new_states[kid]) + '-00.gro'
        pid = trj_obj.run(filename, kid_dir)
        pids.append(pid)
        # wait for a job to finish if maximum number of simulations are
        #still running
        q_check_obj.wait_for_pids(np.array(pids))
    # gather all pids and wait for every simulation to finish
    pids = np.array(pids)
    q_check_obj.wait_for_pids(pids, wait_for_all=True)
    t1 = time.time()
    logging.info("simulations took %0.4f seconds" % (t1 - t0))
    return pids
    

def _move_trjs(gen_dir, msm_dir, gen_num, n_kids):
    """Move finished trajectories to MSM directory.

    Inputs
    ----------
    gen_dir : str,
        Generation directory containing simulations to move.
    msm_dir : str,
        MSM directory where analysis is performed.
    gen_num : int,
        Generation number.
    n_kids : int,
        Number of kids expected in gen directory.
    """
    # iterate over potential kids
    for kid in range(n_kids):
        # trj to move trajectories
        try:
            # specify directroy and file names
            kid_dir = gen_dir + "/kid" + str(kid)
            output_full = msm_dir + '/trajectories_full' + '/trj_gen' + \
                ('%03d' % gen_num) + '_kid' + ('%03d' % kid) + '.xtc'
            output_masses = msm_dir + '/trajectories' + '/trj_gen' + \
                ('%03d' % gen_num) + '_kid' + ('%03d' % kid) + '.xtc'
            cmds = []
            if os.path.exists(output_full):
                logging.info("file '%s' exists! skipping move." % output_full)
            else:
                cmds.append(
                    'mv ' + kid_dir + '/frame0_aligned.xtc ' + output_full)
            if os.path.exists(output_masses):
                logging.info("file '%s' exists! skipping move." % output_masses)
            else:
                cmds.append(
                    'mv ' + kid_dir + '/frame0_masses.xtc ' +output_masses)
            if len(cmds) > 0:
                out = tools.run_commands(cmds)
        # give sad-face expression
        except:
            raise MissingData(
                'trajectory from gen %03d and kid %03d was not found.' %
                (gen_num, kid),
                'Simulation may have crahsed!')
    return


def _pickle_submit(
        msm_dir, base_obj, sub_obj, q_check_obj, gen_num, base_name):
    """Helper function for pickling an object and submitting it to run.

    Inputs
    ----------
    msm_dir : str,
        MSM directory where analysis is performed.
    base_obj : object,
        The object to pickle and submit. Must have `run` as a class
        function.
    sub_obj : object,
        Submission wrapper object.
    q_check_obj : object,
        Queueing system wrapper to determine if submission is still
        running.
    gen_num : int,
        Generation number.
    base_name : str,
        The base name of the job being submitted. Used to name the
        output files.
    """
    # pickle object
    base_pickle = msm_dir + "/" + base_name + ".pkl"
    pickle.dump(base_obj, open(base_pickle, "wb"))
    # determine home dir and switch to msm dir
    home_dir = os.path.abspath("./")
    os.chdir(msm_dir)
    # write python file to open and run pickle object
    f = open(base_name +".py", "w")
    f.write(
        'import pickle\n\n' + \
        'c = pickle.load(open("' + base_pickle + '", "rb"))\n' + \
        'c.run()')
    f.close()
    # generate python run commands for submission
    cmd0 = 'sync\n'
    cmd1 = 'python ' + base_name + '.py'
    cmds = [cmd0, cmd1]
    base_submission = base_name + '_submission'
    # submit and wait for job to finish
    pid = sub_obj.run(cmds, output_name=base_submission)
    q_check_obj.wait_for_pids([pid], wait_for_all=True)
    # clean up submission
    sub_script_name = q_check_obj.get_submission_names(pid)[0]
    sub_output = 'submissions/' + base_name + '_gen' + \
        ('%03d' % gen_num) + '.out'
    base_pickle_output = 'submissions/' + base_name + '_gen' + \
        ('%03d' % gen_num) + '.pkl'
    base_python_output = 'submissions/' + base_name + '_gen' + \
        ('%03d' % gen_num) + '.py'
    base_sub_output = 'submissions/' + base_submission + '_gen' + \
        ('%03d' % gen_num)
    cmd1 = 'mv ' + sub_script_name + ' ' + sub_output + ' --backup=numbered'
    cmd2 = 'mv ' + base_pickle + ' ' + base_pickle_output + \
        ' --backup=numbered'
    cmd3 = 'mv ' + base_name + ".py" + ' ' + base_python_output + \
        ' --backup=numbered'
    cmd4 = 'mv ' + base_submission + ' ' + base_sub_output + \
        ' --backup=numbered'
    cmds = [cmd1, cmd2, cmd3, cmd4]
    _ = tools.run_commands(cmds)
    # change directory back to original
    os.chdir(home_dir)
    return


def _determine_gen(output_dir, ignore_error=False):
    """Determines the current generation number."""
    # determine number of gen folders
    n_gen_folders = len(glob.glob(output_dir+'/gen*'))
    # gets completed sims in msm dir
    trj_names = glob.glob(output_dir + '/msm/trajectories/*.xtc')
    # sorts by unique gen number
    trj_gen_nums = np.array(
        [int(n.split("gen")[-1].split("_")[0]) for n in trj_names])
    # error check that completed sims match number of gen folders
    n_trj_gens = len(np.unique(trj_gen_nums))
    if (n_gen_folders != n_trj_gens) and not ignore_error:
        raise DataInvalid(
            'The number of generations are not consistent with the number ' + \
            'of trajectories. Maybe a simulation crashed?')
    # subtract 1 for 0 based counting
    gen_num = int(np.max([n_trj_gens, n_gen_folders]) - 1)
    return gen_num


def _prop_msm(msm_dir, msm_obj):
    """Propagate MSM files."""
    t0 = time.time()
    # load assignments and build MSM
    assignments = ra.load(msm_dir + '/data/assignments.h5')
    msm_obj.fit(assignments)
    # write counts, probs, and popoulations (if applicable)
    scipy.io.mmwrite(msm_dir + '/data/tcounts.mtx', msm_obj.tcounts_)
    scipy.io.mmwrite(msm_dir + '/data/tprobs.mtx', msm_obj.tprobs_)
    if msm_obj.eq_probs_ is not None:
        np.save(msm_dir + '/data/populations.npy', msm_obj.eq_probs_)
    t1 = time.time()
    logging.info("building MSM took %0.4f seconds" % (t1-t0))
    return msm_obj


def _move_cluster_data(msm_dir, rebuild_num, analysis_obj=None):
    """Move current cluster data to old directory. For rebuilding whole
    MSM and analysis

    Inputs
    ----------
    msm_dir : str,
        MSM directory where analysis is performed.
    rebuild_num : int,
        The rebuild number, i.e. If this is gen 6 and MSMs are being
        rebuild every 2 gens, this would be rebuild number 2 (0 based).
    analysis_obj : object,
        The object used for analysis. This object may contain
        information of a folder that also needs to be moved.
    """
    # define old directory to move into. mkdir if first rebuild.
    old_dir = msm_dir + '/old'
    if rebuild_num == 0:
        try:
            cmd = 'mkdir ' + old_dir
            _ = tools.run_commands(cmd)
        except:
            pass
    # move data and centers
    backup = ' --backup=numbered'
    cmd1 = 'mv ' + msm_dir + '/data ' + old_dir + '/data' + \
        str(rebuild_num) + backup
    cmd2 = 'mv ' + msm_dir + '/centers_masses ' + old_dir + \
        '/centers_masses' + str(rebuild_num) + backup
    cmd3 = 'mv ' + msm_dir + '/centers_restarts ' + old_dir + \
        '/centers_restarts' + str(rebuild_num) + backup
    # rebuild directories
    cmd4 = 'mkdir ' + msm_dir + '/data'
    cmd5 = 'mkdir ' + msm_dir + '/centers_masses'
    cmd6 = 'mkdir ' + msm_dir + '/centers_restarts'
    cmds = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6]
    # if applicable, move analysis folder
    if hasattr(analysis_obj, "output_folder"):
        base_folder = analysis_obj.output_folder.split("/")[-1]
        cmd = 'mv ' + analysis_obj.output_folder + ' ' + old_dir + \
            "/" + base_folder + str(rebuild_num)
        cmds.append(cmd)
    # run move commands
    _ = tools.run_commands(cmds)
    return


def _perform_analysis(
        analysis_obj, msm_dir, gen_num, sub_obj, q_check_obj, update_data):
    """Performs analysis of cluster centers.

    Inputs
    ----------
    analysis_obj : object,
        The object used for analysis.
    msm_dir : str,
        MSM directory where analysis is performed.
    gen_num : int,
        Generation number.
    sub_obj : object,
        Submission wrapper object.
    q_check_obj : object,
        Queueing system wrapper to determine if submission is still
        running.
    update_data : bool,
        Flag for rebuilding whole analysis or analyzing a subset of
        structures.
    """
    t0 = time.time()
    # determine if there is an analysis object
    if analysis_obj is None:
        state_rankings = None
    else:
        # set the objects output
        analysis_obj.set_output(msm_dir, gen_num)
        # optionally set rebuild or continue analysis
        if hasattr(analysis_obj, 'build_full'):
            analysis_obj.build_full = update_data
        # if the output doesn't exists, pickle submit analysis
        if not os.path.exists(analysis_obj.output_name):
            _pickle_submit(
                msm_dir, analysis_obj, sub_obj,
                q_check_obj, gen_num, 'analysis')
        # get rankings
        state_rankings = analysis_obj.state_rankings
        # check that everything went well
        # number of state rankings should be equal to number of state
        # in the assignments
        n_states_ranked = len(state_rankings)
        n_states = len(np.unique(ra.load(msm_dir + '/data/assignments.h5')))
        if n_states_ranked != n_states:
            raise DataInvalid(
                'The number of state rankings does not match the number ' + \
                'of states in the assignments! Analysis may have failed!')
    t1 = time.time()
    logging.info("analysis took %0.4f seconds" %(t1-t0))
    return state_rankings


def push_forward(s, num=0):
    s_out = s.split("\n")
    s_pushed = "\n".join(
        ["".join(itertools.repeat(" ", num)) + l for l in s_out])
    return s_pushed


class AdaptiveSampling(base):
    """Performs adaptive sampling

    Parameters
    ----------
    initial_state : str or MDTraj object,
        The starting structure for adaptive sampling.
    n_gens : int, default=1,
        The number of generations of sampling to perform.
    n_kids : int, default=1,
        The number of simulations per generation of adaptive sampling.
    sim_obj : object, default=None,
        An object that can run simulations. Currently supported within
        this package are Gromacs and Upside wrappers.
    cluster_obj : object, default=None,
        A cluster wrapper that dictates how simulations are clustered.
    save_state_obj : object, default=None,
        Can optionally provide an object that dictates how states are saved.
    msm_obj : enspara.msm.MSM object
        An enspara MSM object. This is used to fit assignments at each
        generation of sampling.
    analysis_obj : object, default=None,
        Type of analysis to perform on each cluster center. Can be used in
        state rankings.
    ranking_obj : rankings object
        This is an object with at least two functions: __init__(**args)
        and select_states(msm, n_clones). The output of this object is
        a list of states to simulate.
    spreading_func : func, default=None,
        Optionally spread state selection by minimizing similarity penalty,
        calculated using the provided metric for calculating state-distances.
        i.e. md.rmsd.
    update_freq : int, default=np.inf,
        The number of generations between a full reclustering of states and
        analysis of cluster centers. Defaults to never reclustering
        (continually adds new cluster centers without changing previously
        discovered centers).
    continue_prev : bool, default=False,
        Flag to indicate if sampling is continuing from a previous run.
        Avoids accidentally overwritting a previous run of sampling.
    sub_obj : object, default=None,
        A submission object that handles submitting clustering, MSM,
        analysis, and save_state routines. Wrappers are available for
        Slurm queueing systems as well as local machines (subprocess calls).
    q_check_obj : object, default=None,
        An object that handles checking queueing system for jobs that are
        still running.
    q_check_obj_sim : object, default=None,
        An object that handles checking queueing system for jobs that are
        still running.
    output_dir : str, default='adaptive_sampling',
        The output directory name for adaptive sampling run.
    """

    def __init__(
            self, initial_state, n_gens=1, n_kids=1, sim_obj=None,
            cluster_obj=None, save_state_obj=None, msm_obj=None,
            analysis_obj=None, ranking_obj=None, spreading_func=None,
            update_freq=np.inf, continue_prev=False, sub_obj=None,
            q_check_obj=None, q_check_obj_sim=None,
            output_dir='adaptive_sampling', verbose=True):
        # Initialize class variables
        self.sim_obj = sim_obj
        self.initial_state = initial_state
        if type(self.initial_state) is str:
            self.initial_state_md = md.load(self.initial_state)
        else:
            self.initial_state_md = self.initial_state
        self.n_gens = n_gens
        self.n_kids = n_kids
        self.cluster_obj = cluster_obj
        if save_state_obj is None:
            self.save_state_obj = SaveWrap()
        else:
            self.save_state_obj = save_state_obj
        self.save_restart_obj = SaveWrap(
            centers='restarts', save_routine='restarts')
        self.analysis_obj = analysis_obj
        # msm obj default is normalize without eq_probs
        if msm_obj is None:
            b = partial(builders.normalize, calculate_eq_probs=False)
            self.msm_obj = MSM(lag_time=1, method=b)
        else:
            self.msm_obj = msm_obj
        # ranking obj default is evens
        if ranking_obj is None:
            self.ranking_obj = rankings.evens()
        else:
            self.ranking_obj = ranking_obj
        self.spreading_func = spreading_func
        self.update_freq = update_freq
        self.continue_prev = continue_prev
        if sub_obj is None:
            self.sub_obj = lsf_subs.LSFSub(
                'bowman', n_tasks=128, R='"model=AMDEPYC_7742"')
        else:
            self.sub_obj = sub_obj
        if q_check_obj is None:
            self.q_check_obj = lsf_subs.LSFWrap()
        else:
            self.q_check_obj = q_check_obj
        if q_check_obj_sim is None:
            self.q_check_obj_sim = lsf_subs.LSFWrap()
        else:
            self.q_check_obj_sim = q_check_obj_sim
        self.output_dir = os.path.abspath(output_dir)
        self.msm_dir = self.output_dir + '/msm'
        self.verbose = verbose

    @property
    def class_name(self):
        return "AdaptiveSampling"

    @property
    def config(self):
        return {
            'initial_state': self.initial_state,
            'n_gens': self.n_gens,
            'n_kids': self.n_kids,
            'sim_obj': self.sim_obj,
            'cluster_obj': self.cluster_obj,
            'msm_obj': self.msm_obj,
            'analysis_obj': self.analysis_obj,
            'ranking_obj': self.ranking_obj,
            'update_freq': self.update_freq,
            'continue_prev': self.continue_prev,
            'sub_obj': self.sub_obj,
            'q_check_obj': self.q_check_obj,
            'q_check_obj_sim': self.q_check_obj_sim,
            'output_dir': self.output_dir,
            'verbose': self.verbose,
        }

    def print_parameters(self):
        print(
            "\n\n#########################################################" + \
            "####################")
        print(
            "                               adaptive sampling!             ")
        print(
            "###########################################################" + \
            "##################")
        if self.continue_prev:
            print("\ncontinuing sampling from a previous run!")
        print("\noutput directory:\n    " + str(self.output_dir))
        print("\nstarting state:\n    " + str(self.initial_state))
        print("\nnumber of gens:\n    " + str(self.n_gens))
        print("\nnumber of kids:\n    " + str(self.n_kids))
        print(
            "\nupdating clustering and analysis every:\n    " + \
            str(self.update_freq) + " gens")
        print("\nsimulation object:\n" + push_forward(str(self.sim_obj), 4))
        print(
            "\nclustering object:\n" + push_forward(str(self.cluster_obj), 4))
        print(
            "\nsave states object:\n" + \
            push_forward(str(self.save_state_obj), 4))
        print("\nanalysis object:\n" + push_forward(str(self.analysis_obj), 4))
#        print("\nMSM object:\n" + push_forward(str(self.msm_obj), 4))
        print("\nranking object:\n" + push_forward(str(self.ranking_obj), 4))
        print("\nsubmission object:\n" + push_forward(str(self.sub_obj), 4))
        print(
            "\nqueue checking object:\n" + \
            push_forward(str(self.q_check_obj), 4))
        print(
            "\nsim queue checking object:\n" + \
            push_forward(str(self.q_check_obj_sim), 4))
        print(
            "\n###########################################################" + \
            "##################\n")
        return

    def run(self):
        # after setup, adaptive sampling proceeds with the following steps:
        # 1) simulate, process trajectories, and move them to the
        #    msm directory
        # 2) cluster the conformations and save assignments and distances
        # 3) optionally save cluster centers
        # 4) analyze the cluster centers and save the results
        #    of the analysis
        # 5) build the MSM and save transition matrix
        # 6) rank states for reselection based on structural analysis
        #    (optional) and MSM statistics
        # 
        # If restarting an adaptive sampling run, attempts to move simulations
        # from the last gen and recluster
        self.print_parameters()
        # set msmdir
        msm_dir = self.output_dir + '/msm'
        # timeit
        t0 = time.time()
        # initilize adaptive sampling if not continuing a previous run
        # builds directories, generates first run of sampling, and
        # clusters data


        ###########################################################
        #               First generation of sampling              #
        ###########################################################

        if not self.continue_prev:
            # build initial directory structure
            logging.info('building directories')
            gen_num = 0
            gen_dir = self.output_dir + '/gen' + str(gen_num)
            _setup_directories(self.output_dir)
            # save starting state in msm directory. Will be used to load
            # and save trajectories for restarting simulations
            try:
                self.initial_state_md.save_gro(msm_dir + '/restart.gro')
            except:
                logging.warning(
                    "Could not save initial state. Initial state is not pdb or gro?")
                self.cluster_obj.base_struct_md.save_gro(msm_dir + '/restart.gro')

            ###########################################################
            #                  STEP 1 (simulations)                   #
            ###########################################################

            # initialize first run of sampling
            logging.info('starting initial simulations')
            _gen_initial_sims(
                self.output_dir, self.initial_state_md, self.sim_obj,
                self.n_kids, self.q_check_obj_sim)
            # move trajectories after sampling
            logging.info('moving trajectories')
            _move_trjs(gen_dir, self.msm_dir, gen_num, self.n_kids)
            # wait for nfs to catch up
            time.sleep(65)

            ###########################################################
            #                  STEP 2 (clustering)                    #
            ###########################################################

            # submit clustering job
            logging.info('clustering simulation data')
            t_pre = time.time()
            # since its the first round of sampling, build full msm
            self.cluster_obj.build_full = True
            self.cluster_obj.set_filenames(self.msm_dir)
            _pickle_submit(
                self.msm_dir, self.cluster_obj, self.sub_obj,
                self.q_check_obj, gen_num, 'clusterer')
            # check that clustering went well
            correct_clust =  self.cluster_obj.check_clustering(
                self.msm_dir, gen_num, self.n_kids)
            if not correct_clust:
                raise MissingData('clustering job failed!')
            # log clustering time
            t_post = time.time()
            logging.info("clustering took %0.4f seconds" % (t_post - t_pre))

            ###########################################################
            #                 STEP 3 (saving states)                  #
            ###########################################################

            if self.save_state_obj is not None:
                logging.info('saving states')
                t_pre = time.time()
                _pickle_submit(
                    self.msm_dir, self.save_state_obj, self.sub_obj,
                    self.q_check_obj, gen_num, 'save_states')
                correct_save = self.save_state_obj.check_save_states(
                    self.msm_dir)
                if not correct_save:
                    raise MissingData('Saving states failed!')
                t_post = time.time()
                logging.info(
                    'saving states took %0.4f seconds' % (t_post - t_pre))


        ###########################################################
        #               restarting adaptive sampling              #
        ###########################################################

        # if continuing from a previous run, determines gen, and
        # attempts to move trajectories, cluster data, build MSM,
        # rank states, and restart simulations
        else:
            # check for valid path to restart from
            if not os.path.exists(self.output_dir):
                raise DataInvalid(
                    "Can't continue run from output directory that doesn't" + \
                    " exist!")
            # determine where adaptive sampling left off (looks at
            # trajectories and folder numbers). Allows for a discrepancy.
            gen_num = _determine_gen(self.output_dir, ignore_error=True)
            gen_dir = self.output_dir + '/gen' + str(gen_num)
            logging.info('continuing adaptive sampling from run %d' % gen_num)
            # try to move trajectories from current gen and initiate clustering
            try:
                # move trajectories
                logging.info('moving trajectories')
                _move_trjs(gen_dir, self.msm_dir, gen_num, self.n_kids)
                # wait for nfs to catch up
                time.sleep(65)
            except:
                pass
            # error check for consistent number of trajectories and
            # gen folders
            gen_num_test = _determine_gen(self.output_dir)
            assert gen_num == gen_num_test

            ###########################################################
            #                  STEP 2 (clustering)                    #
            ###########################################################

            # determine if clustering was completed
            if os.path.exists(self.msm_dir+"/data/assignments.h5"):
                # check if clustering was successful
                correct_clust =  self.cluster_obj.check_clustering(
                    self.msm_dir, gen_num, self.n_kids)
            else:
                correct_clust = False
            # if not, recluster
            if not correct_clust:
                # submit clustering job
                logging.info('clustering simulation data')
                logging.info('updating all cluster centers')
                rebuild_num = int(gen_num / self.update_freq) - 1
                # if restarting from first gen, might not need to move
                # cluster data
                if gen_num == 0:
                    try:
                        _move_cluster_data(
                            self.msm_dir, rebuild_num, self.analysis_obj)
                    except:
                        pass
                else:
                    _move_cluster_data(
                        self.msm_dir, rebuild_num, self.analysis_obj)
                t_pre = time.time()
                # built in rebuild everything if restarting sims
                self.cluster_obj.build_full = True
                self.cluster_obj.set_filenames(self.msm_dir)
                _pickle_submit(
                    self.msm_dir, self.cluster_obj, self.sub_obj,
                    self.q_check_obj, gen_num, 'clusterer')
                correct_clust =  self.cluster_obj.check_clustering(
                    self.msm_dir, gen_num, self.n_kids)
                # if still wrong, raise error
                if not correct_clust:
                    raise MissingData('clustering job failed!')
                t_post = time.time()
                logging.info("clustering took %0.4f seconds" % (t_post - t_pre))

            ###########################################################
            #                 STEP 3 (saving states)                  #
            ###########################################################

            if self.save_state_obj is not None:
                correct_save = self.save_state_obj.check_save_states(
                    self.msm_dir)
                if not correct_save: 
                    logging.info('saving states')
                    t_pre = time.time()
                    _pickle_submit(
                        self.msm_dir, self.save_state_obj, self.sub_obj,
                        self.q_check_obj, gen_num, 'save_states')
                    correct_save = self.save_state_obj.check_save_states(
                        self.msm_dir)
                    if not correct_save:
                        raise MissingData('Saving states failed!')
                    t_post = time.time()
                    logging.info(
                        'saving states took %0.4f seconds' % (t_post - t_pre))

        # determine if updating data
        if int(gen_num % self.update_freq) == 0:
            update_data = True
        else:
            update_data = False

        ###########################################################
        #               STEP 4 (analysis of centers)              #
        ###########################################################

        # run analysis object routine
        logging.info('analyzing cluster data')
        state_rankings = _perform_analysis(
            self.analysis_obj, self.msm_dir, gen_num, self.sub_obj,
            self.q_check_obj, update_data)

        ###########################################################
        #                  STEP 5 (MSM generation)                #
        ###########################################################

        # build msm
        logging.info('building MSM')
        self.msm_obj = _prop_msm(
            self.msm_dir, self.msm_obj)

        ###########################################################
        #                   STEP 6 (rank states)                  #
        ###########################################################

        # if ranking  uses analysis from state centers, update
        # the ranking object
        if hasattr(self.ranking_obj, 'state_rankings'):
            self.ranking_obj.state_rankings = state_rankings
        # if the ranking object uses rmsd information, load cluster
        # centers and update the ranking object
        if hasattr(self.ranking_obj, 'distance_metric'):
            if self.ranking_obj.distance_metric is not None:
                logging.info('loading centers for spreading')
                self.ranking_obj.state_centers = md.load(
                    self.msm_dir+'/data/full_centers.xtc',
                    top=self.msm_dir+'/prot_masses.pdb')
        logging.info('ranking states\n')
        # rank states
        new_states = self.ranking_obj.select_states(self.msm_obj, self.n_kids)
        np.save(
            self.msm_dir + '/rankings/states_to_simulate_gen' + \
                str(gen_num) + '.npy', new_states)

        if (self.save_state_obj.save_routine == 'masses') or \
                (self.save_state_obj.centers == 'none'):
            self.save_restart_obj.gen_num = gen_num
            self.save_restart_obj.run(self.msm_dir)

        ################################################################
        #                 main adaptive sampling loop                  #
        ################################################################

        # iterate adaptive sampling until gen reaches n_gens
        for gen_num in np.arange(gen_num + 1, self.n_gens):

            ###########################################################
            #                  STEP 1 (simulations)                   #
            ###########################################################

            logging.info('STARTING GEN NUM: %d' % gen_num)
            gen_dir = self.output_dir + '/gen' + str(gen_num)
            # propagate trajectories
            str_states = ", ".join([str(state) for state in new_states])
            logging.info('starting simulations for states: ' + str_states)
            _prop_sims(
                self.output_dir, self.sim_obj, gen_num, self.q_check_obj_sim,
                new_states)
            # move trajectories
            logging.info('moving trajectories')
            _move_trjs(gen_dir, self.msm_dir, gen_num, self.n_kids)
            # ensure proper trajectories
            gen_num_test = _determine_gen(self.output_dir)
            assert gen_num == gen_num_test
            # wait for nfs to catch up
            time.sleep(65)

            ###########################################################
            #                  STEP 2 (clustering)                    #
            ###########################################################

            # determine if updating data
            if int(gen_num % self.update_freq) == 0:
                logging.info('updating all cluster centers')
                rebuild_num = int(gen_num / self.update_freq) - 1 
                _move_cluster_data(
                    self.msm_dir, rebuild_num, self.analysis_obj)
                update_data = True
            else:
                update_data = False
            # submit clustering job
            logging.info('clustering simulation data')
            t_pre = time.time()
            self.cluster_obj.build_full = update_data

            # touch trajectories...
            cmd = 'touch %s/trajectories/*.xtc' % self.msm_dir
            tools.run_commands(cmd)

            self.cluster_obj.set_filenames(self.msm_dir)
            _pickle_submit(
                self.msm_dir, self.cluster_obj, self.sub_obj,
                self.q_check_obj, gen_num, 'clusterer')
            correct_clust =  self.cluster_obj.check_clustering(
                self.msm_dir, gen_num, self.n_kids)
            if not correct_clust:
                raise MissingData('clustering job failed!')
            t_post = time.time()
            logging.info("clustering took %0.4f seconds" % (t_post - t_pre))


            ###########################################################
            #                 STEP 3 (saving states)                  #
            ###########################################################

            if self.save_state_obj is not None:
                logging.info('saving states')
                t_pre = time.time()
                _pickle_submit(
                    self.msm_dir, self.save_state_obj, self.sub_obj,
                    self.q_check_obj, gen_num, 'save_states')
                correct_save = self.save_state_obj.check_save_states(
                    self.msm_dir)
                if not correct_save:
                    raise MissingData('Saving states failed!')
                t_post = time.time()
                logging.info(
                    'saving states took %0.4f seconds' % (t_post - t_pre))

            
            ###########################################################
            #               STEP 4 (analysis of centers)              #
            ###########################################################

            # analysis
            logging.info('analyzing cluster data')
            state_rankings = _perform_analysis(
                self.analysis_obj, self.msm_dir, gen_num, self.sub_obj,
                self.q_check_obj, update_data)

            ###########################################################
            #                  STEP 5 (MSM generation)                #
            ###########################################################

            # build msm
            logging.info('building MSM')
            self.msm_obj = _prop_msm(
                self.msm_dir, self.msm_obj)

            ###########################################################
            #                   STEP 6 (rank states)                  #
            ###########################################################

            # rank states and get new 
            if hasattr(self.ranking_obj, 'state_rankings'):
                self.ranking_obj.state_rankings = state_rankings
            if hasattr(self.ranking_obj, 'distance_metric'):
                if self.ranking_obj.distance_metric is not None:
                    logging.info('loading centers for spreading')
                    self.ranking_obj.state_centers = md.load(
                        self.msm_dir+'/data/full_centers.xtc',
                        top=self.msm_dir+'/prot_masses.pdb')
            logging.info('ranking states\n')
            new_states = self.ranking_obj.select_states(self.msm_obj, self.n_kids)
            np.save(
                self.msm_dir + '/rankings/states_to_simulate_gen' + \
                    str(gen_num) + '.npy', new_states)

            # save restarts if not saved previously
            if (self.save_state_obj.save_routine == 'masses') or \
                    (self.save_state_obj.centers == 'none'):
                self.save_restart_obj.gen_num = gen_num
                self.save_restart_obj.run(self.msm_dir)

        t1 = time.time()
        logging.info("Total time took %0.4f seconds" % (t1 - t0))
