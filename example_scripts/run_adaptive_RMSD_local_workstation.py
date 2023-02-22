import glob
import mdtraj as md
import numpy as np
from enspara.cluster import KCenters
from enspara.msm import MSM
from fast import AdaptiveSampling
from fast.md_gen.gromax import Gromax, GromaxProcessing
from fast.msm_gen import ClusterWrap
from fast import RMSDWrap
from fast.sampling import rankings, scalings
from fast.submissions.os_sub import OSWrap, SPSub


def entry_point():


    ###########################################
    #            define parameters            #
    ###########################################

    # simulation parameters
    n_cpus_gromacs = 12                                # number of cpus to use for each simulation
    n_gpus_gromacs = 1                                 # number of gpus to use for each simulation
    sim_name = "protein_X"                             # name of simulation to display on queue
    top_filename = "./input_files/topol.top"           # path to gromacs topology fole
    mdp_filename = "./input_files/npt.mdp"             # path to gromacs .mdp parameter file
    gromacs_source_file = None                         # optionally provide a path to a gromacs source file
    index_file = None                                  # path to index file for simulations
                                                       # i.e. "./input_files/index.ndx"
    itp_files = None                                   # optionally provide a list of gromacs .ipt parameter
                                                       # files for use with simulations,
                                                       # glob.glob("./input_files/params/*.itp")
    pin = 'on'                                         # optionally pin cpu usage (better performance)
    gpu_id = '0'                                       # specify gpu_id of simulations

    # gromacs flags for simulations / processing
    pbc = 'mol'
    ur = 'compact'
    max_warn = 4                                       
    align_grp = 10                                     # gromacs group to align simulations to
    output_grp = 10                                    # gromacs group for output

    # clustering parameters
    cluster_radius = 0.18                              # cluster radius for MSM inbetween rounds
                                                       # of sampling, in nm
    prot_masses = "./input_files/prot_masses.pdb"      # protein topology file for loading and analyzing simulations
                                                       # should be consistent with gromacs output
    atom_indices = "./input_files/atom_indices.dat"    # atom indices for clustering simulations
    n_cpus_clustering = 24                             # number of cpu-cores to use for clustering

    # save states
    save_routine = 'masses'                            # option for saving states in-between rounds
                                                       # masses only saves the protein-masses
    save_centers = 'auto'                              # option for saving fully solvated cluster-centers in-between
                                                       # rounds. `auto` saves only states that are necessary
    mem_efficient = False                              # flag to be memeory efficient at the expense of processing
                                                       # speed. Should only be True if you cannot fit the data into
                                                       # memory.
    save_xtc_centers = True                            # flag for saving cluster centers as a gromacs xtc binary
    n_cpus_save = 24                                   # number of cpu-cores for saving states
    largest_center = cluster_radius                    # upper limit for searching assignments/distance files for
                                                       # cluster centers. Should never be larger than the cluster
                                                       # radius

    # analysis parameters
    target = "./input_files/prot_masses.pdb"           # structure for the target state
    target_iis = "./input_files/atom_indices.dat"      # atom indices to use for comparing rmsd of the target state

    # ranking parameters
    directed_scaling = scalings.feature_scale(         # FAST ranking scaling of the directed component. Should be `True`
        maximize=False)                                #     if maximizing ranking, and `False` if minimizing.
    distance_metric = md.rmsd                          # FAST state selection similarity penalty parameter
    width = 0.2                                        # width of gaussian for generation of similarity penalty score

    # adaptive sampling parameters
    starting_structure = "./input_files/start.gro"     # starting structure for production simulations.
    submission_obj = SPSub(wait=True)                  # how to submit jobs.
    q_check_obj = OSWrap(max_n_procs=1)                # how to check status of jobs (max_n_procs runs 1 simulation at a time)
                                                       #     -> SPSub/OSWrap for running on a standalone work-station
                                                       #     -> SlurmSub/SlurmWrap for running with SLURM
                                                       #     -> LSFSub/LSFWrap for running with LSF
    n_gens = 10                                        # number of generations of adaptive sampling
    n_kids = 10                                        # number of kids per round of adaptive sampling
    update_freq = 1                                    # frequency to refresh clustering between rounds.
                                                       # 1 refreshes MSM every round (best performance/recommended).
                                                       # Values greater than 1 save computation between rounds at the
                                                       # expense of performance.
    continue_prev = False                              # Flag for continuing a previous run (fail-safe to prevent overwriting data)
    output_dir = "FAST-RMSD_proteinX"                  # name of output dir containing the entirety of adaptive sampling

    ############################################
    #            initialize objects            #
    ############################################

    # simulation object
    gro_submission = SPSub(wait=True)
    gro_processing = GromaxProcessing(
        align_group=align_group, output_group=output_group,
        index_file=index_file, pbc=pbc, ur=ur)
    sim_obj = Gromax(
        top_file=top_filename, mdp_file=mdp_filename, n_cpus=n_cpus_gromacs,
        n_gpus=n_gpus_gromacs, processing_obj=gro_processing,
        submission_obj=gro_submission, pin=pin, gpu_id=gpu_id,
        env_exports=md_exports, source_file=gromacs_source_file, index_file=index_file,
        itp_files=itp_files, max_warn=max_warn)

    # clustering object
    base_clust_obj = KCenters(metric=md.rmsd, cluster_radius=cluster_radius)
    clust_obj = ClusterWrap(
        base_struct=prot_masses, base_clust_obj=base_clust_obj,
        atom_indices=atom_indices, n_procs=n_cpus_clustering,
        mem_efficient=mem_efficient)

    # save state object
    save_state_obj = SaveWrap(
        save_routine=save_routine, centers=save_centers, n_procs=n_cpus_save,
        largest_center=largest_center, save_xtc_centers=save_xtc_centers)

    # analysis object
    anal_obj = RMSDWrap(
        base_struct=target_state, atom_indices=atom_indices,
        atom_indices_target=target_state_iis)

    # ranking object
    ranking_obj = rankings.FAST(
        directed_scaling=directed_scaling, distance_metric=distance_metric,
        width=width)

    # sampling object
    a = AdaptiveSampling(
        starting_structure, n_gens=n_gens, n_kids=n_kids, sim_obj=sim_obj,
        cluster_obj=clust_obj, save_state_obj=save_state_obj,
        analysis_obj=anal_obj, ranking_obj=ranking_obj,
        continue_prev=continue_prev, update_freq=update_freq,
        sub_obj=submission_obj, output_dir=output_dir, q_check_obj=q_check_obj,
        q_check_obj_sim=q_check_obj)

    ##############################################
    #                run sampling                #
    ##############################################

    # run
    a.run()
    


if __name__=='__main__':
    entry_point()
