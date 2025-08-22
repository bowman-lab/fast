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


def map_sasa_core(trajectoryfile, top, probe_radius, **kwargs):
    import mdtraj as md

    print("kwargs: ", kwargs)
    trj = md.load(trajectoryfile, top=top, **kwargs)

    print(
        "computing using threading ",
        probe_radius,
        "nm sasa for",
        trajectoryfile,
        "using topology",
        top,
    )
    sasas = md.shrake_rupley(
        trj,
        probe_radius=probe_radius,
    )

    return sasas


def condense_sasas(sasas, top, residue_indices, sidechain_only=True):
    import time
    import numpy as np

    assert top.n_atoms == sasas.shape[1], "%s != %s" % (top.n_atoms, sasas.shape[1])
    if sidechain_only:
        SELECTION = (
            "not (name N or name C or name CA or name O or "
            "name HA or name H or name H1 or name H2 or name "
            "H3 or name OXT)"
        )
        sc_ids = [
            top.top.select("resid %s and ( %s )" % (r, SELECTION))
            for r in residue_indices
        ]
    else:
        sc_ids = [top.top.select("resid %s" % r) for r in residue_indices]

    rsd_sasas = np.zeros((sasas.shape[0], len(sc_ids)), dtype="float32")

    for i in range(len(sc_ids)):
        try:
            rsd_sasas[:, i] = sasas[:, sc_ids[i]].sum(axis=1)
        except:
            print("condensing residue", i, "of", top.n_residues)
            print(sc_ids[i])
            print(sasas.shape)
            print(rsd_sasas.shape)
            raise

    return rsd_sasas


def assemble_sasa_h5(sasas, filename):

    import os
    import tables

    if not os.path.isdir(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))

    if os.path.isfile(filename):
        raise FileExistsError(f"File '{filename}' already exists.")

    compression = tables.Filters(complevel=9, complib="zlib", shuffle=True)
    n_zeros = len(str(len(sasas))) + 1

    print(filename)
    with tables.open_file(filename, "a") as handle:
        shape = None

        for i, sasa in enumerate(sasas):
            # does it need to be transposed?
            data = sasa

            atom = tables.Atom.from_dtype(data.dtype)
            tag = "sasas_" + str(i).zfill(n_zeros)

            if tag in handle.root:
                logger.warn("Tag %s already existed in %s. Overwriting.", tag, filename)
                handle.remove_node("/", name=tag)

            if shape is None:
                shape = data.shape
            elif len(shape) > 1:
                assert shape[1] == data.shape[1], (
                    "We had %s residues, but then loaded trajectory %s and it had %s."
                    % (shape[1], i, data.shape[1])
                )

            node = handle.create_carray(
                where="/", name=tag, atom=atom, shape=data.shape, filters=compression
            )
            node[:] = data

    return filename


def cluster_features(
    features,
    trj_lengths,
    cluster_distance="euclidean",
    algorithm="kcenters",
    n_clusters=None,
    cluster_radius=None,
    kmedoids_updates=5,
    **kwargs,
):
    import os
    from enspara import cluster
    import subprocess

    # get distance metric
    euclidean_distance = cluster.util._get_distance_method(cluster_distance)
    if algorithm == "kcenters":
        clusterer = cluster.KCenters(
            metric=euclidean_distance,
            n_clusters=n_clusters,
            cluster_radius=cluster_radius,
            **kwargs,
        )
    elif algorithm == "khybrid":
        clusterer = cluster.KHybrid(
            metric=euclidean_distance,
            n_clusters=n_clusters,
            cluster_radius=cluster_radius,
            **kwargs,
        )

    clusterer.fit(features)
    center_indices, distances, assignments, center_features = (
        clusterer.result_.partition(trj_lengths)
    )

    return center_indices, distances, assignments, center_features


def write_struct_ctrs(
    trajectoryfiles, topoology, ctr_inds, ctr_structs_file, stride=1, n_procs=1
):

    import os
    import pickle
    import mdtraj as md
    from enspara.util.load import load_as_concatenated

    top = topoology.top
    try:
        lengths, xyz = load_as_concatenated(
            filenames=[trajectoryfiles[tr] for tr, fr in ctr_inds],
            args=[{"frame": fr * stride, "top": top} for tr, fr in ctr_inds],
            processes=4,
        )
    except IndexError:
        print(len(trajectoryfiles), len(ctr_inds), max([tr for tr, fr in ctr_inds]))
        raise

    ctr_structs = md.Trajectory(xyz=xyz, topology=top)
    # align to first center
    ctr_structs.superpose(ctr_structs[0])
    ctr_structs.save(ctr_structs_file)

    return ctr_structs_file


def load_trjs(trj_filenames, n_procs=1, **kwargs):
    """Parallelize loading trajectories from msm directory."""
    # get filenames
    trj_filenames_test = np.array(
        [os.path.abspath(f) for f in np.sort(np.array(glob.glob("trajectories/*.xtc")))]
    )
    t0 = time.time()
    diffs = np.setdiff1d(trj_filenames, trj_filenames_test)
    while diffs.shape[0] != 0:
        t1 = time.time()
        logging.info(
            "waiting on nfs. missing %d files (%0.2f s)"
            % (trj_filenames.shape[0] - trj_filenames_test.shape[0], t1 - t0)
        )
        time.sleep(15)
        _ = tools.run_commands("ls trajectories/*.xtc")
        trj_filenames_test = np.array(
            [
                os.path.abspath(f)
                for f in np.sort(np.array(glob.glob("trajectories/*.xtc")))
            ]
        )
        diffs = np.setdiff1d(trj_filenames, trj_filenames_test)
    # parallelize load with **kwargs
    partial_load = partial(md.load, **kwargs)
    pool = Pool(processes=n_procs)
    trjs = pool.map(partial_load, trj_filenames)
    pool.terminate()
    return trjs


class SASAClusterWrap(base):
    """Clustering wrapper function

    Parameters
    ----------
    base_struct : str or md.Trajectory,
        A structure with the same topology as the trajectories to load.
    base_clust_obj :
        A callable object with a fit method that will cluster SASA data
    probe_radius : float
        Probe radius used for SASA calculation
    residue_indices : string or np.array,
        The residue indices of the base_struct to cluster with.
    sidechain_only : boolean
        Indicator that determines whether SASA is computed for an entire residue
        or only the sidechain atoms
    build_full : bool, default = True,
        Flag for building from scratch. NEEDS TO BE IMPLEMENTED
    n_procs : int, default = 1,
        The number of processes to use when loading, clustering and
        saving conformations.
    """

    def __init__(
        self,
        base_struct,
        base_clust_obj=None,
        probe_radius=0.14,
        residue_indices=None,
        sidechain_only=True,
        n_procs=1,
    ):
        # determine base_struct
        self.base_struct = base_struct
        if type(base_struct) is md.Trajectory:
            self.base_struct_md = base_struct
        else:
            self.base_struct_md = md.load(base_struct)
        # determine base clustering object
        if base_clust_obj is None:
            euclidean_distance = cluster.util._get_distance_method(cluster_distance)
            self.base_clust_obj = cluster.KCenters(
                metric=euclidean_distance, cluster_radius=1.0
            )
        else:
            self.base_clust_obj = base_clust_obj
        self.probe_radius = probe_radius
        self.residue_indices = residue_indices
        # determine residue indices
        if type(residue_indices) is str:
            try:
                self.residue_indices_vals = np.loadtxt(residue_indices, dtype=int)
            except ValueError:
                print("\n")
                logging.warning(
                    "Residue indices for SASA clustering are not integers. "
                    "Attempting to convert"
                )
                non_int_vals = np.loadtxt(residue_indices)
                self.residue_indices_vals = np.array(non_int_vals, dtype=int)
                # ensure no conversion error
                diffs = self.residue_indices_vals - non_int_vals
                assert np.all(diffs == np.zeros(non_int_vals.shape[0]))
        self.n_procs = n_procs
        self.sidechain_only = sidechain_only
        # self.build_full = build_full
        self.trj_filenames = None

    def check_clustering(self, msm_dir, gen_num, n_kids, verbose=True):
        correct_clustering = True
        total_assignments = (gen_num + 1) * n_kids
        assignments = ra.load(msm_dir + "/data/assignments.h5")
        n_assignments = len(assignments)
        if total_assignments != n_assignments:
            correct_clustering = False
            logging.info(
                "inconsistent number of trajectories between assignments and data!"
            )
        return correct_clustering

    @property
    def class_name(self):
        return "SASAClusterWrap"

    @property
    def config(self):
        return {
            "base_struct": self.base_struct,
            "base_clust_obj": self.base_clust_obj,
            "residue_indices": self.residue_indices,
            # 'build_full': self.build_full,
            "n_procs": self.n_procs,
            "trj_filenames": self.trj_filenames,
        }

    def set_filenames(self, msm_dir):
        self.trj_filenames = np.sort(
            np.array(glob.glob(msm_dir + "/trajectories/*.xtc"))
        )
        return

    def run(self):
        # Determine trajectories
        self.set_filenames(".")
        self.base_struct_md.save_pdb("./prot_masses.pdb")
        print(self.trj_filenames)

        # determine if there is sasas.h5 present already
        # we try to avoid deleting sasas.h5 when restarting clustering
        # because this can be an expensive step
        current_sasa_file = glob.glob("data/sasas.h5")

        if current_sasa_file == []:
            # to reduce duplicative computation check for presence
            # of old sasa files

            old_sasa_files = glob.glob("old/data*/sasas.h5")
            if old_sasa_files == []:
                print("no old sasa files available -- in gen0")
                completed_sasa_calc_index = 0
            else:
                print("will load old sasa file")
                # sort by gen num
                # pattern of files is old/data0/assignments.h5
                old_sasa_files = sorted(
                    old_sasa_files, key=lambda x: int(x.split("/")[1][4:])
                )
                old_condensed_sasas = ra.load(old_sasa_files[-1])

                # old_sasas has shape n_traj, n_timepoints, n_resids
                # n_traj is the number of trajectories whose sasas have been computed
                completed_sasa_calc_index = old_condensed_sasas.shape[0]

            print("running SASA core")
            t0 = time.time()

            # partial_map_sasa = partial(map_sasa_core, top=self.base_struct_md, probe_radius=self.probe_radius)
            # pool = Pool(processes=self.n_procs)
            # sasas = pool.map(partial_map_sasa, self.trj_filenames)
            # pool.terminate()

            new_sasas = [
                map_sasa_core(
                    trj,
                    self.base_struct_md,
                    self.probe_radius,
                )
                for trj in self.trj_filenames[completed_sasa_calc_index:]
            ]

            t1 = time.time()
            total_time = t1 - t0
            print(
                f"ran SASA core in {total_time}"
            )  # can time to see how much parallelization improves code
            print("shape of new_sasas[0] is :", new_sasas[0].shape)
            new_trj_lengths = [sasa.shape[0] for sasa in new_sasas]
            print(new_trj_lengths)
            print("condensing SASA")
            new_condensed_sasas = [
                condense_sasas(
                    sasa,
                    self.base_struct_md,
                    self.residue_indices_vals,
                    self.sidechain_only,
                )
                for sasa in new_sasas
            ]
            print("finished condensing SASA")

            # if gen0 sasas is simply new_sasas
            if old_sasa_files == []:
                condensed_sasas = np.array(new_condensed_sasas)
                trj_lengths = new_trj_lengths
            else:
                condensed_sasas = np.concatenate(
                    (old_condensed_sasas, new_condensed_sasas)
                )
                old_lengths = [sasa.shape[0] for sasa in old_condensed_sasas]
                trj_lengths = np.concatenate((old_lengths, new_trj_lengths))

            # write out h5 file with all sasas
            SC_SASA_FILE = "./data/sasas.h5"
            ra.save(SC_SASA_FILE, condensed_sasas)
            print(trj_lengths)
            # sasa_sidechain_h5 = assemble_sasa_h5(
            #     sasas=condensed_sasas,
            #     filename=SC_SASA_FILE)
        else:
            condensed_sasas = ra.load(current_sasa_file[0])
            trj_lengths = [t.shape[0] for t in condensed_sasas]
            print(trj_lengths)

        # center_indices, distances, assignments, centers = (
        #     cluster_features(np.concatenate(condensed_sasas),
        #                      trj_lengths,
        #                      algorithm='kcenters',
        #                      n_clusters=None,
        #                      cluster_radius=None,))

        print(f"condensed_sasas has shape {condensed_sasas.shape}")
        print(f"clustering data matrix of size {np.concatenate(condensed_sasas).shape}")
        self.base_clust_obj.fit(np.concatenate(condensed_sasas))
        center_indices, distances, assignments, centers = (
            self.base_clust_obj.result_.partition(trj_lengths)
        )

        # Save output
        ra.save("./data/assignments.h5", assignments)
        ra.save("./data/distances.h5", distances)
        np.save("./data/center-indices.npy", center_indices)
        np.save("./data/sasa-centers.npy", centers)
        # save number of states in txt file for ease of inspection
        n_states = len(center_indices)
        with open("./data/unique_states.txt", "w") as f:
            f.write("%d" % n_states)

        # save unique states npy for save states to use
        np.save("./data/unique_states.npy", np.arange(n_states))

        write_struct_ctrs(
            self.trj_filenames,
            self.base_struct_md,
            center_indices,
            "./data/full_centers.xtc",
            n_procs=self.n_procs,
        )
