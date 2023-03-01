Running Adaptive Sampling
=========================

With all of the files needed, running adaptive sampling is as simple as executing a python script!

Example scripts can be found in `fast/example_scripts/`

Currently, there are two example scripts:
1. FAST-RMSD simulations run on a local work-station by minimizing RMSD between two structures.
2. FAST-Distance simulations run on an HPC with LSF by maximizing distances between pairs of atoms.

Adaptive sampling has a lot of moving parts, so an attempt has been made to make this code modular and independent (for customizations and methods development).

The main sampling function is `fast.sampling.core.AdaptiveSampling`, which takes in all of the parameters/modules for orchestrating events. Most modules follow the m = Module(); m.run() paradigm where modules are initialized and passed into the main function to be run at production time.

Adaptive sampling proceeds as follows:

1. Simulate
2. Cluster/MSM
3. Analyze cluster-centers
4. Rank states
5. Repeat 1-4 for some number of rounds

There are modules that take in local parameters that are then fed into `AdaptiveSampling`. The python script runs in the background on the head-node or work-station and submits various jobs to monitor their progress. The next sections will have a breif description of the various events and modules that control them.

1. Simulate
-------------

Currently, there is only support for simulating using GROMACS (OpenMM, NAMD, and Upside coming soon!).

The module `fast.md_gen.gromax` contains basic python wrappers for submitting/processing GROMACS jobs.
`fast.md_gen.gromax.Gromax` is used to submit a simulation/energy minimization run. Inputs are simulation parameters, such as topology file, mdp file, number of cpus/gpus to use, etc. Any input here will be executed on the command-line as a GROMACS parameter, i.e. pin='on' -> `gmx ... pin='on'`.

Module will also take in a `prosessing_obj` for removing solvent/ions and removing periodic boundaries. This object comes from `fast.md_gen.gromax.GromaxProcessing`, that defines how to process production trajectories for clustering/analysis.

Simulations are submitted as jobs based on the input `submission_obj`, which is described further below.

2. Cluster/MSM
----------------

Clustering is performed on the back-end using the `enspara` package. `fast.msm_gen.clustering.ClusterWrap` is a wrapper for the `enspara` clustering. It takes in an `enspara` clustering module and other relevant clustering parameters.

After clustering, states are optionally saved using `fast.msm_gen.save_states.SaveWrap`, either as `protein_masses` and/or `full_centers`.

3. Analysis
-------------

Cluster centers are analyzed to generate order parameters to use for ranking states.

Current analysis routines supported are:

1. native contacts
2. distances between pairs of atoms / center of mass
3. potential energy
4. pocket volumes (based on an in house python ligsite implementation)
5. RMSD

These modules operate on cluster centers and output numpy lists of a value per structure center.

4. Ranking
------------

The core of adaptive sampling is in how states are ranked for selection.

Here, `fast.sampling.rankings` has a number of modular ranking classes.

Supported statistical ranking modules are:

1. Evens
2. Mincounts
3. Page-ranking

Each of these rankings can be passed into state/structure-based rankings

1. FAST
2. String

Additionally, states can be "spread out" using a similarity penalty. In practice this works by selecting the top ranked state and adding a gaussian penalty to states similar to this state. Then picking the next state and repeating. Similarity can be based on any metric but RMSD is a natural metric.


