# FAST

FAST (fluctuation amplification of specific traits), is a goal-oriented adaptive sampling algorithm designed to rapidly search through conformational space. To achieve this, FAST iterates between launching simulations and analyzing the already existing simulations for structural states nearest the desired structural properties. Starting places for new simulations are started both from states near the desired structural property (exploitation) as well as from states further from the desired property (exporation) in an attempt to minimize "tunneling" behavior and enable backtracking. 

# Reference
If you use `FAST` for published research, please cite us:

Zimmerman, M. I., & Bowman, G. R. (2015). FAST conformational searches by balancing exploration/exploitation trade-offs. Journal of chemical theory and computation, 11(12), 5747-5757.

For more information about FAST, please read:
Zimmerman, M. I., & Bowman, G. R. (2016). How to run FAST simulations. Methods in Enzymology, 578, 213-225.

# Installation
1. Download FAST to your local machine/cluster:

```
git clone https://github.com/bowman-lab/fast.git
```

2. Add the FAST path to your python path. 

To add to a conda/mamba environment:
`ln -s /path/to/fast/ /path/to/conda/envs/env_name/lib/pythonVersion/site-packages`

To add to all environments:
Edit your bashrc file to include:
`export PYTHONPATH="/path/to/fast/../:$PYTHONPATH"`

# Running FAST

We have provided an example ipython notebook demonstrating several use cases of FAST. Please take a look at `fast/examples/FAST-tutorial.ipynb` to get started.

# FAQ
1. I want a continuous trajectory from my starting structure to this interesting structure I found, how do I make that?

Take a peek at the script: `fast/examples/stitch-fast-trjs.py`. Usage is `python stitch-fast-trjs.py '/path/to/trjs/*.xtc' /path/to/topologyfile.pdb /path/to/assignments_file.h5 state_of_interest`. By default, this will generate an xtc with no downsampling from the state FAST started from to your specified end state. Optionally, you can specify how you'd like the trajectory aligned, subsampling rates, and output names.

2. My FAST run errored out. What do I do now?

Usually the default output file has some clue as to what happened. Occasionally, you'll need to delve deeper to figure out what is causing the error. FAST will write more detailed log files for the currently running step in: `/FAST-output-dir/msm/` and subsequently move them to: `FAST-output-dir/msm/submissions`. Errors in simulations will be written to the log file in `FAST-output-dir/genX/kidX/`.

If your simulation errored out due to hardware errors, you should be able to manually restart the simulation by resubmitting the submission script. If your simulation crashed for other reasons, you may wish to re-evaluate your simulation parameters to ensure you have a stable simulation.

If your clustering failed, you may wish to check that you have sufficient system RAM to load your dataset. Optionally, look at subsampling your trajectories.

3. Why is my FAST directory so storage heavy?
FAST writes out fully solvated trajectories to enable restarting simulations from any frame. These are saved in `FAST-output-dir/msm/trajectories_full`. Once you are fully finished analyzing/running your FAST simulation you may wish to offload these solvated trajectories.

Similarly, if you saved all centers, the `FAST-output-dir/msm/centers_restarts` folder can be fairly storage heavy as it contains fully solvated `.gro` files for each simulation restart. These are saved for each FAST generation and the archived restarts are placed in `FAST-output-dir/msm/old`.


