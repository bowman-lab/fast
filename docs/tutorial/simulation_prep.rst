Simulation Prep
===============

Running adaptive sampling requires some prepared inputs beforehand.

A minimal list of necessary files are as follows:

1: Initial structure(.gro)/topology(.top) files for production simulations
2: Simulation parameter (.mdp) file for production simulations
3: Reference structure (.pdb) file for clustering/analysis (usually protein masses without solvent/ions)
4: Text file containing atomic indices (.txt or .dat) that will be used for clustering. Based off of #3


1. Prepare your starting state
--------------------------------

As of now, this package only supports running simulations with GROMACS. Tutorials for using GROMACS and preparing a starting state for a production run can be found at (http://www.mdtutorials.com/gmx/).



2. Simulation parameter file
-----------------------------

This be a GROMACS .mdp file that will be used for production runs.

For a reference on parameter options, please refer to (https://manual.gromacs.org/documentation/current/user-guide/mdp-options.html).

3. Reference Structure
-----------------------

The reference structure is a PDB file containing the relevant atoms for clustering/analysis. Post-processed simulations will be loaded using this structure and should constitute a minimal construct for analysis. A good choice for this structure is the protein masses (stripped of solvent/ions) from the starting .gro file.

GROMACS can prepare this structure from #1:

.. code-block:: bash

    gmx trjconv -f input.gro -s md.tpr -o prot_masses.pdb -center

Where `prot_masses.pdb` is the reference structure and the user should select `prot-masses` for both centering and output.

4. Atom indices for clustering
-------------------------------

The atom indices here will be used for clustering simulations between rounds. They should be focused to the dynamics that we care to characterize since they will define the conformational landscape.

If we are interested in the backbone dynamics (topological conformational changes) a good approach is to use the backbone indices + CB atoms. An easy way to do this is using MDTraj:

.. code-block:: python

    import numpy as np
    import mdtraj as md
    pdb = md.load("prot_masses.pdb")
    atom_indices = pdb.top.select_atom_indices('minimal')
    np.savetxt("atom_indices.dat", atom_indices, fmt='%d')

Here, the `atom_indices.dat` is a single column of atomic indices that will be used for clustering (zero-indexed). These indices should be based on the reference structure in #3.

