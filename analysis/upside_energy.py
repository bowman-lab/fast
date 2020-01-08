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
import mdtraj as md
import mdtraj_upside as mu # has function for parsing Upside trajectories into MDTraj
import numpy as np
import os
import upside_engine as ue # computing energies 
from .base_analysis import base_analysis
from .. import tools


#######################################################################
# code
#######################################################################


class UpsideEnergyWrap(base_analysis):
    """Analyses upside energies.

    Parameters
    ----------
    sim_filename : str,
        The simulation engine to use for calculating upside energies.

    Attributes
    ----------
    output_name : str,
        The file containing rankings.
    """
    def __init__(
            self, sim_filename):
        # determine base_struct
        self.sim_filename = os.path.abspath(sim_filename)

    @property
    def class_name(self):
        return "UpsideEnergysWrap"

    @property
    def config(self):
        return {
            'sim_filename': self.sim_filename,
        }

    @property
    def analysis_folder(self):
        return None

    @property
    def base_output_name(self):
        return "upside_energy_per_state"

    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load(
                "./data/full_centers.xtc", top="./prot_masses.pdb")
            pos = mu.extract_bb_pos_angstroms(centers)
            engine = ue.Upside(self.sim_filename)
            # calculate and save energies
            energies = np.zeros(centers.n_frames)
            for i, pos_i in enumerate(pos):
                energies[i] = engine.energy(pos_i)
            np.save(self.output_name, energies)        
