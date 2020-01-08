# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential

#######################################################################
# imports
#######################################################################


import numpy as np
from ..base import base


#######################################################################
# code
#######################################################################


class base_analysis(base):
    """Base class for all analysis to inherit.

    Contains the folder name to store and load analysis,
    a loader for rankings, and a function to set the output
    to a specific gen number and MSM location.
    """
    def __init__(self):
        pass

    @property
    def ranking_folder(self):
        folder_name = "rankings"
        return folder_name

    @property
    def state_rankings(self):
        return np.load(self.output_name)

    def set_output(self, msm_dir, gen_num):
        self.msm_dir = msm_dir
        # set analysis output
        if self.analysis_folder is not None:
            self.output_folder = msm_dir + "/" + self.analysis_folder
        # set ranking output
        self.output_name = msm_dir + "/" + self.ranking_folder + "/" + \
            self.base_output_name + str(gen_num) + ".npy"

