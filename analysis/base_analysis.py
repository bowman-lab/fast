import numpy as np
from ..base import base

class base_analysis(base):
    def __init__(self):
        pass

    @property
    def ranking_folder(self):
        return "rankings"

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

