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


class feature_scale(base):
    """Feature scales data: (x - xmin) / (xmax - xmin)"""
    def __init__(self, maximize=True):
        self.maximize = maximize

    @property
    def class_name(self):
        return "feature_scale"

    @property
    def config(self):
        return {
            'maximize': self.maximize
        }

    def scale(self, values):
        value_spread = values.max() - values.min()
        if value_spread == 0.0:
            scaled_values = np.zeros(values.shape)
        else:
            if self.maximize:
                scaled_values = (values - values.min()) / value_spread
            else:
                scaled_values = (values.max() - values) / value_spread
        return scaled_values


class sigmoid_scale:
    """Scales values with a sigmoid"""
    def __init__(self, maximize=True, a=3):
        self.maximize = maximize
        self.a = a

    def scale(self, values):
        sigma = np.median(values)
        sig_scale = (1 + ((values/sigma)**self.a))**-1
        if self.maximize:
            sig_scale = 1 - sig_scale
        return sig_scale

