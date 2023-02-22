# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential


#######################################################################
# imports
#######################################################################


import enspara.tpt
#import msmbuilder.tpt
import numpy as np
import time
import scipy.sparse as spar
from . import scalings
from ..base import base
from ..exception import ConvergenceWarning, ImproperlyConfigured


########################################################################
# helper functions
########################################################################


def _evens_select_states(unique_states, n_clones):
    """Helper function for evens state selection. Picks among all
    discovered states evenly. If more states were discovered than
    clones, randomly picks remainder states."""
    # calculate the number of clones per state and the balance to
    # match n_clones
    clones_per_state = int(n_clones / len(unique_states))
    remainder_states = n_clones % len(unique_states)
    # generate states to simulate list
    repeat_states_to_simulate = np.repeat(unique_states, clones_per_state)
    remainder_states_to_simulate = np.random.choice(
        unique_states, remainder_states, replace=False)
    total_states_to_simulate = np.concatenate(
        [repeat_states_to_simulate, remainder_states_to_simulate])
    return total_states_to_simulate


def _unbias_state_selection(states, rankings, n_selections, select_max=True):
    """Unbiases state selection due to state labeling. Shuffles states
    with equivalent rankings so that state index does not influence
    selection probability."""
    # determine the unique ranking values
    unique_rankings = np.unique(rankings)
    # sort states and rankings
    iis_sort = np.argsort(rankings)
    sorted_rankings = rankings[iis_sort]
    sorted_states = states[iis_sort]
    # if we're maximizing the rankings, reverse the sorts
    if select_max:
        unique_rankings = unique_rankings[::-1]
        sorted_rankings = sorted_rankings[::-1]
        sorted_states = sorted_states[::-1]
    # shuffle states with unique rankings to unbias selections
    # this is done upto n_selections
    tot_state_count = 0
    for ranking_num in unique_rankings:
        iis = np.where(sorted_rankings == ranking_num)[0]
        sorted_states[iis] = sorted_states[np.random.permutation(iis)]
        tot_state_count += len(iis)
        if tot_state_count > n_selections:
            break
    return sorted_states[:n_selections]


def _select_states_spreading(
        rankings, unique_states, n_clones, centers, distance_metric,
        select_max=True, width=1.0, non_overlap=True):
    """Selects states that maximize ranking and are structurally unique.
    This proceeds as follows: 1) select a state that maximizes the
    ranking 2) penalize states that are structurally similar to
    previously selected, 3) select a new state that maximizes the
    updated ranking, and 4) repeat 2-3 until n_clones are chosen.

    Parameters
    ----------
    rankings : array, shape=(n_states,)
        The list of state rankings.
    unique_states : array, shape=(n_states,)
        The list of state indices
    n_clones : int,
        Number of states to select.
    centers : array-like,
        A list of state centers.
    distance_metric : function,
        Function for calculating distances between state centers.
    select_max : bool, default = True
        Optionally select from maximum.
    width : float, default = 1.0
        Gaussian width for calculating penalties.
    non_overlap : bool, default = True
        Optionally ensure that states are not sampled with replacement.
    
    Returns
    ----------
    states_to_simulate : array, shape=(n_clones,)
        The list of state indices to restart simulations from.
    """
    # pick the first state
    states_to_simulate = [
        _unbias_state_selection(
            unique_states, rankings, 1, select_max=select_max)[0]]
    # initialize distance list
    dist_list = []
    # iterate state selection
    for num in range(n_clones-1):
        # get distances to previously selected states
        dist_list.append(
            distance_metric(
                centers[unique_states], centers[states_to_simulate[-1]]))
        # convert distances to gaussian penalties
        gaussian_weights = np.sum(
            [
                (1 - np.exp(-(dists**2)/float(2.0*(width**2))))
                for dists in dist_list], axis=0) / len(dist_list)
        # generate new rankings
        new_rankings = rankings + gaussian_weights
        # if only selecting new states, zero the rankings of those
        # previously selected
        if non_overlap:
            states_to_zero = np.array(
                [
                    np.where(unique_states == state)[0]
                    for state in states_to_simulate])
            if select_max:
                new_rankings[states_to_zero] = 0
            else:
                new_rankings[states_to_zero] = np.inf
        # pick next state
        states_to_simulate.append(
            _unbias_state_selection(
                unique_states, new_rankings, 1, select_max=select_max)[0])
    # numpy it and return
    states_to_simulate = np.array(states_to_simulate)
    return states_to_simulate


def get_unique_states(msm):
    """returns a list of the visited states within an msm object"""
    tcounts = msm.tcounts_
    unique_states = np.unique(np.nonzero(tcounts)[0])
    return unique_states


########################################################################
#                            page rankings                             #
########################################################################


def generate_aij(tcounts, spreading=False):
    """Generates the adjacency matrix used for page ranking.

    Parameters
    ----------
    tcounts : matrix, shape=(n_states, n_states)
        The count matrix of an MSM. Can be dense or sparse.
    spreading : bool, default=False
        Optionally transposes matrix to do counts spreading instead of
        page rank.

    Returns
    ----------
    aij : matrix, shape=(n_states, n_states)
        The adjacency matrix used for page ranking.

    """
    # check if sparse matrix
    if not spar.isspmatrix(tcounts):
        iis = np.where(tcounts != 0)
        tcounts = spar.coo_matrix(
            (tcounts[iis], iis), shape=(len(tcounts), len(tcounts)))
    else:
        tcounts = tcounts.tocoo()
    # get row and col information
    row_info = tcounts.row
    col_info = tcounts.col
    tcounts.data = np.zeros(len(tcounts.data)) + 1
    tcounts.setdiag(0) # Sets diagonal to zero
    tcounts = tcounts.tocsr()
    tcounts.eliminate_zeros()
    # optionally does spreading
    if spreading:
        tcounts = (tcounts + tcounts.T) / 2.
        aij = normalize(tcounts, norm='l1', axis=1)
    else:
        # Set 0 connections to 1 (
        # this doesn't change anything and avoids dividing by zero)
        connections = tcounts.sum(axis = 1)
        iis = np.where(connections == 0)
        connections[iis] = 1
        # Convert 1/Connections to sparse matrix format
        connections = spar.coo_matrix(connections).data
        con_len = len(connections)
        iis = (np.array(range(con_len)), np.array(range(con_len)))
        inv_connections = spar.coo_matrix(
            (connections**-1, iis), shape=(con_len, con_len))
        aij = tcounts.transpose() * inv_connections
    return aij


def rank_aij(aij, d=0.85, Pi=None, max_iters=100000, norm=True):
    """Ranks the adjacency matrix.
    
    Parameters
    ----------
    aij : matrix
        The adjacency matrix used for ranking.
    d : float
        The weight of page ranks [0, 1]. A value of 1 is pure page rank
        and 0 is all the initial ranks.
    Pi : array, default=None
        The prior ranks.
    max_iters : int, default=100000
        The maximum number of iterations to check for convergence.
    norm : bool, default=True
        Normilizes output ranks

    Returns
    ----------
    The rankings of each state

    """
    N = float(aij.shape[0])
    # if Pi is None, set it to 1/total states
    if Pi is None:
        Pi = np.zeros(int(N))
        Pi[:] = 1/N
    # set error for page ranks
    error = 1 / N**2
    # first pass of rankings
    new_page_rank = (1 - d) * Pi + d * aij.dot(Pi)
    pr_error = np.sum(np.abs(Pi - new_page_rank))
    page_rank = new_page_rank
    # iterate until error is below threshold
    iters = 0
    while pr_error > error:
        new_page_rank = (1 - d) * Pi + d * aij.dot(page_rank)
        pr_error = np.sum(np.abs(page_rank - new_page_rank))
        page_rank = new_page_rank
        iters += 1
        # error out if does not converge
        if iters > max_iters:
            raise ConvergenceWarning(
                'page ranking failed to converge in %s steps' % max_iters)
    # normalize rankings
    if norm:
        page_rank *= 100./page_rank.sum()
    return page_rank


########################################################################
#                           ranking classes                            #
########################################################################


class evens(base):
    """Evens ranking object"""

    def __init__(self):
        pass

    @property
    def class_name(self):
        return "evens"

    @property
    def config(self):
        return {
        }

    def rank(self, msm, unique_states=None):
        return None

    def select_states(self, msm, n_clones):
        unique_states = get_unique_states(msm)
        return _evens_select_states(unique_states, n_clones)


class base_ranking(base):
    """base ranking class. Pieces out selection of states from
    independent rankings"""

    def __init__(
            self, maximize_ranking=True, state_centers=None,
            distance_metric=None, width=1.0):
        self.maximize_ranking = maximize_ranking
        self.state_centers = state_centers
        self.distance_metric = distance_metric
        self.width = width

    def select_states(self, msm, n_clones):
        # determine discovered states from msm
        unique_states = get_unique_states(msm)
        # if not enough discovered states for selection of n_clones,
        # selects states using the evens method
        if len(unique_states) < n_clones:
            states_to_simulate = _evens_select_states(unique_states, n_clones)
        # selects the n_clones with minimum counts
        else:
            rankings = self.rank(msm, unique_states=unique_states)
            # if a state-ranking is `nan`, does not consider it in rankings.
            non_nan_rank_iis = np.where(1 - np.isnan(rankings))[0]
            # if not enought non-`nan` states are discivered, performs evens
            if len(non_nan_rank_iis) < n_clones:
                states_to_simulate = _evens_select_states(
                    unique_states[non_nan_rank_iis], n_clones)
            else:
                if (self.state_centers is None) or (self.distance_metric is None):
                    states_to_simulate = _unbias_state_selection(
                        unique_states[non_nan_rank_iis],
                        rankings[non_nan_rank_iis], n_clones,
                        select_max=self.maximize_ranking)
                else:
                    states_to_simulate = _select_states_spreading(
                        rankings[non_nan_rank_iis],
                        unique_states[non_nan_rank_iis], n_clones,
                        centers=self.state_centers,
                        distance_metric=self.distance_metric,
                        select_max=self.maximize_ranking, width=self.width)
        return states_to_simulate


class page_ranking(base_ranking):
    """page ranking. ri = (1-d)*init_ranks + d*aij"""

    def __init__(
            self, d, init_pops=True, max_iters=100000, norm=True,
            spreading=False, maximize_ranking=True, **kwargs):
        """
        Parameters
        ----------
        d : float
            The weight of page ranks [0, 1]. A value of 1 is pure page rank
            and 0 is all the initial ranks.
        init_pops : bool, default=True
            Optionally uses the populations within an MSM as the initial ranks.
        max_iters : int, default=100000
            The maximum number of iterations to check for convergence.
        norm : bool, default=True
            Normilizes output ranks
        spreading : bool, default = False
            Solves for page ranks with the transpose of aij.
        """
        self.d = d
        self.init_pops = init_pops
        self.max_iters = max_iters
        self.norm = norm
        self.spreading = spreading
        base_ranking.__init__(
            self, maximize_ranking=maximize_ranking, **kwargs)

    @property
    def class_name(self):
        return "page_ranking"

    @property
    def config(self):
        return {
            'd': self.d,
            'init_pops': self.init_pops,
            'max_iters': self.max_iters,
            'norm': self.norm,
            'spreading': self.spreading,
            'maximize_ranking': self.maximize_ranking,
        }

    def rank(self, msm, unique_states=None):
        # generate aij matrix
        if unique_states is None:
            unique_states = get_unique_states(msm)
        tcounts = msm.tcounts_
        tcounts_sub = tcounts.tocsr()[:, unique_states][unique_states, :]
        aij = generate_aij(tcounts_sub)
        # determine the initial ranks
        if self.init_pops:
            Pi = msm.eq_probs_[unique_states]
        else:
            Pi = None
        rankings = rank_aij(
            aij, d=self.d, Pi=Pi, max_iters=self.max_iters, norm=self.norm)
        return rankings


class counts(base_ranking):
    """Min-counts ranking object. Ranks states based on their raw
    counts."""

    def __init__(self, maximize_ranking=False, scaling=None, **kwargs):
        self.scaling = scaling
        base_ranking.__init__(
            self, maximize_ranking=maximize_ranking, **kwargs)

    @property
    def class_name(self):
        return "counts"

    @property
    def config(self):
        return {
            'maximize_ranking': self.maximize_ranking,
        }

    def rank(self, msm, unique_states=None):
        counts_per_state = np.array(msm.tcounts_.sum(axis=1)).flatten()
        if unique_states is None:
            unique_states = np.where(counts_per_state > 0)[0]
        counts_return = counts_per_state[unique_states]
        if self.scaling is not None:
            counts_return = self.scaling.scale(counts_return)
        return counts_return


class FAST(base_ranking):
    """FAST ranking object"""

    def __init__(
            self, state_rankings=None,
            directed_scaling = scalings.feature_scale(maximize=True),
            statistical_component = counts(),
            statistical_scaling = scalings.feature_scale(maximize=False),
            alpha = 1, alpha_percent=False, maximize_ranking=True,
            **kwargs):
        """
        Parameters
        ----------
        state_rankings : array, shape = (n_states, )
            An array with ranking values for each state. This is
            previously calculated for each state in the MSM.
        directed_scaling : scaling object, default = feature_scale(maximize=True)
            An object used for scaling the directed component values.
        statistical_component : ranking function
            A function that has an enspara msm object as input, and
            returns the unique states and statistical rankings per
            state.
        statistical_scaling : scaling object, default = feature_scale(maximize=False)
            An object used for scaling the statistical component values.
        alpha : float, default = 1
            The weighting of the statistical component to FAST.
            i.e. r_i = directed + alpha * undirected
        alpha_percent : bool, default=False
            Optionally treat the alpha value as a percent.
            i.e. r_i = (1 - alpha) * directed + alpha * undirected
        """
        self.state_rankings = state_rankings
        self.directed_scaling = directed_scaling
        self.statistical_component = statistical_component
        self.statistical_scaling = statistical_scaling
        self.alpha = alpha
        self.alpha_percent = alpha_percent
        if self.alpha_percent and ((self.alpha < 0) or (self.alpha > 1)):
            raise ImproperlyConfigured(
                'alpha_percent is selected, although alpha is not between 0 and 1!')
        base_ranking.__init__(
            self, maximize_ranking=maximize_ranking, **kwargs)

    @property
    def class_name(self):
        return "FAST"

    @property
    def config(self):
        return {
            'state_ranking': self.state_rankings,
            'directed_scaling': self.directed_scaling,
            'statistical_component': self.statistical_component,
            'statistical_scaling': self.statistical_scaling,
            'alpha': self.alpha,
            'alpha_percent': self.alpha_percent,
            'maximize_ranking': self.maximize_ranking
        }

    def rank(self, msm, unique_states=None):
        # determine unique states
        if unique_states is None:
            unique_states = get_unique_states(msm)
        if self.statistical_component is None:
            statistical_weights = np.zeros(unique_states.shape)
        else:
            # get statistical component
            statistical_ranking = self.statistical_component.rank(msm)
            # scale the statistical weights
            statistical_weights = self.statistical_scaling.scale(
                statistical_ranking)
        # scale the directed weights
        directed_weights = self.directed_scaling.scale(
            self.state_rankings[unique_states])
        # determine rankings
        if self.alpha_percent:
            total_rankings = (1-self.alpha)*directed_weights + \
                self.alpha*statistical_weights
        else:
            total_rankings = directed_weights + self.alpha*statistical_weights
        return total_rankings


class string(base_ranking):
    """Uses the string method with MSMs to relax pathway. Samples from
    states along the highest flux pathway between start-states and
    end-states. Uses the statistical component to rank the states on
    this pathway.

    Parameters
    ----------
    start_states : int or array-like, shape = (n_start_states, )
        The starting states for defining the pathway.
    end_states : int of array-like, shape = (n_end_states, )
        The ending states for defining the pathway.
    statistical_component : ranking function
        A ranking class object to rank the pathway states. If none is
        selected, evens is used.
    maximize_ranking : bool, default=False,
        Optionally maximize the ranking. This will favor states with
        high statistical components, i.e. favor states with high
        counts (unlikely to be desireable).
    """
    def __init__(
            self, start_states, end_states, statistical_component=None,
            maximize_ranking=False, **kwargs):
        self.start_states = start_states
        self.end_states = end_states
        self.statistical_component = statistical_component
        base_ranking.__init__(
            self, maximize_ranking=maximize_ranking, **kwargs)

    @property
    def class_name(self):
        return "string"

    @property
    def config(self):
        return {
            'start_states': self.start_states,
            'end_states': self.end_states,
            'statistical_component': self.statistical_component,
            'maximize_ranking': self.maximize_ranking,
        }

    def rank(self, msm, unique_states=None):
        # determine unique states
        if unique_states is None:
            unique_states = get_unique_states(msm)
        if self.statistical_component is None:
            statistical_ranking = np.zeros(unique_states.shape)
        else:
            # get statistical component
            statistical_ranking = self.statistical_component.rank(msm)
        # determine the highest flux pathway between states
        if spar.issparse(msm.tprobs_):
            tprobs = np.array(msm.tprobs_.todense())
        else:
            tprobs = msm.tprobs_
        nfm = enspara.tpt.net_fluxes(
            tprobs, self.start_states,
            self.end_states, populations=msm.eq_probs_)
        path, flux = msmbuilder.tpt.top_path(
            self.start_states, self.end_states, nfm)
        # make all non-pathway states `nan`
        path_states = np.unique(path.flatten())
        path_iis = np.array(
            [
                np.where(unique_states == path_state)[0][0]
                for path_state in path])
        non_path_iis = np.setdiff1d(range(len(unique_states)), path_iis)
        new_rankings = np.array(np.copy(statistical_ranking), dtype=float)
        new_rankings[non_path_iis] = np.nan
        return new_rankings

