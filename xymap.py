#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:04:07 2020

@author: andrea mazzolini
"""

import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib as _mpl
#import sys as _sys


class XYMap:
    """
    Class which runs the generalized fitness-complexity algorithm for binary 
    matrices. If the matrix is not binary, it is automatically binarized 
    putting 1 when the element is larger than 0.
    The adjacency matrix must be a two-dimensional numpy array.
    We refet to x as the score for the first layer of the network (nodes
    corresponding to the rows of the matrix) and y to the second score.
    Initial conditions fixed to vectors of ones.
    The parameters of the algorithm can be found in the 'params' dictionary.
    """
    
    def __init__(self, adj_mat, **params_kw):
        """
        Arguments:
            
            adj_mat : 2d numpy array defining the adjacency matrix of the 
            bipartite network.
            
        params_kw (parameters can be changed also prof the params dicionary):
            
            delta_conv (float): threshold for the average x change below which
            the algorithm reaches the stationary state. 
        
            t_max (int): max number of the map iterations.
        
            eps (float): threshold under which x and y are 
            considered zero.       
            
            print_info (bool): if the algorithm is verbose
        """
        
        if not isinstance(adj_mat, _np.ndarray) or len(adj_mat.shape) != 2:
            raise Exception ("Invalid matrix type. Must be a 2d array.")
        
        self._set_params(**params_kw)
        
        # Map between row/col indexes and list of linked non-zero col/row indexes
        self._neighb = self._get_index_lists(adj_mat)
        # Dimensions of the network
        self.d = len(self._neighb[0]), len(self._neighb[1])
        
        
    def run(self, axis, gamma):
        """
        Run the algorithm for the scores in the axis layer (0 for x, 1 for y).
        This score and the associated ranking are reliable, the ranking of the
        opposite score is not.
        """
    
        # Trajectories of the main score to compute ant the opporite one
        traj_s, traj_o = [_np.ones(self.d[axis])], [_np.ones(self.d[1-axis])]
        # Ranked indices of the scores
        rank_s, rank_o = _np.array([], dtype=int), _np.array([], dtype=int)
        # List of node indices that have reached the zero threshold
        zeros_s, zeros_o = _np.array([], dtype=int), _np.array([], dtype=int)

        # Main loop
        for t in range(int(self.params['t_max'])):
            
            # Computing the opposite score without approx
            o = self._one_step(gamma, 1-axis, traj_s[-1])
            rank_o, zeros_o = self._update_zero_rank(o, zeros_o, rank_o)
            traj_o = _np.concatenate((traj_o, [o]))
            
            # Computing the main score (given the opposite one) without approx
            s = self._one_step(gamma, axis, o)
            rank_s, zeros_s = self._update_zero_rank(s, zeros_s, rank_s)
            # Imposing the threshold to the score
            s[zeros_s] = self.params['low_bound']
            traj_s = _np.concatenate((traj_s, [s]))
            
            # Checking the convergence
            if self._converg_check(axis, t, traj_s):
                break

        # Finalize the ranking of the positive scores
        rank_s = _np.append(rank_s, _np.argsort(s)[len(zeros_s):])[::-1]
        rank_o = _np.append(rank_o, _np.argsort(o)[len(zeros_o):])[::-1]

        # Update the class variables
        self._update_vars(axis, traj_s, traj_o, rank_s, rank_o, t)
        
    
    def plot_traj(self, axis, l_width=1.2):
        
        if axis==0:
            traj = self.x_traj
            y_label = '$x$ score'
        else:
            traj = self.y_traj
            y_label = '$y$ score'
            
        obs_to_color = _np.array([len(ind) for ind in self._neighb[axis]])
        cmap = _plt.cm.jet
        norm = _mpl.colors.Normalize(vmin=min(obs_to_color), vmax=max(obs_to_color))
        smap = _mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        smap.set_array([])
        
        fig, ax = _plt.subplots()
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_yscale('log')
    
        for i in range(_np.shape(traj)[1]):
            ax.plot(traj[:, i], color=smap.to_rgba(obs_to_color[i]), lw=l_width)
    
        cbar = fig.colorbar(smap)
        cbar.set_label('Node degree', fontsize=14)
        
        return fig, ax
    
                
    def _one_step(self, gamma, axis, opp_scores):
        """
        Compute one step of the map for a score (of the given axis and with 
        given neighbours) as a function of the opposite score
        """
        opp_exp = opp_scores**gamma
        s = _np.array([])
        for i in range(self.d[axis]):
            s = _np.append(s, _np.take(opp_exp, self._neighb[axis][i]).sum())
        return s/_np.mean(s)
    
    def _update_zero_rank(self, s, zero_ind, rank):
        """
        Given the scores s, the list of nodes that already reached the low
        boundary and the ranking computed so far, check if new nodes have 
        reached the low boundary, and, if so, updated the ranking with those
        """
        lb = self.params['low_bound']
        new_zeros_ind = _np.setdiff1d(_np.nonzero(s <= lb), zero_ind)
        if len(new_zeros_ind) > 0:
            sorted_zeros_ind = new_zeros_ind[_np.argsort(s[new_zeros_ind])]
            rank = _np.append(rank, sorted_zeros_ind)
            zero_ind = _np.append(zero_ind, new_zeros_ind)
        return rank, zero_ind
    
    def _converg_check(self, axis, t, traj):
        if t > 1:
            delta = _np.abs(traj[t] - traj[t-1]).sum() / self.d[axis]
            if delta < self.params['delta_conv']:
                return True
        return False
    
    def _update_vars(self, axis, traj_s, traj_o, rank_s, rank_o, t):  
        """
        Update the class variables after the algorithm execution
        """
        if axis == 0:
            self.x_traj = traj_s
            self.x_ranking = rank_s
            self.x_scores = traj_s[-1]
            self.inverse_y_traj = traj_o
            self.inverse_y_ranking = rank_o
        else:
            self.y_traj = traj_s
            self.y_ranking = rank_s
            self.y_scores = traj_s[-1]
            self.inverse_x_traj = traj_o 
            self.inverse_x_ranking = rank_o
            
        if self.params['print_info']:
            t = len(traj_s)-1
            print ("Convergence in " + str(t) + " time steps.")
            if t >= self.params['t_max']:
                print("Warning. Stationary state not reached.")
            
    def _set_params(self, **params_kw):
        
        self.params = {
            'delta_conv' : 10**(-8),
            't_max' : 1000,
            'low_bound' : 10**(-30),
            'print_info' : True
        }
        
        if params_kw is not None:
            for k,val in params_kw.items():
                if k in self.params:
                    self.params[k] = val
                else:
                    raise Exception("Non valid parameter " + str(k))
                    
                    
    def _get_index_lists(self, mat):
        """ 
        Get two lists (for rows and columns) where the row/col index gives the
        list of col/row indexes of the non-zero elements.
        """
        n_row, n_col = mat.shape
        
        col_ind_at_row, row_ind_at_col = [],[]
        for i in range(n_row):
            aux_ind = _np.where(mat[i]>0)[0]
            if len(aux_ind) == 0:
                raise Exception('Row {} is composed of zeros'.format(i))
            col_ind_at_row.append(aux_ind)
            
        for j in range(n_col):
            aux_ind = _np.where(mat[:,j]>0)[0]
            if len(aux_ind) == 0:
                raise Exception('Column {} is composed of zeros'.format(j))
                
            row_ind_at_col.append(aux_ind)
            
        return col_ind_at_row, row_ind_at_col  
    
    


    

    
