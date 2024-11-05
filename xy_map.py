import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib
import map_utils as ut


"""
Generalized fitness - Complexity algorithm for ranking nodes in bipartite systems.
"""


class XYMap:
    """
    Class which runs the generalized fitness-complexity algorithm for binary matrices. 
    The input matrix must be a pandas.DataFrame or a 2D numpy.
    We refet to x as the score for the first layer of the network (nodes corresponding to the rows of the matrix) and y to the second score.
    Initial conditions fixed to vectors of ones.
    
    Attributes:
        mat_frame (pandas.Dataframe): dataframe storing the adjacency matrix
        exp (float): map exponend of the last run
        x_traj (2d numpy.array): list of the x scores for each iteration of the map
        y_traj (2d numpy.array): list of the y scores for each iteration of the map
        c_traj (2d numpy.array): list of the inverse y scores for each iteration of the map
    """
    

    def __init__(self, mat, **params_kw):
        """
        Arguments:
            mat : 2d numpy array or DataFrame for the adjacency matrix of the bipartite network.
            
        params_kw (parameters can be changed also from the params dicionary):
            delta_conv (float): threshold for the average x change below which the algorithm reaches the stationary state. 
            t_max (int): max number of the map iterations.
            low_bound (float): threshold under which x and y are  considered zero.       
            print_info (bool): if the algorithm is verbose
        """
        
        if isinstance(mat, pd.DataFrame): self.mat_frame = mat
        else: self.mat_frame = pd.DataFrame(mat)
        self.n_row, self.n_col = np.array(mat).shape
        self.exp = None
        self.x_traj, self.c_traj, self.y_traj = None, None, None
        self.x_ranking, self.y_ranking = None, None
        self._set_params(**params_kw)
        self._row_ind_list, self._col_ind_list = self._get_index_lists(np.array(mat))

        
    def run(self, expn):
        """
        Single run of the map at given exponent
        """
        
        # Init to vector of ones
        self.x_traj, self.c_traj, self.y_traj = [np.ones(self.n_row)], [np.ones(self.n_col)], [np.ones(self.n_col)]
        self.exp = expn
        
        for t in range(int(self.params['t_max'])):
            
            # The update is done with the inverse y score in case of negative exponent.
            # This makes the algorithm more robust to numerical approximations
            if self.exp >= 0: 
                self._one_step_positive_exp()
            else: 
                self._one_step_negative_exp()
            
            # Checking the convergence
            da = sum([abs(self.x_traj[t][i] - self.x_traj[t-2][i]) for i in range(self.n_row)]) / self.n_row
            if t > 1 and da < self.params['delta_conv']:
                break
                    
        if self.params['print_info']:
            print ("Convergence in " + str(len(self.x_traj)) + " time steps.")
        
        
    def get_stat(self, axis, th_stat=10**(-2)):
        """
        Gets the fraction of positive stationary trajectories. Can be called after a run of the map.
        
        Arguments:
            axis: (int) 0 for x scores (rows), 1 for y scores (columns), 1 for inverse y scores or complexities (columns)
        """
        
        if self.exp == None:
            print ("You have to perform a single run first")
            return
        
        if axis == 0:
            traj = self.x_traj
        elif axis == 1:
            traj = self.y_traj
        else:
            traj = self.c_traj
            
        low_boundary = self.params['low_bound']*(10**10)
        t_steps = len(traj)-1
        stat = 0
        for i in range(len(traj[0])):
            # zeros under the lower boundary
            if (traj[t_steps][i] < low_boundary) and (traj[t_steps - 1][i] < low_boundary):
                continue
            # stationary (stat_th) over period 2
            elif abs(np.log(traj[t_steps][i]) - np.log(traj[t_steps - 2][i])) < th_stat:
                # stationary over period 1
                if abs(np.log(traj[t_steps][i]) - np.log(traj[t_steps - 1][i])) < th_stat:
                    stat += 1
        return stat

    
    def get_ranking(self, axis, return_obj=True):
        """
        Return the nodes ranking
        
        Arguments:
            axis: (int) 0 for rows, 1 for columns
            return_obj: (bool) true: returns ranked objects, false: returns the ranked indexes
        """
        
        if self.exp == None:
            print ("You have to perform a single run first")
            return
        
        if axis == 0:
            traj = self.x_traj
            obj = self.mat_frame.index.values
        else:
            traj = self.c_traj
            obj = self.mat_frame.columns
        sensitive_boundary = self.params['low_bound'] * 10**20
        last_values = np.array(traj[-1])
        x_above_lowb = np.array([[i[0], x] for i, x in np.ndenumerate(last_values) if x > sensitive_boundary])
        ind = np.argsort(x_above_lowb[:, 1])[::-1]
        ranking = np.array([int(index) for index in x_above_lowb[ind][:, 0]])
        x_below_lowb = np.array([i[0] for i, x in np.ndenumerate(last_values) if x <= sensitive_boundary])
        step = -1
        while True:
            step -= 1
            aux_index_set = []
            for i, index in np.ndenumerate(x_below_lowb):
                if traj[step][index] > sensitive_boundary and index not in ranking:
                    aux_index_set.append([index, traj[step][index]])
            if len(aux_index_set) > 0:
                ind = np.argsort(np.array(aux_index_set)[:, 1])[::-1]
                for i in ind:
                    ranking = np.append(ranking, int(aux_index_set[i][0]))
            if len(ranking) == len(traj[0]) or step <= -len(traj):
                break
        if not return_obj:
            return ranking
        if return_obj:
            return np.array([obj[i] for i in ranking])
    
    
    def ext_area(self, axis, expn):
        """
        Compute the extinction removing rows (axis=0) or columns (axis=1) for the ranking at given exponent
        """
        self.run(expn)
        if axis == 1:
            rank = self.get_ranking(axis, return_obj=False)[::-1]
        else:
            rank = self.get_ranking(axis, return_obj=False)
        return ut.ext_area(axis, rank, (self._row_ind_list, self._col_ind_list))
            
        
    def find_best_ext_area(self, exp_start=-1.5, exp_end=-0.8, n_exp=100):
        """
        Find the best extinction area removing rows with a simple grid search in the specified exponent range.
        To compute the one removing columns it is suggested to define a new XYMap with the transpose of the matrix.
        It returns the best exponent and the best extinction area
        """
        self._set_params(print_info=False)
        expns = np.linspace(exp_start, exp_end, n_exp)
        return ut.find_max_grid(lambda x : self.ext_area(0, x), expns)
    
        
    def nest_temp(self, expn):
        """
        Compute the nested temperature for the ranking at given exponent. 
        It returns the temperature and the matrix ordered with the obtained ranking.
        """
        self.run(expn)
        rank_row = np.array(self.get_ranking(0))
        rank_col = np.array(self.get_ranking(1))[::-1]
        sort_mat = self.mat_frame.loc[rank_row, rank_col]
        T = ut.compute_nestedness_T(sort_mat.to_numpy())
        return T, sort_mat
    
    
    def find_best_nest_temp(self, exp_start=-1.5, exp_end=0.0, n_exp=150):
        """
        Find the best nested temperature with a simple grid search in the specified exponent range.
        It returns the best exponent and the best minimum temperature
        """
        self._set_params(print_info=False)
        expns = np.linspace(exp_start, exp_end, n_exp)
        gamma, T = ut.find_max_grid(lambda x : -self.nest_temp(x)[0], expns)
        return gamma, -T


    def plot_traj(self, axis, ax=None, color_by_size=True):
        """
        Plot the x score (axis=0), y_score (axis=1) or inverse y_score (axis=2) trajectories
        """
        if axis == 0:
            traj = self.x_traj
            size = self.mat_frame.sum(axis=1).values
            label = 'Row size'
        elif axis == 1:
            traj = self.y_traj
            size = self.mat_frame.sum(axis=0).values
            label = 'Column size'
        else:
            traj = self.c_traj
            size = self.mat_frame.sum(axis=0).values
            label = 'Column size'
            
        if color_by_size:
            return self._plot_traj(traj, ax, obs_to_color=size, color_label=label, x_range=[0, len(self.x_traj)])
        else:
            return self._plot_traj(traj, ax, x_range=[0, len(self.x_traj)])
        
        
    ###################################
    #### Private auxiliary methods ####
    ###################################
        
    
    def _one_step_positive_exp(self):
        """ 
        Step of the map that uses the y score for positive exponents
        """        
        # List of exponentiated x and y of the previous step
        x_exp, y_exp = self.x_traj[-1]**self.exp, self.y_traj[-1]**self.exp
        # Building the new x list and checking the low boundary
        x_new = np.array([sum(np.take(y_exp, self._row_ind_list[i])) for i in range(self.n_row)])
        x_new[x_new < self.params['low_bound']] = self.params['low_bound']
        # Building the new y
        y_new = np.array([sum(np.take(x_exp, self._col_ind_list[j])) for j in range(self.n_col)])
        # Normalizing
        self.x_traj = np.concatenate((self.x_traj, [x_new/np.mean(x_new)]))
        self.y_traj = np.concatenate((self.y_traj, [y_new/np.mean(y_new)]))
        c_aux = 1/y_new    
        self.c_traj = np.concatenate((self.c_traj, [c_aux/np.mean(c_aux)]))
                                                              
            
    def _one_step_negative_exp(self):
        """ 
        Step of the map that uses the inverse y score for negative exponents
        """        
        # List of exponentiated x and y of the previous step
        x_exp, c_exp = self.x_traj[-1]**self.exp, self.c_traj[-1]**(-self.exp)
        # Building the new x list and checking the low boundary
        x_new = np.array([sum(np.take(c_exp, self._row_ind_list[i])) for i in range(self.n_row)])
        x_new[x_new < self.params['low_bound']] = self.params['low_bound']
        # Building the new c list
        c_new = np.array([1/sum(np.take(x_exp, self._col_ind_list[j])) for j in range(self.n_col)])
        # Normalizing
        self.x_traj = np.concatenate((self.x_traj, [x_new/np.mean(x_new)]))
        self.c_traj = np.concatenate((self.c_traj, [c_new/np.mean(c_new)]))
        y_new = 1/c_new
        self.y_traj = np.concatenate((self.y_traj, [y_new/np.mean(y_new)]))
            
        
    def _get_index_lists(self, mat):
        """ 
        Get two lists (for rows and columns) where the row/col index gives the
        list of col/row indexes of the non-zero elements.
        """
        n_row, n_col = mat.shape
        
        col_ind_at_row, row_ind_at_col = [],[]
        for i in range(n_row):
            aux_ind = np.where(mat[i]>0)[0]
            if len(aux_ind) == 0:
                raise Exception('Row {} is composed of zeros'.format(i))
            col_ind_at_row.append(aux_ind)
            
        for j in range(n_col):
            aux_ind = np.where(mat[:,j]>0)[0]
            if len(aux_ind) == 0:
                raise Exception('Column {} is composed of zeros'.format(j))
                
            row_ind_at_col.append(aux_ind)
            
        return col_ind_at_row, row_ind_at_col 
    
    
    def _set_params(self, **params_kw):
        
        self.params = {
            'delta_conv' : 10**(-8),
            't_max' : 1000,
            'low_bound' : 10**(-100),
            'print_info' : True
        }
        
        if params_kw is not None:
            for k,val in params_kw.items():
                if k in self.params:
                    self.params[k] = val
                else:
                    raise Exception("Non valid parameter " + str(k))
    

    def _plot_traj(self, traj, ax, obs_to_color=[], color_label='Label', y_range=False, 
                  x_range=False, y_label='x score', y_log=True, l_width=1.2):

        if len(obs_to_color) == 0:
            obs_to_color = np.linspace(0, np.shape(traj)[1], np.shape(traj)[1])
        cmap = plt.cm.jet
        norm = matplotlib.colors.Normalize(vmin=min(obs_to_color), vmax=max(obs_to_color))
        smap = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        smap.set_array([])

        if ax == None:
            fig, ax = plt.subplots()

        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        if not x_range != False:
            ax.set_xlim(x_range)
        if y_range != False:
            ax.set_ylim(y_range)
        if y_log == True:
            ax.set_yscale('log')

        for i in range(np.shape(traj)[1]):
            ax.plot(traj[:, i], color=smap.to_rgba(obs_to_color[i]), lw=l_width)

        cbar = fig.colorbar(smap)
        cbar.set_label(color_label, fontsize=14)

        return ax




