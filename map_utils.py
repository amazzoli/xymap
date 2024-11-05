import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib
import copy
from scipy.optimize import brentq
from scipy.integrate import quad
import itertools as it


# GRID MAXIMUM SEARCH

def find_max_grid(func, xs):
    """
    Get the (average) values of xs at which func is maximized and the value of func at the maximum
    """
    f_vals = np.array([func(x) for x in xs])
    best_xs = np.array(xs)[np.array(f_vals) == max(f_vals)]
    return np.mean(best_xs), max(f_vals)


# EXTINCTION AREA

def ext_area(axis, rank, indexes_lists):
    """
    Compute the extinction areas of a binary matrix
    """
    
    if axis == 0:
        indexes_lists = copy.deepcopy(indexes_lists)
    else:
        indexes_lists = copy.deepcopy(indexes_lists[::-1])
        
    if len(rank) != len(indexes_lists[0]):
        print ('Dimensions do not match')
        return
    
    count = sum(1 for j in range(len(indexes_lists[1])) if len(indexes_lists[1][j]) == 0)
    traj = [count]
    for i in range(len(rank) - 1):
        for c in indexes_lists[0][rank[i]]:
            is_empty = True
            for k in range(len(indexes_lists[1][c])):
                if indexes_lists[1][c][k] == rank[i]:
                    indexes_lists[1][c][k] = -1
                else:
                    if indexes_lists[1][c][k] != -1:
                        is_empty = False
            if is_empty:
                count += 1
        traj.append(count)
        
    return sum(traj) / float(len(indexes_lists[0]) * len(indexes_lists[1]))


# NESTEDNESS MEASURES

def isocline(x, p, N, M):
    return np.real(0.5/N + (N-1)/N*( 1 - (1 - (M*x - 0.5)/(M - 1) )**p )**(1/p))


def area_above_isocline(p, N, M):
    return 1 - quad(lambda x : isocline(x, p, N, M), 0, 1)[0]


def get_parameter_p(dens, N, M, a, b):
    return brentq(lambda p : area_above_isocline(p, N, M) - dens, a, b)


def compute_nestedness_T(matrix):
    
    N, M = len(matrix), len(matrix[0])
    dens = np.sum(matrix) / M / N
    p_iso = get_parameter_p(dens, N, M, 0.1, 20)
    U = 0
    
    for i in range(1, N+1):
        for j in range(1, M+1):

            xA, yA = (j-0.5)/M, 1 - (i-0.5)/N
            is_up_isocline = isocline(xA, p_iso, N, M) < yA

            if (is_up_isocline and matrix[i-1,j-1] == 0) or (not is_up_isocline and matrix[i-1,j-1] == 1):
                A = np.array([xA,yA])
                
                if xA + yA < 1:
                    D = np.sqrt(2) * (xA + yA)
                else:
                    D = np.sqrt(2) * (2 - xA - yA)

                f = lambda x : isocline(x, p_iso, N, M) + x - xA - yA
                if abs(f(0.5/len(matrix[0]))) < 10**-10:
                    Bx = 0.5/len(matrix[0])
                else:
                    Bx = brentq(f, 0.5/len(matrix[0]), 1)
                By = xA + yA - Bx
                B = np.array([Bx, By])

                frac = np.linalg.norm(A - B)
                coef = (frac / D)**2
                U += coef
                #print(A, B, frac, D, coef)
            
    return 100 * U / 0.04145 / N / M


def compute_NODF(mat):
    row_idxs = [ set(np.where(i == 1)[0]) for i in mat ]
    col_idxs = [ set(np.where(i == 1)[0]) for i in mat.transpose() ]
    row_pairs = it.combinations(row_idxs, 2)
    col_pairs = it.combinations(col_idxs, 2)
    Np_row = [ 0 if len(l) >= len(u) else 100*len(u & l) / len(l) for u, l in row_pairs ]
    Np_col = [ 0 if len(l) >= len(u) else 100*len(u & l) / len(l) for u, l in col_pairs ]
    return np.mean(Np_row + Np_col)