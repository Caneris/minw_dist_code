import numpy as np
import numpy.random as rd
from numba import njit

@njit
def h_buy_goods(f_ids, supply, N_good, f_float_mat, h_float_mat, h_ids,
                demand, tol):

    price_arr = f_float_mat[17]
    while np.sum(demand > 0) and np.sum(supply > 0):
        h_rand_ids = rd.choice(h_ids, len(h_ids), replace=False) # shuffle households
        d_m = demand[h_rand_ids] > 0  # demand mask

        for h_id in h_rand_ids[d_m]:
            if np.sum(supply > 0) == 0 or np.sum(demand > 0) == 0:
                break
            subN = np.minimum(N_good, np.sum(supply > 0))
            f_rand_ids = rd.choice(f_ids[supply > 0], subN, replace=False) # choose random sample of firms
            prices = price_arr[f_rand_ids]
            ind = np.argsort(prices)[0]  # take cheapest price
            p = prices[ind]
            f_id = f_rand_ids[ind]
            # buy consumption good
            A_h = h_float_mat[9, h_id]
            c = np.array([A_h/p, demand[h_id], supply[f_id]]).min()
            expenditure = c * p

            if A_h > tol:
                h_float_mat[6, h_id] += expenditure
                h_float_mat[9, h_id] -= expenditure
                h_float_mat[4, h_id] += c
                f_float_mat[0, f_id] += c
                demand[h_id] -= c
                supply[f_id] -= c
                if h_float_mat[9, h_id] < tol:
                    h_float_mat[9, h_id] = 0
                if np.abs(demand[h_id]) < tol:
                    demand[h_id] = 0
                if np.abs(supply[f_id]) < tol:
                    supply[f_id] = 0
            else:
                demand[h_id] = 0


@njit
def goods_market_matching(h_float_mat, f_float_mat, h_ids, f_ids, N_good, tol):
    """
    Runs goods market matching.

    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param h_ids (ndarray): 1D array containing household IDs with 'int' type.
    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param N_good (int): number of firms observed by single households
    :param tol (float): tolerance level for zero values.
    """

    demand = np.copy(h_float_mat[5])
    supply = f_float_mat[8] + f_float_mat[12]
    h_buy_goods(f_ids, supply, N_good, f_float_mat,
                h_float_mat, h_ids, demand, tol)