import numpy as np
import numpy.random as rd
from tools import get_i_int_i_skill, Update_N, update_v, vec_mat_mul
from numba import njit



def firms_fire_workers(worker_type, f_ids, h_ids, f_int_mat,
                       h_float_mat, fired_arr, emp_mat, skill_mat):
    """
    Firms fire workers. The fired workers loose their jobs in the subsequent period.

    :param worker_type (str): 'unskilled' or 'skilled'
    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param h_ids (ndarray): 1D array containing household IDs with 'int' type.
    :param f_int_mat (ndarray): 2D array containing firm data with 'int' type.
    :param h_float_mat (ndarray): 2D array containing household data with 'int' type.
    :param fired_arr (ndarray): 1D array containing firing data with 'bool' type.
    :param emp_mat (ndarray): 2D array containing employment data with 'bool' type.
    :param skill_mat (ndarray): 2D array containing skill type data with 'bool' type.
    """

    i_int, i_skill = get_i_int_i_skill(worker_type)

    wages = h_float_mat[0]
    # vacancies with negative values
    m_s = skill_mat[i_skill]
    vacancies = f_int_mat[i_int]
    f_mask = vacancies < 0
    val = f_mask.sum()
    fire_arr = vacancies[f_mask]
    ids = f_ids[f_mask]  # ids of firms that want to fire workers
    n_fire_arr = np.abs(fire_arr)

    if val > 0: # if any firm wants to fire
        for i in range(len(ids)):
            # get id of the firm, and number of workers it wants to fire
            f_id, n = int(ids[i]), int(n_fire_arr[i])
            emp_row = emp_mat[f_id, :]
            emp_mask = emp_row > 0
            # get employees
            cond = np.logical_and(emp_mask, m_s)
            emp_ids = h_ids[cond]
            # look at wages of the employees
            w_arr = wages[emp_ids]
            # take indices of employees with highest wages
            mask = np.argsort(w_arr)[-n:]
            fired_ids = emp_ids[mask]
            fired_arr[fired_ids] = True


def Pr_LM(w_old, w_new, lambda_LM):
    """
    Returns the probabilities that already employed workers
    switche their employers.

    :param w_old (float): current wage
    :param w_new (float): wage offered by potential employer
    :param lambda_LM (float): loyalty parameter
    :return (float): switch probability
    """
    diff = (w_old - w_new)/w_old
    result = np.zeros(len(diff))
    cond = diff < 0
    result[cond] = 1 - np.exp(np.asarray([lambda_LM])*diff[cond])
    return result


def draw_ones(P):
    """
    Draws ones given the probabilities in array 'P'.

    :param P (ndarray): 1D array containing switch probabilities with 'float' type.
    :return (ndarray): 1D array containing ones drawn from uniform distribution given 'P'.
    """

    num = rd.rand(len(P)) # draw random numbers between 0 and 1
    return np.asarray([0])*(num >= P) + np.asarray([1])*(num<P) # '1' if greater than probability, 0 otherwise.


def households_get_employed(h_ids, f_id, min_w, emp_row, vacancies, wages, d_wages, fired_time):
    """
    Adjusts variables after households get employed by a firm.
    """

    # number of hoseholds that get employed
    n = h_ids.size
    emp_row[h_ids] = True
    vacancies[f_id] -= n  # correct vacancies
    # update wages
    wages[h_ids] = np.maximum(d_wages[h_ids], min_w)
    # update desired wage in case of minimum wage
    d_wages[h_ids] = wages[h_ids]
    fired_time[h_ids] = 0


def firm_employs_applicants(f_id, chosen_apps, min_w, vacancies, h_bool_mat,
                            wages, d_wages, app_mat, emp_mat, fired_time, lambda_LM):
    """
    Chosen applicants get offer. Unemployed applicants always accept the offer, employed applicants accept the offer
    with a probability based the difference between current and potential wage and the loyalty parameter' lambda_LM'
    (the switch probability is returned by the function 'Pr_LM()').

    :param chosen_apps (ndarray): 1D array containing IDs of chosen applicants with 'int' type.
    """

    job_offers = h_bool_mat[0]
    job_offers[chosen_apps] = True  # chosen applicants get job offer
    app_mat[:, chosen_apps] = 0  # delete all applications of chosen applicants
    x = np.sum(emp_mat[:, chosen_apps], axis=0)  # choose already employed workers
    switch_cond = x > 0
    emp_row = emp_mat[f_id]

    # employ unemployed applicants
    unemp_mask = np.invert(switch_cond)
    unemp_ids = chosen_apps[unemp_mask]

    # unemployed households get employed
    households_get_employed(unemp_ids, f_id, min_w, emp_row, vacancies, wages, d_wages, fired_time)

    if np.sum(switch_cond) > 0:

        switch_ids = chosen_apps[switch_cond]  # ids of workers that might switch
        w_old = wages[switch_ids]
        w_new = d_wages[switch_ids]
        # get switch probability
        Pr = Pr_LM(w_old, w_new, lambda_LM)  # switch probabilites

        switch_mask = draw_ones(Pr) > 0

        if np.sum(switch_mask) > 0:
            switch_id_arr = switch_ids[switch_mask]  # workers that are switching
            # switching households quit current job
            emp_mat[:, switch_id_arr] = np.zeros(emp_mat[:, switch_id_arr].shape)
            households_get_employed(switch_id_arr, f_id, min_w, emp_row, vacancies, wages, d_wages,
                                    fired_time)


def firms_employ_applicants(rand_f_ids, v_arr, app_mat, m_skill, h_ids, d_wages, min_w, vacancies,
                            h_bool_mat, wages, emp_mat, fired_time, lambda_LM):
    """
    Goes trough the shuffled list of firms and chooses applicants demanding the lowest wages.
    """

    for i in range(len(rand_f_ids)):
        f_id = np.int(rand_f_ids[i])
        v = int(v_arr[f_id])

        # cond = (v > 0) * np.sum(app_mat[f_id, skill_mat[0]]) > 0

        # get ids of applicants
        f_app_row = app_mat[f_id, :]
        mask = f_app_row[m_skill]
        h_app_ids = h_ids[m_skill][mask]
        sorted_app_ids = np.argsort(d_wages[h_app_ids])  # sort from lowest to highest
        chosen_apps = h_app_ids[sorted_app_ids][0:v]  # take the v cheapest offers

        # employ applicants
        firm_employs_applicants(f_id, chosen_apps, min_w, vacancies, h_bool_mat,
                                wages, d_wages, app_mat, emp_mat, fired_time, lambda_LM)


def lm_matching(worker_type, f_ids, h_ids, h_float_mat, f_int_mat, app_mat, skill_mat,
                min_w, h_bool_mat, emp_mat, fired_time, lambda_LM, t):
    """
    Runs labor market matching until no matching is possible.

    :param worker_type (str): 'unskilled' or 'skilled'.
    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param h_ids (ndarray): 1D array containing households IDs with 'int' type.
    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    :param f_int_mat (ndarray): 2D array containing firm data with 'int' type.
    :param app_mat (ndarray): 2D array containing application data with 'bool' type.
    :param skill_mat (ndarray): 2D array containing data with 'bool' type. Used for masking skill types.
    :param min_w (float): level of current minimum wage.
    :param h_bool_mat (ndarray): 2D array containing household data with 'bool' type.
    :param emp_mat (ndarray): 2D array containing employment data with 'bool' type.
    :param fired_time (ndarray): 1D array containing firing data with 'int' type.
    :param lambda_LM (float): loyalty parameter
    """

    i_int, i_skill = get_i_int_i_skill(worker_type)
    # get demanded wages
    d_wages = h_float_mat[1]
    wages = h_float_mat[0]
    vacancies = f_int_mat[np.int(i_int)]
    m_skill = skill_mat[np.int(i_skill)]
    skill_ids = h_ids[m_skill]

    val = True
    while val:
        # shuffle firms
        rand_f_ids = rd.choice(f_ids, len(f_ids), replace=False)
        v_arr = vacancies.copy()
        v_arr[vacancies < 0] = 0
        M = app_mat[:, skill_ids]
        arr = vec_mat_mul(v_arr, M)
        val = arr.sum()
        # if t == 854:
        #     print("arr[20]: {}".format(arr[20]))
        #     print("M[20,:]: {}".format(M[20,:]))
        #     print("v_arr[20]: {}".format(v_arr[20]))
        #     print("v_arr: {}".format(v_arr))
        #     print("arr: {}".format(arr))
        #     print("len arr: {}".format(len(arr)))


        firms_employ_applicants(rand_f_ids, v_arr, app_mat, m_skill, h_ids, d_wages, min_w,
                                vacancies, h_bool_mat, wages, emp_mat, fired_time, lambda_LM)

        # update N and v
        Update_N(f_int_mat, emp_mat, skill_mat)
        update_v(f_int_mat)


