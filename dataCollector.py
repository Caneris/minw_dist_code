import numpy as np

def get_gini(wages):
    mask = wages > 0
    wages = wages[mask]
    wages = np.sort(wages)
    n = wages.size
    indices = np.arange(1, n + 1)
    return ((np.sum((2*indices - n - 1)*wages))/(n*np.sum(wages)))


def data_collector(n_refin, t, H, data_mat, emp_mat, f_float_mat, h_float_mat,
                   default_arr, skill_mat, w_dist_mat):

    # collect GDP, Unemployment rate, mean wages
    # open vacancies, share of defaulted firms, decile rations,
    # mean markups, aggregate consumption etc.


    active_arr = np.invert(default_arr)

    m_w = h_float_mat[0] > 0
    wages = h_float_mat[0, m_w]

    m_u = skill_mat[0, m_w]
    m_s = skill_mat[1, m_w]

    # unemployment rate
    data_mat[0, t] = 1 - np.sum(emp_mat) / H
    # nominal GDP
    data_mat[1, t] = np.sum(f_float_mat[17]*f_float_mat[8]) # prices time goods produced
    Y = f_float_mat[8].sum()
    # real GDP
    data_mat[2, t] = Y
    # mean prices
    data_mat[3, t] = data_mat[1, t]/Y # GDP/produced goods
    # wage distribution
    w_dist_mat[t, :] = h_float_mat[0]
    # mean nominal wages
    data_mat[4, t] = wages.mean()
    # median real wages
    data_mat[5, t] = np.median(wages)/data_mat[3,t]
    # mean real wages
    data_mat[6, t] = data_mat[4,t]/data_mat[3,t]
    # Gini coefficient
    data_mat[7, t] = get_gini(wages)
    # share of inactive firms
    n_def = np.sum(default_arr)
    data_mat[8, t] = n_def / default_arr.size
    # share of refinanced firms
    data_mat[9, t] = n_refin[t] / default_arr.size
    # mean u real wages
    data_mat[10, t] = np.mean(wages[m_u])/data_mat[3, t]
    # mean s real wages
    data_mat[11, t] = np.mean(wages[m_s])/data_mat[3, t]
    # wage ratio s/u
    data_mat[12, t] = data_mat[11, t]/data_mat[10, t]
    # aggregate consumption
    data_mat[13, t] = np.sum(h_float_mat[4])
    # desired consumption
    data_mat[14, t] = np.sum(h_float_mat[5])
    # total expenditure
    data_mat[15, t] = np.sum(h_float_mat[6])
    # average households wealth and average liquid assets of firms
    mean_Ah = np.mean(h_float_mat[9])
    mean_Af = np.mean(f_float_mat[16])
    # average wealth ratio firm to household
    data_mat[16, t] = mean_Af/mean_Ah
    # mean mark-ups
    data_mat[17, t] = np.sum(f_float_mat[11]*f_float_mat[10]*f_float_mat[8])/Y
    # mean unit-costs
    data_mat[18, t] = np.sum(f_float_mat[10]*f_float_mat[8])/Y
    # delta
    data_mat[19, t] = np.mean(f_float_mat[15, active_arr])
    # total inventories
    data_mat[20, t] = np.sum(f_float_mat[12])
    # unemployment unskilled
    data_mat[21, t] = 1 - np.sum(emp_mat[:,skill_mat[0]])/np.sum(skill_mat[0])
    # unemployment skilled
    data_mat[22, t] = 1 - np.sum(emp_mat[:,skill_mat[1]])/np.sum(skill_mat[1])