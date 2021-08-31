import numpy as np
import numpy.random as rd
from numba import njit





################## GENERAL ##################


@njit
def set_seed(seed):
    """
    This function is used of seeding the jitted part of the code...

    :param seed (int): seed number
    """

    rd.seed(seed)

def expectation(z, z_e, lambda_exp):
    """
    This function returns the expected value for the
    variable z for the current period.

    :param z (float): previous period observation
    :param z_e (float): previous period expectation
    :param lambda_exp (float): adjustment parameter
    :return (float): current period expectation
    """

    error = z - z_e
    return z_e + lambda_exp*error

@njit
def vec_mat_mul(vec, mat):
    result = np.zeros(mat.shape[1])
    for j in range(mat.shape[1]):
        x = vec*mat[:,j]
        result[j] = x.sum()
    return result


@njit
def get_i_int_i_skill(worker_type):
    """
    Returns indices for the bool matrix "skill_mat"
    (which is used for masking different types of workers)
    and for the last two rows of the integer matrix "f_int_mat"
    (which give the number of open vacancies of firms for different types of workers).

    :param worker_type (str): either "unskilled" or "skilled"
    :return (int, int): worker type index for f_int_mat, index for skill_mat
    """

    i_int, i_skill = None, None

    if worker_type == "unskilled":
        i_int, i_skill = 4, 0
    elif worker_type == "skilled":
        i_int, i_skill = 5, 1
    else:
        print("Invalid worker type:")
        print("Please choose either 'unskilled' or 'skilled'")
        print()

    return int(i_int), int(i_skill)


################## HOUSEHOLDS ##################

def update_p_h(h_float_mat):
    # take households that could buy goods and divide their expenditure by number of goods...
    cond = h_float_mat[4] > 0
    h_float_mat[2, cond] = h_float_mat[6, cond] / h_float_mat[4, cond]

def fired_workers_loose_jobs(fired_time, emp_mat, h_float_mat, h_bool_mat):
    m = fired_time == 1
    emp_mat[:, m] = False
    h_float_mat[0, m] = 0
    h_bool_mat[1, m] = False
    fired_time[m] = 0

def update_d_w(h_float_mat, min_w, emp_arr, job_offers, sigma_w, tol):
    """
    Adjusts the wages demanded by households. If an Household had no job offer or is unemployed,
    she will adjust her demanded wage downwards. If the Household had a job offer and is employed she will
    adjust her demanded wage upwards.

    :param d_wages(ndarray): 1D array containing demanded wages of workers with 'float' type.
    :param emp_arr (ndarray): 1D array containing data with 'bool' type. 'True' if worker is employed.
    :param job_offers (ndarray): 1D array containing data with 'bool' type. 'True' if worker had job offer.
    :param sigma_w (float): adjustment parameter for wages.
    """
    d_wages = h_float_mat[1].copy()
    cond2 = np.logical_and(emp_arr, job_offers) # if employed AND job offer
    cond1 = np.invert(cond2) # if unemployed or no job offer
    rand_num = np.abs(rd.randn(np.sum(cond1)))
    d_wages[cond1] = d_wages[cond1] * (1-rand_num*sigma_w)
    rand_num = np.abs(rd.randn(np.sum(cond2)))
    d_wages[cond2] = d_wages[cond2] * (1+rand_num*sigma_w)
    lb = np.maximum(min_w, 0.1)
    h_float_mat[1] = np.maximum(d_wages, lb) # minimum wage or constant (exogenous) lower bound


def update_dc(h_float_mat, alpha_1, alpha_2):
    """
    Adjusts desired consumption level of households based on expected prices, current wages
    and available liquid assets.

    :param H (int): Number of household.
    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    :param alpha_1 (float): Propensity to consume income.
    :param alpha_2 (float): Propensity to consume wealth.
    """

    wages, dividends = h_float_mat[0], h_float_mat[7]
    p, wealth = h_float_mat[3], h_float_mat[9]
    I = np.maximum((wages + dividends - h_float_mat[8]), 0)
    d_c = alpha_1 * (I / p) + alpha_2 * (wealth / p)
    h_float_mat[5] = d_c


@njit
def get_app_mat(H, f_ids, N_app):
    """
    Returns a FxH (number of firms x number of households) application matrix
    that was initialized locally.

    :param H (int): number households
    :param f_ids (ndarray): 1D array containing the firm IDs with 'int' type.
    :param N_app (int): number of applications sent per household.
    :return (ndarray): application matrix, a 2D array containing data with 'bool' type.
    """
    app_mat = np.zeros((len(f_ids), H)) > 0
    firm_inds = lambda: rd.choice(f_ids, N_app, replace=False)

    for i in range(H):
        app_mat[firm_inds(), i] = 1

    return app_mat


@njit
def get_row_share_mat(mat):
    """
    Return a matrix containing the shares of the row sums of the input matrix.
    Used for getting the shares of firms owned by households in order to compute
    dividends.

    :param mat(ndarray): 2D array containing data with 'float' type.
    :return (ndarray): matrix (dtype = float) containing the shares of the row sums of the input matrix.
    """
    return (mat.transpose()/mat.sum(axis=1)).transpose()


@njit
def get_div_mat(F, H, equity_mat, dividends):
    """
    Returns a FxH matrix showing the amount of dividends each firm owes to each household.

    :param F (int): number of firms
    :param H (int): number of households
    :param equity_mat (ndarray): 2D array with 'float' type containing information about household investments.
    :param dividends (ndarray): 1D array (dtype = float) containing total amount of dividends each firm has to pay.
    :return (ndarray):
    """
    share_mat = get_row_share_mat(equity_mat)
    div_mat = np.zeros((F, H))
    for f_id in np.arange(F):
        div_mat[f_id, :] = share_mat[f_id] * dividends[f_id]
    return div_mat


@njit
def get_emp_ids(f_ids, h_ids, emp_mat):
    """
    Returns a 1D array containing the employer ID of each household. If the household
    is unemployed the corresponding array element takes the value '-1'.

    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param h_ids (ndarray): 1D array containing household IDs with 'int' type.
    :param emp_mat (ndarray): 2D (FxH) array containing data with 'bool' type. If True employed.
    :return (ndarray): containing the employer ID of each household with 'int' type.
    """

    emp_ids = np.full(len(h_ids), -1)
    for i in range(len(h_ids)):
        h_id = h_ids[i]
        if len(f_ids[emp_mat[:, h_id]]) > 0:
            emp_ids[i] = f_ids[emp_mat[:, h_id]][0]
    return emp_ids


def update_A_h(f_ids, emp_mat, h_ids, H, f_float_mat,
               h_float_mat, tol):
    """
    This function is updating the amount of liquid assets hold by households.

    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param emp_mat (ndarray): 2D (FxH) array containing data with 'bool' type. If True employed.
    :param h_ids (ndarray): 1D array containing household IDs with 'int' type.
    :param H (int): number of households
    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    """
    thetas, wages, dividends = f_float_mat[18], h_float_mat[0], h_float_mat[7]

    emp_ids = get_emp_ids(f_ids, h_ids, emp_mat)

    theta_arr = np.zeros(H)
    cond = emp_ids > -1 # choose employed households
    ids = emp_ids[cond] # get ids of employers

    # get theta values for the employed households
    theta_arr[cond] = thetas[ids]
    h_float_mat[9] += theta_arr * wages + dividends # add to liquid assets (index 9 in h_float_mat)
    h_float_mat[9, h_float_mat[9] < tol] = 0


@njit
def pay_refin_costs(weights, refin, h_float_mat, equity_mat, f_id):
    """
    Updates liquid assets and refinancing cost entry after households pay refinancing costs.

    :param weights (ndarray): 1D array containing weights of costs paid by households, with 'float' type.
    :param refin (float): refinancing costs that are going to be paid by households.
    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    :param equity_mat (ndarray): 2D array with 'float' type containing information about household investments.
    :param f_id (int): ID of firm that gets the refinancing by it's shareholders.
    """
    costs = weights * refin
    h_float_mat[9] -= costs # update liquid assets
    h_float_mat[8] += costs # update refinancing cost entry
    equity_mat[f_id] += costs # update equity matrix


@njit
def refin_firms(rand_ids, default_f_ids, f_float_mat, netA_arr, active_arr, wealth_tot, weights,
                default_arr, n_refin, t, h_float_mat, equity_mat, A_f):
    """
    Chooses a defaulted firm from a shuffled list and it's refinancing. The amount of liquid assets given
    to the defaulted firm is chosen as following: First sample an active firm from the 30th to the 50th
    percentile, then take the amount of it's liquid assets plus the debt owned by the defaulted firm.

    :param rand_ids (ndarray): 1D array containing indices of netA_arr in a random order (dtype = int).
    :param default_f_ids (ndarray): 1D array containing the IDs of defaulted firms.
    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param netA_arr (ndarray): 1D array containing amount of debts owned by defaulted firms (dtype = float).
    :param active_arr (ndarray): 1D array containing data with 'bool' type. If True the firm is active.
    :param wealth_tot (float): total amount of liquid assets owned by households.
    :param weights (float): 1D array containing weights of costs paid by households, with 'float' type.
    :param default_arr (ndarray): 1D array containing data with 'bool' type. If True the firm has defaulted.
    :param n_refin (ndarray): 1D (1xT) array with 'int' type data.
    :param t (int): number of the current period
    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    :param equity_mat (ndarray): 2D array with 'float' type containing information about household investments.
    """
    A_arr = f_float_mat[16]

    for i in rand_ids:

        f_id = default_f_ids[i]
        if active_arr.sum() > 0:
            active_id = sample_surviving_f_id(30, 50, active_arr, A_arr)
            refin = A_arr[active_id] - netA_arr[i] # amount of financing demanded by defaulted firm
        else:
            refin = A_f - netA_arr[i]
        refin = np.minimum(wealth_tot, refin)
        wealth_tot -= refin

        if refin > (-1)*netA_arr[i]:
            f_float_mat[16, f_id] += refin
            # f_float_mat[4, f_id] = f_float_mat[4, active_id]
            # f_float_mat[5, f_id] = f_float_mat[5, active_id]
            default_arr[f_id] = False
            n_refin[t] += 1

            # households pay refin costs, entry in equity matrix
            pay_refin_costs(weights, refin, h_float_mat, equity_mat, f_id)


def households_refin_firms(h_float_mat, f_float_mat, active_arr, default_arr, f_ids, n_refin, equity_mat, t, A_f):
    """
    Households refinance firms as shareholders.

    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param active_arr (ndarray): 1D array containing data with 'bool' type. If True the firm is active.
    :param default_arr (ndarray): 1D array containing data with 'bool' type. If True the firm has defaulted.
    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param n_refin (ndarray): 1D (1xT) array with 'int' type data.
    :param equity_mat (ndarray): 2D array with 'float' type containing information about household investments.
    :param t (int): number of current period
    """
    wealth_tot = np.sum(h_float_mat[9])
    x = np.sum(wealth_tot) > 0
    if x:
        weights = h_float_mat[9] / wealth_tot
        netA_arr = f_float_mat[16, default_arr] + f_float_mat[13, default_arr]
        ids = np.arange(len(netA_arr))
        rand_ids = rd.choice(ids, len(netA_arr), replace=False)
        default_f_ids = f_ids[default_arr]

        refin_firms(rand_ids, default_f_ids, f_float_mat, netA_arr,
                    active_arr, wealth_tot, weights, default_arr, n_refin, t,
                    h_float_mat, equity_mat, A_f)


################## FIRMS ##################


@njit
def Update_N(f_int_mat, emp_mat, skill_mat):
    """
    Updates number of current employees.

    :param f_int_mat (ndarray): 2D array containing firm data with 'integer' type.
    :param emp_mat (ndarray): 2D array containing data with 'bool' type.
    :param skill_mat (ndarray): 1D array containing data with 'bool' type.
    """
    # Update Nu
    f_int_mat[0, :] = np.sum(emp_mat[:, skill_mat[0]], axis=1)
    # Update Ns
    f_int_mat[2, :] = np.sum(emp_mat[:, skill_mat[1]], axis=1)


@njit
def update_W_fs(f_float_mat, h_float_mat, f_ids, skill_mat, emp_mat):
    """
    Update labor cost data of firms (average and total wage bills).

    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param skill_mat (ndarray): 2D array containing data with 'bool' type. used for masking skill types.
    :param emp_mat (ndarray): 2D array containing employment data with 'bool' type.
    """
    m_u, m_s = skill_mat[0], skill_mat[1]
    wages = h_float_mat[0]

    for i in f_ids:
        emp_row = emp_mat[i, :] #employees of firm i
        w_u_arr = wages[m_u][emp_row[m_u]]  # get wages of unskilled workers
        w_s_arr = wages[m_s][emp_row[m_s]]  # get wages of skilled workers

        # take averages wages

        if w_u_arr.size > 0:
            f_float_mat[2, i] = w_u_arr.mean() # Wu

        if w_s_arr.size > 0:
            f_float_mat[3, i] = w_s_arr.mean() # Ws

        # sum up wages
        f_float_mat[6, i] = w_u_arr.sum() # Wu_tot
        f_float_mat[7, i] = w_s_arr.sum() # Ws_tot


def CES_production(N_u, N_s, mu_u, mu_s, eta):
    """
    CES production function: Y = [(mu_u*N_u)^rho + (mu_s*N_s)^rho]^(1/rho)

    :param N_u (ndarray): 1D array containing number of unskilled workers with 'int' type.
    :param N_s (ndarray): 1D array containing number of skilled workers with 'int' type.
    :param mu_u (float): productivity parameter of unskilled workers
    :param mu_s (float): productivity parameter of skilled workers
    :param eta (float): elasitcity of substition
    :return (ndarray): 1D array containing number of goods that firms are going to produce with 'float' type.
    """
    rho = (eta - 1) / eta # get substition parameter rho
    X_1, X_2 = np.zeros(N_u.size), np.zeros(N_s.size)
    result = np.zeros(N_u.size)
    m_u, m_s = N_u > 0, N_s > 0
    X_1[m_u] = (mu_u * N_u[m_u]) ** rho
    X_2[m_s] = (mu_s * N_s[m_s]) ** rho
    m_res = X_1 + X_2 > 0
    result[m_res] = (X_1[m_res] + X_2[m_res]) ** (1 / rho)
    return result


def get_Omega(Wu_hat, Ws_hat, mu_u, mu_s, eta):
    """
    Return omega optimal ratio of skilled to unskilled workers

    :param Wu_hat (ndarray): 1D array containing estimated average costs for unskilled labor with 'float' type.
    :param Ws_hat (ndarray): 1D array containing estimated average costs for skilled labor with 'float' type.
    :param mu_u (float): productivity parameter of unskilled workers
    :param mu_s (float): productivity parameter of skilled workers
    :param eta (float): elasticity of substition
    :return Omega (ndarray): 1D array containing firm specific omegas with 'float' type.
    """

    rho = (eta - 1)/eta
    X_1, X_2, X_3 = Wu_hat / mu_u, Ws_hat / mu_s, mu_s / mu_u
    Omega = ((X_1/X_2)**(1/(rho-1)))*X_3
    return Omega


def update_d_y(f_float_mat, f_ids, default_arr, nu):
    """
    Updates the desired level of production 'd_y' for active firms. The desired level of
    production is based on the level of inventories 'inv' and on the expected level of sales
    's_hat', it also has a lower bound 'min_d_y'. 'min_d_y' is chosen such that at least one
    skilled worker and int(Omega) unskilled workers are needed, where Omega is the estimated optimal ratio of
    unskilled to skilled labor.

    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param default_arr (ndarray): 1D array containing data with 'bool' type. If True the firm has defaulted.
    :param mu_u (float): productivity parameter of unskilled workers
    :param mu_s (float): productivity parameter of skilled workers
    :param nu (float): firm's inventory target share
    :param (float): elasticity of substitution
    """

    s_hat, inv = f_float_mat[1], f_float_mat[12]
    desired_output = np.zeros(len(f_ids))
    active_inds = f_ids[np.invert(default_arr)]
    d_y = np.maximum(s_hat[active_inds] * (1 + nu) - inv[active_inds], 0)
    desired_output[active_inds] = d_y
    f_float_mat[9] = desired_output


def get_d_Ns_non_binding(d_y, Omega, mu_u, mu_s, eta):
    """
    Returns the desired number of skilled workers in the non binding case, i.e., if the
    firm has not enough liquid assets to produce the desired amount of goods.

    :param d_y (ndarray): 1D array containing desired level of outputs with 'float' type.
    :param Omega (ndarray): 1D array containing firm specific omegas with 'float' type.
    :param mu_u (float): productivity parameter of unskilled workers
    :param mu_s (float): productivity parameter of skilled workers
    :param eta (float): elasticity of substitution
    :return (ndarray): 1D array containing data with (int) type.
    """
    rho = (eta - 1)/eta
    X_1, X_2 = (d_y/mu_s), ((mu_u/mu_s)*Omega)**rho
    Ns = X_1*((1 + X_2)**(-1/rho))
    return Ns


def get_d_Ns_binding(A_f, Wu_hat, Ws_hat, Omega):
    """
    Returns the desired number of skilled workers in the binding case, i.e., if the
    firm has not enough liquid assets to produce the desired amount of goods.

    :param A_f (ndarray): 1D array containing amount of liquid assets of firms with 'float' type.
    :param Wu_hat (ndarray): 1D array containing estimated average costs for unskilled labor with 'float' type.
    :param Ws_hat (ndarray): 1D array containing estimated average costs for skilled labor with 'float' type.
    :param Omega (ndarray): 1D array containing firm specific omegas with 'float' type.
    :return (ndarray): 1D array containing data with (int) type.
    """
    X_1 = Wu_hat*Omega
    Ns = A_f*((X_1 + Ws_hat)**(-1))
    return Ns


def update_d_N(f_float_mat, f_int_mat, f_ids, mu_u, mu_s, eta, default_arr):
    """
    Updates the desired number of employees.

    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param f_int_mat (ndarray): 2D array containing firm data with 'int' type.
    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param mu_u (float): productivity parameter of unskilled workers
    :param mu_s (float): productivity parameter of skilled workers
    :param eta (float): elasticity of substitution
    :param default_arr (ndarray): 1D array containing data with 'boolean' type.
    """

    # compute 1. Case (non-binding)
    def_inds = f_ids[default_arr]
    active_inds = f_ids[np.invert(default_arr)]
    s_hat, W_u_hat, W_s_hat = f_float_mat[1, active_inds], f_float_mat[4, active_inds], f_float_mat[5,active_inds]
    d_y, A_f, prices = f_float_mat[9, active_inds], f_float_mat[16, active_inds], f_float_mat[17, active_inds]

    # check first case without binding budget constraint
    Omega = get_Omega(W_u_hat, W_s_hat, mu_u, mu_s, eta)
    d_Ns = np.round(get_d_Ns_non_binding(d_y, Omega, mu_u, mu_s, eta))
    d_Nu = np.round(d_Ns * Omega)

    # check feasibility
    C = W_u_hat * d_Nu + W_s_hat * d_Ns  # compute total costs
    cond1 = C > A_f + prices * s_hat

    if np.sum(cond1) > 0:
        d_Ns[cond1] = np.round(get_d_Ns_binding(A_f[cond1], W_u_hat[cond1], W_s_hat[cond1], Omega[cond1]))
        d_Ns[cond1] = d_Ns[cond1]
        d_Nu[cond1] = np.round(d_Ns[cond1] * Omega[cond1])

    cond2 = d_Ns == 0
    d_Ns[cond2] = 1
    d_Nu[cond2] = np.round(d_Ns[cond2]*Omega[cond2])
    d_y = CES_production(d_Nu, d_Ns, mu_u, mu_s, eta)

    # update matrix
    f_int_mat[1, active_inds] = d_Nu
    f_int_mat[3, active_inds] = d_Ns
    f_float_mat[9, active_inds] = d_y

    # defaulted firms don't produce
    f_int_mat[1, def_inds] = 0
    f_int_mat[3, def_inds] = 0

@njit
def update_v(f_int_mat):
    """
    Updates number of open vacancies.

    :param f_int_mat (ndarray): 2D array containing firm data with 'int' type.
    """

    f_int_mat[4] = f_int_mat[1] - f_int_mat[0] # vacancies for unskilled
    f_int_mat[5] = f_int_mat[3] - f_int_mat[2] # vacancies for skilled


def update_m(f_float_mat, nu, sigma_chi, default_arr, tol):
    """
    Updates fir mark-ups. If a firm has more inventory than it's target than it is
    going to adjust its mark-up upwards, otherwise downwards.

    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param f_ids (ndarray): 1D array containing firm IDs with 'int' type.
    :param nu (float): firm's inventory target share
    :param sigma_chi (float): adjustment parameter
    :param default_arr (ndarray): 1D array containing data with 'boolean' type.
    """
    m_arr = f_float_mat[11].copy()
    active_arr = np.invert(default_arr)

    cond = f_float_mat[12] < nu * f_float_mat[1]
    cond1 = np.logical_and(cond, active_arr)
    cond2 = np.invert(cond1)

    # less inventories than target and active:
    m_arr[cond1] = m_arr[cond1]*(1+np.abs(rd.randn(np.sum(cond1))) * sigma_chi)

    # more inventories than target or inactive:
    m_arr[cond2] = m_arr[cond2]*(1-np.abs(rd.randn(np.sum(cond2))) * sigma_chi)

    lb = 0.01 # lower bound for m
    m_arr = np.maximum(m_arr, lb)
    f_float_mat[11] = m_arr


def update_uc(f_float_mat, f_int_mat):
    """
    Updates unit costs (only if the firm has any employees).

    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param f_int_mat (ndarray): 2D array containing firm data with 'float' type.
    :param default_arr (ndarray): 1D array containing data with 'boolean' type.
    """
    d_Nu, d_Ns = f_int_mat[1], f_int_mat[3]
    Nu, Ns = f_int_mat[0], f_int_mat[2]
    W_u_hat, W_s_hat = f_float_mat[4], f_float_mat[5]
    d_y = f_float_mat[9]
    m = np.logical_or(Nu > 0, Ns > 0)
    C = W_u_hat[m] * d_Nu[m] + W_s_hat[m] * d_Ns[m]
    f_float_mat[10, m] = C / d_y[m] # update uc_arr


def update_delta(f_float_mat, nu, sigma_w, tol, default_arr):
    """
    Updates the delta (the share of firm profits distributed as dividends).
    If the firm plans to expand production in the subsequent period it will adjust delta downwards.
    if it plans to lower production in the subsequent period, it will adjust delta upwards.

    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param nu (float): firm's inventory target share
    :param sigma_w (float): adjustment parameter
    :param F (int): number of firms
    :return:
    """
    active_arr = np.invert(default_arr)
    s_hat, inv, d_y = f_float_mat[1], f_float_mat[12], f_float_mat[9]
    delta = f_float_mat[15].copy()

    cond1 = np.logical_and(s_hat * (1 + nu) - inv > d_y, active_arr)
    cond2 = np.logical_and(s_hat * (1 + nu) - inv <= d_y, active_arr)

    delta[cond1] = delta[cond1] * (1-np.abs(rd.randn(np.sum(cond1))) * sigma_w)
    delta[cond2] = delta[cond2] * (1+np.abs(rd.randn(np.sum(cond2))) * sigma_w)
    x = np.minimum(delta, 1)
    f_float_mat[15] = np.maximum(tol, x)



def distribute_dividends(f_float_mat, F, H, equity_mat, h_float_mat, tol):
    """
    Distributes dividends to shareholders.

    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    :param F (int): number of firms
    :param H (int): number of households
    :param equity_mat (ndarray): 2D array containing data wiht 'float' type.
    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    :param tol (float): tolerance level for setting low number to zero.
    """
    pi, div_f, delta = f_float_mat[13], f_float_mat[14], f_float_mat[15]

    # update div_f
    f_float_mat[14] = np.maximum(delta*pi, np.zeros(F))
    # div_mat: first axis -> dividends distributing firms, second axis -> dividends receiving households
    div_mat = get_div_mat(F, H, equity_mat, div_f)
    # array with how much dividends each household receives in total
    div_h_arr = div_mat.sum(axis=0)
    # firms pay dividends (substract from profits)
    pi -= div_mat.sum(axis=1)
    pi[np.abs(pi) < tol] = 0.0
    # households receive their dividends
    h_float_mat[7] = div_h_arr


def update_Af(f_float_mat):
    """
    Update liquid assets of firms.

    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type.
    """

    s, p = f_float_mat[0], f_float_mat[17]
    Wu_tot, Ws_tot = f_float_mat[6], f_float_mat[7]
    theta, div = f_float_mat[18], f_float_mat[14]
    revenue = s*p
    labor_costs = theta * (Wu_tot + Ws_tot) # theta*(Wu_tot + Ws_tot)
    dividends = div
    # update A_f
    f_float_mat[13] = revenue - labor_costs - dividends
    f_float_mat[16] += f_float_mat[13]


# def sample_surviving_f_id(lower_p, upper_p, active_arr, A_arr):
#     """
#     Samples a surviving firm ID.
#
#     :param lower_p (int): lower percentile
#     :param upper_p (int): upper percentile
#     :param active_arr (ndarray): 1D array containing data with 'boolean' type.
#     :param A_arr (ndarray): 1D array containing liquid assets of firms with 'float' type.
#     :return (int): sampled firm ID
#     """
#     percentile = rd.randint(lower_p, upper_p)  # choose percentile between lower_p and upper_p
#     x = np.percentile(A_arr[active_arr], percentile, interpolation='lower')
#     return np.where(A_arr == x)[0][0]


@njit
def sample_surviving_f_id(lower_p, upper_p, active_arr, A_arr):
    percentile = rd.randint(lower_p, upper_p)  # choose percentile between lower_p and upper_p
    x = np.percentile(A_arr[active_arr], percentile)
    arr = np.argsort(A_arr)
    i, val = 0, 0
    while val < x:
        f_id = arr[i]
        val = A_arr[f_id]
        i += 1
    f_id = arr[i - 2]
    return f_id


def def_firms_pay_remaining_bills(emp_mat, f_float_mat):
    """
    Defaulted firms that couldn't be refinanced pay their remaining wage bills.
    (their thetas get updated)

    :param emp_mat (ndarray): 2D array containing employment data with 'bool' typ.
    :param f_float_mat (ndarray): 2D array containing firm data with 'float' type,
    """

    s, A_f, p = f_float_mat[0], f_float_mat[16], f_float_mat[17]
    Wu_tot, Ws_tot = f_float_mat[6], f_float_mat[7]

    # choose firms that have employees
    m = emp_mat.sum(axis=1) > 0
    # compute theta for each firm
    LA = np.maximum(A_f[m] + s[m]*p[m], np.zeros(np.sum(m))) # liquid assets + revenue
    WB = Wu_tot[m] + Ws_tot[m] # wage bills (debt)
    # update thetas, index 18 in f_float_mat
    f_float_mat[18, m] = np.minimum(LA / WB, np.ones(np.sum(m)))


@njit
def def_firms_loose_employeees(default_arr, emp_mat, h_float_mat, h_bool_mat,
                               fired_time):
    """
    Defaulted firms loose their employees.

    :param default_arr (ndarray): 1D array containing data with 'bool' type.
    :param emp_mat (ndarray): 2D array containing employment data with 'bool' type.
    :param h_float_mat (ndarray): 2D array containing household data with 'float' type.
    :param h_bool_mat (ndarray): 2D array containing household data with 'bool' type.
    :param fired_time (ndarray): 1D array containing data with 'int' type.
    """

    def_ids = np.nonzero(default_arr)[0]
    wages = h_float_mat[0]
    job_offers = h_bool_mat[1]

    if default_arr.sum() > 0:
        for f_id in def_ids:
            emp_ids = np.nonzero(emp_mat[f_id, :])[0] # get employee IDs
            emp_mat[f_id, :] = False
            wages[emp_ids] = 0.0
            job_offers[emp_ids] = False
            fired_time[emp_ids] = 0


# get quantiles
def get_q_vals(q_arr, w_dist_mat):
    T = w_dist_mat.shape[0]
    q_mat = np.zeros((T, q_arr.size))
    for t in range(T):
        w_arr = w_dist_mat[t, :]
        w_dist = np.log(w_arr[w_arr>0])
        q_mat[t, :] = [np.quantile(w_dist, q, interpolation='lower') for q in q_arr]
    return q_mat

def get_q_vals2(q_arr, w_dist_mean):
    w_dist_mean = np.log(w_dist_mean[w_dist_mean>0])
    q_vals = np.array([np.quantile(w_dist_mean, q) for q in q_arr])
    return q_vals