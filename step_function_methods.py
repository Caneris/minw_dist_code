from tools import *
from labor_market import firms_fire_workers, lm_matching
from goods_market import goods_market_matching
from dataCollector import data_collector
from init_tools import *
from calibration import calibrate_model


def beginning_of_period(fired_time, emp_mat, h_float_mat, h_bool_mat, f_int_mat, skill_mat, default_arr):

    fired_time[h_bool_mat[1]] += np.ones(np.sum(h_bool_mat[1]), dtype=np.int)
    fired_workers_loose_jobs(fired_time, emp_mat, h_float_mat, h_bool_mat)
    Update_N(f_int_mat, emp_mat, skill_mat)


def household_decisions(H, h_float_mat, min_w, emp_arr, h_bool_mat, sigma_w, lambda_H, alpha_1, alpha_2, tol):

    # households update demanded wages
    update_d_w(h_float_mat, min_w, emp_arr, h_bool_mat[0], sigma_w, tol)
    h_float_mat[1] = np.maximum(h_float_mat[0], h_float_mat[1])
    # price expectations
    h_float_mat[3] = expectation(h_float_mat[2], h_float_mat[3], lambda_H)
    # desired consumption
    update_dc(h_float_mat, alpha_1, alpha_2)
    # reset job_offer, expenditure and consumption
    h_bool_mat[0] = False
    h_float_mat[6] = np.zeros(H) # expenditure
    h_float_mat[4] = np.zeros(H) # consumption
    h_float_mat[8] = np.zeros(H) # refinancing costs


def firm_decisions(min_w, f_ids, f_float_mat, f_int_mat, h_float_mat, skill_mat, emp_mat,
                   lambda_F, default_arr, mu_u, mu_s, nu, eta, sigma_m, tol):

    # firms compute average and total wages
    update_W_fs(f_float_mat, h_float_mat, f_ids, skill_mat, emp_mat)
    # unskilled wage expectations
    x = expectation(f_float_mat[2], f_float_mat[4], lambda_F)
    f_float_mat[4] = np.maximum(x, min_w)
    # skilled wage expectations
    x = expectation(f_float_mat[3], f_float_mat[5], lambda_F)
    f_float_mat[5] = np.maximum(x, min_w)
    # firms decide on desired production, labor demand and prices
    update_d_y(f_float_mat, f_ids, default_arr, nu)
    update_d_N(f_float_mat, f_int_mat, f_ids, mu_u, mu_s, eta, default_arr)
    update_v(f_int_mat)
    update_m(f_float_mat, nu, sigma_m, default_arr, tol)
    update_uc(f_float_mat, f_int_mat)
    # update prices
    f_float_mat[17] = f_float_mat[10]*(1 + f_float_mat[11]) # p = uc*(1+m)
    # reset firms' sales
    f_float_mat[0] = np.zeros(len(f_ids))


def run_labor_market(H, f_ids, h_ids, N_app, f_float_mat, f_int_mat, h_float_mat, h_bool_mat, skill_mat, min_w,
                     emp_mat, fired_time, lambda_LM, t):

    # households apply
    app_mat = get_app_mat(H, f_ids, N_app)
    # firms employ unskilled applicants
    firms_fire_workers("unskilled", f_ids, h_ids, f_int_mat, h_float_mat, h_bool_mat[1],
                       emp_mat, skill_mat)
    firms_fire_workers("skilled", f_ids, h_ids, f_int_mat, h_float_mat, h_bool_mat[1],
                       emp_mat, skill_mat)

    lm_matching("unskilled", f_ids, h_ids, h_float_mat, f_int_mat, app_mat, skill_mat,
                min_w, h_bool_mat, emp_mat, fired_time, lambda_LM, t)
    # firms employ skilled applicants
    lm_matching("skilled", f_ids, h_ids, h_float_mat, f_int_mat, app_mat, skill_mat,
                min_w, h_bool_mat, emp_mat, fired_time, lambda_LM, t)
    # firms update labor costs
    update_W_fs(f_float_mat, h_float_mat, f_ids, skill_mat, emp_mat)


def run_goods_market(h_ids, f_ids, mu_u, mu_s, eta, f_float_mat, f_int_mat, h_float_mat,
                     N_good, default_arr, tol, lambda_F):

    # firms produce goods
    f_float_mat[8] = CES_production(f_int_mat[0], f_int_mat[2], mu_u, mu_s, eta)
    goods_market_matching(h_float_mat, f_float_mat, h_ids, f_ids, N_good, tol, default_arr)
    # choose firms that are either active or have sold their inventories despite bankruptcy
    mask = f_float_mat[0] > 0 # for the defaulted firm which have sold goods on the gm
    active_arr = np.invert(default_arr)
    m = np.logical_or(mask, active_arr)
    # update expected sales
    f_float_mat[1, m] = expectation(f_float_mat[0, m], f_float_mat[1, m], lambda_F)
    # update inventories, inv_{t} = inv_{t-1} + (y_t - s_t)
    f_float_mat[12, m] += f_float_mat[8, m] - f_float_mat[0, m]
    f_float_mat[12, f_float_mat[12] < tol] = 0
    # households update prices
    update_p_h(h_float_mat)


def firm_profits_and_dividends(F, H, nu, sigma_delta, f_float_mat, h_float_mat,
                               equity_mat, tol, default_arr):

    # update delta (share of profits distributed as dividends)
    update_delta(f_float_mat, nu, sigma_delta, tol, default_arr)
    # update pi: pi = p_f*s_f - Wu_tot - Ws_tot
    f_float_mat[13] = f_float_mat[17]*f_float_mat[0] - f_float_mat[6] - f_float_mat[7]
    # firms determine dividends and distribute to households:
    distribute_dividends(f_float_mat, F, H, equity_mat, h_float_mat, tol)


def refin_default_firms(f_ids, n_refin, equity_mat, f_float_mat, h_float_mat, default_arr, emp_mat, t, A_f):
    # households refinance firms
    active_arr = np.invert(default_arr)
    households_refin_firms(h_float_mat, f_float_mat, active_arr, default_arr, f_ids, n_refin, equity_mat, t, A_f)
    # defaulted firms pay remaining wage bills (update thetas)
    def_firms_pay_remaining_bills(emp_mat, f_float_mat)


def end_of_period(n_refin, f_float_mat, f_ids, emp_mat, h_ids, H, h_float_mat, default_arr,
                  h_bool_mat, fired_time, data_mat, t, skill_mat, tol, w_dist_mat):

    # firms and households update liquid assets
    update_Af(f_float_mat)
    # Households update_A_h
    update_A_h(f_ids, emp_mat, h_ids, H, f_float_mat, h_float_mat, tol)
    # defaulted firms loos employees
    def_firms_loose_employeees(default_arr, emp_mat, h_float_mat, h_bool_mat, fired_time)
    # collect data
    data_collector(n_refin, t, H, data_mat, emp_mat, f_float_mat, h_float_mat, default_arr, skill_mat, w_dist_mat)

    # print("p_h - p_h_hat: {}".format(np.sum((h_float_mat[2] - h_float_mat[3]) > 0)))


def step_function(alpha_1, alpha_2, F, H, f_float_mat, f_int_mat, h_float_mat, h_bool_mat,
                  default_arr, fired_time, emp_mat, skill_mat, min_w_par, lambda_F, lambda_H, mu_u,
                  mu_s, nu, eta, sigma_m, sigma_w, sigma_delta, equity_mat, tol, n_refin,
                  f_ids, h_ids, data_mat, t, N_app, N_good, lambda_LM, A_f, w_dist_mat):

    beginning_of_period(fired_time, emp_mat, h_float_mat, h_bool_mat, f_int_mat, skill_mat, default_arr)
    emp_arr = np.sum(emp_mat, axis=0, dtype=np.bool)
    min_w = min_w_par*np.median(h_float_mat[0, emp_arr])
    # update all wages
    x = np.maximum(h_float_mat[0, emp_arr], min_w)
    h_float_mat[0, emp_arr] = x


    household_decisions(H, h_float_mat, min_w, emp_arr, h_bool_mat, sigma_w, lambda_H, alpha_1, alpha_2, tol)
    firm_decisions(min_w, f_ids, f_float_mat, f_int_mat, h_float_mat, skill_mat, emp_mat, lambda_F, default_arr,
                   mu_u, mu_s, nu, eta, sigma_m, tol)

    run_labor_market(H, f_ids, h_ids, N_app, f_float_mat, f_int_mat, h_float_mat, h_bool_mat, skill_mat, min_w, emp_mat,
                     fired_time, lambda_LM, t)
    run_goods_market(h_ids, f_ids, mu_u, mu_s, eta, f_float_mat, f_int_mat, h_float_mat,
                     N_good, default_arr, tol, lambda_F)

    firm_profits_and_dividends(F, H, nu, sigma_delta, f_float_mat, h_float_mat, equity_mat, tol, default_arr)
    # households refinance firms
    default_arr[:] = f_float_mat[16] + f_float_mat[13] < 0
    refin_default_firms(f_ids, n_refin, equity_mat, f_float_mat, h_float_mat, default_arr, emp_mat, t, A_f)

    end_of_period(n_refin, f_float_mat, f_ids, emp_mat, h_ids, H, h_float_mat, default_arr, h_bool_mat, fired_time,
                  data_mat, t, skill_mat, tol, w_dist_mat)


def run(T = 1000, alpha_2 = 0.25, N_good = 6, m = 0.1, delta = 1, lambda_LM = 10,
        sigma_m = 0.35, sigma_w = 0.4, sigma_delta = 0.0001, nu = 0.1, u_r = 0.08, lambda_F = 0.5, lambda_H = 1.0,
        F = 160, H = 1000, N_app = 6, eta = 1.5, mu_u = 0.4, gamma_s = 0.4, min_w_par = 0.4, W_u = 1, Ah = 1, tol = 1e-14):

    mu_s, W_s, Af, uc, p, y_f, pi_f, div_h, div_f, c, alpha_1 = calibrate_model(H, F, Ah, u_r, mu_u, W_u,
                                                                                gamma_s, m, eta, delta, alpha_2)
    # H_u, H_s: Number of unskilled resp. skilled workers
    H_u = int(np.round(H*(1-gamma_s)))
    H_s = int(H - H_u)
    # agent ids:
    f_ids, h_ids = np.arange(F), np.arange(H)
    # Data-Matrix
    data_mat = np.zeros((23, T))
    w_dist_mat = np.zeros((T, H))
    # firm data
    default_arr = np.full(F, 0, dtype = np.bool)
    f_int_mat = np.zeros((6, F), dtype=np.int64)
    f_init_vals = np.array([y_f, y_f, W_u, W_s, W_u, W_s, 0.0, 0.0, y_f, y_f, uc, m, nu*y_f, pi_f, div_f, delta, Af, p, 1.0], dtype=np.float64)
    f_float_mat = init_float_mat(f_init_vals, F)

    # household data
    h_bool_mat = np.full((2, H), 0, dtype=np.bool)
    skill_mat = get_skill_mat(H, H_u)
    fired_time = np.zeros(H, dtype=np.int64)
    n_refin = np.zeros(T, dtype=np.int64)
    h_init_vals = np.array([0, 0, p, p, c, c, c*p, div_h, 0.0, Ah], dtype=np.float64)
    h_float_mat = init_float_mat(h_init_vals, H)

    # initiate employment situation by creating an employment FxH matrix
    # employed workers have value one
    emp_mat = init_emp_mat(F, H, u_r)
    emp_arr = np.sum(emp_mat, axis=0, dtype=np.bool)
    # initiate desired wages
    h_float_mat[1,:] = np.concatenate((np.full(H_u, W_u), np.full(H_s,W_s)))
    # initiate wages
    h_float_mat[0,:] = emp_arr*h_float_mat[1, :]
    # initiate job offers
    h_bool_mat[0, :] = emp_arr

    #initiate equity matrix
    equity_mat = np.tile(h_float_mat[9], (F,1))
    # Update current number of employees
    Update_N(f_int_mat, emp_mat, skill_mat)


    for t in range(T):
        step_function(alpha_1, alpha_2, F, H, f_float_mat, f_int_mat, h_float_mat, h_bool_mat, default_arr, fired_time,
                      emp_mat, skill_mat, min_w_par, lambda_F, lambda_H, mu_u, mu_s, nu, eta, sigma_m, sigma_w,
                      sigma_delta, equity_mat, tol, n_refin, f_ids, h_ids, data_mat, t, N_app, N_good, lambda_LM, Af,
                      w_dist_mat)


    # return data_mat[:,-periods:].mean(axis=1)
    return data_mat, w_dist_mat