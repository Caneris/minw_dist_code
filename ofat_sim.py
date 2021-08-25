import numpy as np
import numpy.random as rd
import concurrent.futures
import csv
from setting_generator import ind
from switcher_module import run_OFAT_ID
import time

def run_model(arg):
    ID, T, par_val, periods = arg
    data = run_OFAT_ID(ID, T, par_val)
    data = np.concatenate((data[:, -periods:].mean(axis=1), data[:, -periods:].std(axis=1)))
    data = np.insert(data, [0], par_val)
    return list(data)

def write_down_results(par_name, results):
    with open('ofat_results/{}_ofat.csv'.format(par_name), 'w', newline = '') as csvfile:
        filewriter = csv.writer(csvfile, delimiter = ',')
        filewriter.writerow([par_name,
                             "unemployment_rate", "nominal_GDP", "real_GDP", "mean_price",
                             "mean_wage", "median_real_wage", "mean_real_wage", "gini_coefficient",
                             "share_of_inactive", "share_of_refinanced", "unskilled_wage", "skilled_wages",
                             "wage_ratio_s_to_u", "C", "DC", "total_expenditure", "A_ratio_FtoH",
                             "mean_m", "mean_uc", "mean_delta", "tot_inv", "ur_unskilled",
                             "ur_skilled",
                             "unemployment_rate_sd", "nominal_GDP_sd", "real_GDP_sd", "mean_price_sd",
                             "mean_wage_sd", "median_real_wage_sd", "mean_real_wage_sd", "gini_coefficient_sd",
                             "share_of_inactive_sd", "share_of_refinanced_sd", "unskilled_wage_sd", "skilled_wages_sd",
                             "wage_ratio_s_to_u_sd", "C_sd", "DC_sd", "total_expenditure_sd", "A_ratio_FtoH_sd",
                             "mean_m_sd", "mean_uc_sd", "mean_delta_sd", "tot_inv_sd", "ur_unskilled_sd",
                             "ur_skilled_sd"])
        filewriter.writerows(results)

def run_ofat(ofat_arg):
    ID, NC, T, vals, par_name, periods = ofat_arg
    seq = rd.permutation(NC)
    args = [(ID, T, vals[ind(vals, seq[j])], periods) for j in range(NC)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = np.array(list(executor.map(run_model, args)))
    write_down_results(par_name, results)
    print("Finished OFAT for {}.".format(par_name))

def run_ofat_simulations(T, periods, nsim, par_list, par_names):
    """
    Run a OFAT sensitivity analysis.

    :param T (int): number of periods per simulation run
    :param periods (int): number of periods per simulation run used for analysis (T - periods for burn-in phase)
    :param nsim (int): number of simulation per parameter value
    :param par_list (list): list of parameter values
    :param par_names (list): list of parameter names
    """

    # number of parameters
    NP = len(par_list)
    # number of cases per parameter array
    NC = [len(par_list[i])*nsim for i in range(len(par_list))]
    # create list of params for ofat simulations
    ofat_args = [(ID, NC[ID], T, par_list[ID], par_names[ID], periods) for ID in range(NP)]
    print("start OFAT simulations....")
    start_t = time.time()
    for ofat_arg in ofat_args:
        run_ofat(ofat_arg)
    end_t = time.time() - start_t
    print("{} simulations took {} minutes".format(np.sum(NC), end_t/60))


if __name__ == '__main__':

    N_app_arr = np.array([2, 3, 5, 8, 12]).astype(int)
    N_good_arr = np.array([2, 3, 5, 8, 12]).astype(int)
    lambda_LM_arr = np.array([1, 3, 5, 10, 15])
    sigma_w_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    #    sigma_m_arr = np.array([0.3, 0.325, 0.35, 0.375, 0.4])
    min_w_par_arr = np.array([1e-14, 0.2, 0.4, 0.6, 0.8])

    par_list = [min_w_par_arr, N_good_arr, lambda_LM_arr, sigma_w_arr, N_app_arr]
    par_names = ["min_w_par", "N_good", "lambda_LM", "sigma_w", "N_app"]

    run_ofat_simulations(T = 1000, periods = 300, nsim = 50, par_list = par_list, par_names = par_names)